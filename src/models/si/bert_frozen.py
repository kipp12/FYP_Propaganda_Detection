import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.data.bio import char_offsets_to_token_bio, token_bio_to_char_offsets
from src.evaluation.si_eval import evaluate_si


def _compute_class_weights(dataset) -> torch.Tensor:
    """
    Compute per-class weights inversely proportional to token frequency.

    Tokens labelled -100 (subword continuations and padding) are ignored.
    Returns a weight tensor [w_O, w_B, w_I] suitable for
    torch.nn.CrossEntropyLoss(weight=...).
    """
    counts = [0, 0, 0]  # O=0, B=1, I=2
    for example in dataset:
        for label in example['labels']:
            if label != -100:
                counts[label] += 1
    total = sum(counts)
    # balanced weighting: total / (num_classes * class_count)
    weights = [total / (3 * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float)


class _SIDataset(Dataset):
    """
    Tokenises each article and maps gold SI spans to token-level BIO labels.
    Articles longer than max_length are truncated.
    """

    def __init__(self, articles: list, tokenizer, max_length: int):
        self.examples = []
        self.offset_mappings = []
        self.articles = articles

        for article in articles:
            encoding = tokenizer(
                article.text,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                padding=False,
            )
            offset_mapping = encoding.pop('offset_mapping')
            labels = char_offsets_to_token_bio(
                article.si_spans, offset_mapping, len(article.text)
            )
            encoding['labels'] = labels
            self.examples.append(encoding)
            self.offset_mappings.append(offset_mapping)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class FrozenBERTSI:
    """
    Frozen BERT encoder for span identification.

    BERT is used purely as a feature extractor. The encoder weights are
    frozen throughout training and only the linear classification head is
    updated. Each article is tokenised using BERT's WordPiece tokeniser,
    gold character-level span annotations are mapped to token-level BIO
    tags via bio.py, and the full token sequence is passed through the
    frozen encoder. The resulting contextual embeddings are fed into a
    linear head that predicts a BIO tag per token.

    Class-weighted cross-entropy loss is used to counter the severe
    token-level imbalance between O (non-propaganda) and B/I tokens.
    Weights are computed from the training set and applied via a custom
    loss function rather than the model's built-in loss.

    Predicted BIO sequences are converted back to character offsets for
    evaluation against the gold standard.
    """

    MODEL_NAME = 'bert-base-uncased'

    def __init__(
        self,
        max_length: int = 512,
        batch_size: int = 8,
        lr: float = 1e-3,
        epochs: int = 20,
        patience: int = 5,
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = None

    def fit(self, train_articles: list, dev_articles: list):
        """
        Train the classification head on train_articles, with early stopping
        based on span-level F1 on dev_articles.

        Args:
            train_articles: list of Article objects with si_spans
            dev_articles:   list of Article objects for early stopping
        """
        # Build model and freeze encoder
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.MODEL_NAME, num_labels=3
        )
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        self.model.to(self.device)

        train_dataset = _SIDataset(train_articles, self.tokenizer, self.max_length)
        dev_dataset   = _SIDataset(dev_articles,   self.tokenizer, self.max_length)

        from transformers import DataCollatorForTokenClassification
        collator = DataCollatorForTokenClassification(self.tokenizer, label_pad_token_id=-100)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collator
        )

        # Weighted loss to address severe O vs B/I class imbalance
        class_weights = _compute_class_weights(train_dataset).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        print(f'Class weights — O: {class_weights[0]:.3f}, B: {class_weights[1]:.3f}, I: {class_weights[2]:.3f}')

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )

        best_f1       = -1.0
        best_state    = None
        patience_count = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                outputs = self.model(**batch)
                # Use weighted loss instead of the model's built-in unweighted loss
                logits = outputs.logits  # (batch, seq_len, num_labels)
                loss = loss_fn(logits.view(-1, 3), labels.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluate on dev using span-level F1
            dev_preds = self._predict_articles(dev_dataset)
            result    = evaluate_si(dev_articles, dev_preds)
            f1        = result['f1']

            print(f'Epoch {epoch + 1:2d} | loss {total_loss / len(train_loader):.4f} | dev F1 {f1:.4f}')

            if f1 > best_f1:
                best_f1        = f1
                best_state     = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    print(f'Early stopping at epoch {epoch + 1} (best dev F1: {best_f1:.4f})')
                    break

        self.model.load_state_dict(best_state)
        return self

    def predict(self, articles: list) -> list:
        """
        Predict propaganda spans for a list of articles.

        Args:
            articles: list of Article objects

        Returns:
            List of lists of (start, end) character offset tuples.
        """
        dataset = _SIDataset(articles, self.tokenizer, self.max_length)
        return self._predict_articles(dataset)

    # ------------------------------------------------------------------

    def _predict_articles(self, dataset: _SIDataset) -> list:
        """Run inference over a dataset and return span predictions."""
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i, example in enumerate(dataset):
                input_ids      = torch.tensor(example['input_ids']).unsqueeze(0).to(self.device)
                attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                token_labels = outputs.logits[0].argmax(dim=-1).cpu().tolist()

                spans = token_bio_to_char_offsets(token_labels, dataset.offset_mappings[i])
                predictions.append(spans)

        return predictions
