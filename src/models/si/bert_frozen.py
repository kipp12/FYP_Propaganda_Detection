import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.data.bio import char_offsets_to_token_bio, token_bio_to_char_offsets
from src.evaluation.si_eval import evaluate_si


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
        patience: int = 3,
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
                outputs = self.model(**batch)
                loss = outputs.loss
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
