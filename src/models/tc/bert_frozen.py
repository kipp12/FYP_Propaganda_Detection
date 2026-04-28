import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.utils.rnn import pad_sequence

from src.evaluation.tc_eval import evaluate_tc, TC_LABELS


def _compute_class_weights(dataset) -> torch.Tensor:
    """
    Compute per-class weights inversely proportional to span frequency.

    Returns a weight tensor of length 14 suitable for
    torch.nn.CrossEntropyLoss(weight=...).
    """
    counts = [0] * len(TC_LABELS)
    for label in dataset.labels:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (len(TC_LABELS) * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float)


# Map technique strings to integer indices and back
LABEL2ID = {label: i for i, label in enumerate(TC_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(TC_LABELS)}


class _TCDataset(Dataset):
    """
    Each example is a propaganda span tokenised together with its
    surrounding article context. The span text is passed as the primary
    sequence and the article text provides context via BERT's pair encoding.
    The [CLS] embedding will be used for classification.
    """

    def __init__(self, articles: list, tokenizer, max_length: int, has_labels: bool = True):
        self.examples = []
        self.labels = []

        for article in articles:
            for span in article.tc_spans:
                span_text = article.text[span.start:span.end]

                encoding = tokenizer(
                    span_text,
                    article.text,
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                )
                self.examples.append(encoding)

                if has_labels:
                    self.labels.append(LABEL2ID[span.technique])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = dict(self.examples[idx])
        if self.labels:
            item['labels'] = self.labels[idx]
        return item


def _collate(batch):
    """Pad a batch of variable-length encodings."""
    keys = batch[0].keys()
    result = {}
    for key in keys:
        tensors = [torch.tensor(b[key]) for b in batch]
        if key == 'labels':
            result[key] = torch.stack(tensors)
        else:
            result[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
    return result


class FrozenBERTTC:
    """
    Frozen BERT encoder for technique classification.

    BERT is used as a frozen feature extractor. Each propaganda span,
    together with its surrounding article context, is tokenised and passed
    through the frozen BERT encoder. The embedding of the [CLS] token is
    extracted as a fixed-length 768-dimensional summary representation and
    passed into a linear classification head predicting one of 14 technique
    classes. Only the classification head is updated during training.

    Class-weighted cross-entropy loss is used to counter the span-level
    class imbalance across the 14 technique classes.

    Early stopping is applied based on macro F1 on the development set.
    """

    MODEL_NAME = 'bert-base-uncased'

    def __init__(
        self,
        max_length: int = 512,
        batch_size: int = 32,
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
        based on macro F1 on dev_articles.

        Args:
            train_articles: list of Article objects with tc_spans
            dev_articles:   list of Article objects for early stopping
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=len(TC_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        # Freeze encoder — only train classifier head
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        self.model.to(self.device)

        train_dataset = _TCDataset(train_articles, self.tokenizer, self.max_length)
        dev_dataset   = _TCDataset(dev_articles,   self.tokenizer, self.max_length)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=_collate
        )

        # Weighted loss to address span-level class imbalance across 14 techniques
        class_weights = _compute_class_weights(train_dataset).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        print(f'Class weights (min: {class_weights.min():.3f}, max: {class_weights.max():.3f})')

        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )

        best_macro_f1  = -1.0
        best_state     = None
        patience_count = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                outputs = self.model(**batch)
                loss = loss_fn(outputs.logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluate on dev using macro F1
            gold  = [span.technique for a in dev_articles for span in a.tc_spans]
            preds = self.predict_flat(dev_articles)
            result = evaluate_tc(gold, preds)
            macro_f1 = result['macro_f1']

            print(f'Epoch {epoch + 1:2d} | loss {total_loss / len(train_loader):.4f} | dev macro F1 {macro_f1:.4f}')

            if macro_f1 > best_macro_f1:
                best_macro_f1  = macro_f1
                best_state     = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    print(f'Early stopping at epoch {epoch + 1} (best macro F1: {best_macro_f1:.4f})')
                    break

        self.model.load_state_dict(best_state)
        return self

    def predict(self, articles: list) -> list:
        """
        Predict technique labels for all TC spans in each article.

        Returns:
            List of lists of predicted technique strings,
            one list per article, aligned with article.tc_spans.
        """
        self.model.eval()
        dataset = _TCDataset(articles, self.tokenizer, self.max_length, has_labels=False)
        loader  = DataLoader(dataset, batch_size=self.batch_size, collate_fn=_collate)

        all_preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch).logits
                preds  = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend([ID2LABEL[p] for p in preds])

        # Re-group by article
        predictions = []
        idx = 0
        for article in articles:
            n = len(article.tc_spans)
            predictions.append(all_preds[idx:idx + n])
            idx += n

        return predictions

    def predict_flat(self, articles: list) -> list:
        """Flat list of predictions aligned with flat gold labels."""
        return [label for article_preds in self.predict(articles) for label in article_preds]
