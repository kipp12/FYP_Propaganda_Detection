"""
Experiment: Frozen BERT TC
----------------------------
BERT encoder with frozen weights; only the linear classification head is trained.
Dev split is used for early stopping; final evaluation is on the test split.

Run from the project root:
    python -m experiments.tc.run_bert
"""

import os

from src.data.corpus import load_corpus
from src.data.splits import make_splits
from src.models.tc.bert_frozen import FrozenBERTTC
from src.evaluation.tc_eval import evaluate_tc

DATA_DIR = os.getenv('DATA_DIR', 'data')


def main():
    print('=== Frozen BERT TC ===\n')

    # Load and split data
    articles = load_corpus('train', DATA_DIR)
    train, dev, test = make_splits(articles, seed=42)
    print(f'Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}\n')

    # Train — dev is used internally for early stopping
    model = FrozenBERTTC()
    print(f'Device: {model.device}')
    print(f'Model : {FrozenBERTTC.MODEL_NAME}')
    print(f'Epochs (max): {model.epochs}, patience: {model.patience}\n')
    model.fit(train, dev)

    # Evaluate on test split
    gold   = [span.technique for a in test for span in a.tc_spans]
    preds  = model.predict_flat(test)
    result = evaluate_tc(gold, preds)

    print('\nTest results:')
    print(f'  Micro F1 : {result["micro_f1"]:.4f}')
    print(f'  Macro F1 : {result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
