"""
Experiment: Frozen BERT SI
----------------------------
BERT encoder with frozen weights; only the linear classification head is trained.
Dev split is used for early stopping; final evaluation is on the test split.

Run from the project root:
    python -m experiments.si.run_bert [--run N]
"""

import os

from src.data.corpus import load_corpus
from src.data.splits import make_splits
from src.models.si.bert_frozen import FrozenBERTSI
from src.evaluation.si_eval import evaluate_si
from experiments.utils import parse_run_arg, save_results

DATA_DIR = os.getenv('DATA_DIR', 'data')


def main():
    run = parse_run_arg()
    print(f'=== Frozen BERT SI (run {run}) ===\n')

    # Load and split data
    articles = load_corpus('train', DATA_DIR)
    train, dev, test = make_splits(articles, seed=42)
    print(f'Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}\n')

    # Train — dev is used internally for early stopping
    model = FrozenBERTSI()
    print(f'Device: {model.device}')
    print(f'Model : {FrozenBERTSI.MODEL_NAME}')
    print(f'Epochs (max): {model.epochs}, patience: {model.patience}\n')
    model.fit(train, dev)

    # Evaluate on test split
    preds  = model.predict(test)
    result = evaluate_si(test, preds)

    print('\nTest results:')
    print(f'  Precision : {result["precision"]:.4f}')
    print(f'  Recall    : {result["recall"]:.4f}')
    print(f'  F1        : {result["f1"]:.4f}')

    path = save_results('si', 'bert_frozen', run, result)
    print(f'\nSaved → {path}')


if __name__ == '__main__':
    main()
