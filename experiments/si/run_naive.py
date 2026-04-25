"""
Experiment: Naive SI baseline
------------------------------
Predicts no propaganda spans for every article (all-O predictor).
Precision, recall, and F1 are all 0 by construction.

Run from the project root:
    python -m experiments.si.run_naive
"""

import os

from src.data.corpus import load_corpus
from src.data.splits import make_splits
from src.models.si.naive import NaiveSI
from src.evaluation.si_eval import evaluate_si

DATA_DIR = os.getenv('DATA_DIR', 'data')


def main():
    print('=== Naive SI Baseline ===\n')

    # Load and split data
    articles = load_corpus('train', DATA_DIR)
    train, dev, test = make_splits(articles, seed=42)
    print(f'Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}\n')

    # Train (no-op for naive baseline)
    model = NaiveSI()

    # Evaluate on test split
    preds  = model.predict(test)
    result = evaluate_si(test, preds)

    print('Test results:')
    print(f'  Precision : {result["precision"]:.4f}')
    print(f'  Recall    : {result["recall"]:.4f}')
    print(f'  F1        : {result["f1"]:.4f}')


if __name__ == '__main__':
    main()
