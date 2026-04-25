"""
Experiment: Logistic Regression TC
--------------------------------------
Multi-class technique classifier using TF-IDF span representations.
Trained on the training split, evaluated on the test split.

Run from the project root:
    python -m experiments.tc.run_lr
"""

import os

from src.data.corpus import load_corpus
from src.data.splits import make_splits
from src.models.tc.logistic_regression import LogisticRegressionTC
from src.evaluation.tc_eval import evaluate_tc

DATA_DIR = os.getenv('DATA_DIR', 'data')


def main():
    print('=== Logistic Regression TC ===\n')

    # Load and split data
    articles = load_corpus('train', DATA_DIR)
    train, dev, test = make_splits(articles, seed=42)
    print(f'Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}\n')

    # Train on training split
    model = LogisticRegressionTC()
    print('Training...')
    model.fit(train)

    # Evaluate on test split
    gold   = [span.technique for a in test for span in a.tc_spans]
    preds  = model.predict_flat(test)
    result = evaluate_tc(gold, preds)

    print('\nTest results:')
    print(f'  Micro F1 : {result["micro_f1"]:.4f}')
    print(f'  Macro F1 : {result["macro_f1"]:.4f}')


if __name__ == '__main__':
    main()
