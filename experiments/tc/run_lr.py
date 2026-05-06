"""
Experiment: Logistic Regression TC
--------------------------------------
Multi-class technique classifier using TF-IDF span representations.
Trained on the training split, evaluated on the test split.

Run from the project root:
    python -m experiments.tc.run_lr [--run N]
"""

import os

from src.data.corpus import load_corpus
from src.data.splits import make_splits
from src.models.tc.logistic_regression import LogisticRegressionTC
from src.evaluation.tc_eval import evaluate_tc, TC_LABELS
from experiments.utils import parse_run_arg, save_results, save_confusion_matrix, save_classification_report_figure, save_errors

DATA_DIR = os.getenv('DATA_DIR', 'data')


def main():
    run = parse_run_arg()
    print(f'=== Logistic Regression TC (run {run}) ===\n')

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
    print(f'  Micro — P: {result["micro_precision"]:.4f}  R: {result["micro_recall"]:.4f}  F1: {result["micro_f1"]:.4f}')
    print(f'  Macro — P: {result["macro_precision"]:.4f}  R: {result["macro_recall"]:.4f}  F1: {result["macro_f1"]:.4f}')

    path = save_results('tc', 'lr', run, result)
    print(f'\nSaved → {path}')

    cm_path = save_confusion_matrix('tc', 'lr', run, gold, preds, TC_LABELS)
    print(f'Confusion matrix → {cm_path}')

    report_path = save_classification_report_figure('tc', 'lr', run, gold, preds, TC_LABELS)
    print(f'Classification report → {report_path}')

    errors_path = save_errors('tc', 'lr', run, test, gold, preds)
    print(f'Errors → {errors_path}')


if __name__ == '__main__':
    main()
