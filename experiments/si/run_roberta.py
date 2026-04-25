"""
Experiment: Fine-tuned RoBERTa SI
------------------------------------
Full RoBERTa encoder fine-tuned end-to-end with a linear warmup schedule.
Dev split is used for early stopping; final evaluation is on the test split.

Run from the project root:
    python -m experiments.si.run_roberta
"""

import os

from src.data.corpus import load_corpus
from src.data.splits import make_splits
from src.models.si.roberta_finetuned import FinetunedRoBERTaSI
from src.evaluation.si_eval import evaluate_si

DATA_DIR = os.getenv('DATA_DIR', 'data')


def main():
    print('=== Fine-tuned RoBERTa SI ===\n')

    # Load and split data
    articles = load_corpus('train', DATA_DIR)
    train, dev, test = make_splits(articles, seed=42)
    print(f'Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}\n')

    # Train — dev is used internally for early stopping
    model = FinetunedRoBERTaSI()
    print(f'Device: {model.device}')
    print(f'Model : {FinetunedRoBERTaSI.MODEL_NAME}')
    print(f'Epochs (max): {model.epochs}, patience: {model.patience}, lr: {model.lr}\n')
    model.fit(train, dev)

    # Evaluate on test split
    preds  = model.predict(test)
    result = evaluate_si(test, preds)

    print('\nTest results:')
    print(f'  Precision : {result["precision"]:.4f}')
    print(f'  Recall    : {result["recall"]:.4f}')
    print(f'  F1        : {result["f1"]:.4f}')


if __name__ == '__main__':
    main()
