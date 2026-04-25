"""
Experiment: Fine-tuned RoBERTa TC
------------------------------------
Full RoBERTa encoder fine-tuned end-to-end with a linear warmup schedule.
Dev split is used for early stopping; final evaluation is on the test split.

Run from the project root:
    python -m experiments.tc.run_roberta [--run N]
"""

import os

from src.data.corpus import load_corpus
from src.data.splits import make_splits
from src.models.tc.roberta_finetuned import FinetunedRoBERTaTC
from src.evaluation.tc_eval import evaluate_tc
from experiments.utils import parse_run_arg, save_results

DATA_DIR = os.getenv('DATA_DIR', 'data')


def main():
    run = parse_run_arg()
    print(f'=== Fine-tuned RoBERTa TC (run {run}) ===\n')

    # Load and split data
    articles = load_corpus('train', DATA_DIR)
    train, dev, test = make_splits(articles, seed=42)
    print(f'Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}\n')

    # Train — dev is used internally for early stopping
    model = FinetunedRoBERTaTC()
    print(f'Device: {model.device}')
    print(f'Model : {FinetunedRoBERTaTC.MODEL_NAME}')
    print(f'Epochs (max): {model.epochs}, patience: {model.patience}, lr: {model.lr}\n')
    model.fit(train, dev)

    # Evaluate on test split
    gold   = [span.technique for a in test for span in a.tc_spans]
    preds  = model.predict_flat(test)
    result = evaluate_tc(gold, preds)

    print('\nTest results:')
    print(f'  Micro — P: {result["micro_precision"]:.4f}  R: {result["micro_recall"]:.4f}  F1: {result["micro_f1"]:.4f}')
    print(f'  Macro — P: {result["macro_precision"]:.4f}  R: {result["macro_recall"]:.4f}  F1: {result["macro_f1"]:.4f}')

    path = save_results('tc', 'roberta_finetuned', run, result)
    print(f'\nSaved → {path}')


if __name__ == '__main__':
    main()
