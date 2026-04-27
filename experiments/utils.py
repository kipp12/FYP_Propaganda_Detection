"""
Shared utilities for experiment scripts.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def parse_run_arg() -> int:
    """
    Parse the --run argument from the command line.

    Returns the run number (default 1). Used to distinguish repeated
    runs of the same model for computing mean and std across runs.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--run', type=int, default=1,
                        help='Run number (default: 1). Used to name the output file.')
    args, _ = parser.parse_known_args()
    return args.run


def save_results(task: str, model: str, run: int, metrics: dict) -> str:
    """
    Save experiment results to a JSON file.

    File is saved to results/{task}/{model}_run{run}.json relative to
    the current working directory (expected to be the project root).

    Args:
        task:    'si' or 'tc'
        model:   model name, e.g. 'naive', 'lr', 'bert_frozen', 'roberta_finetuned'
        run:     run number
        metrics: dict of metric name → float value

    Returns:
        Path to the saved file.
    """
    out_dir = os.path.join('results', task)
    os.makedirs(out_dir, exist_ok=True)

    payload = {'task': task, 'model': model, 'run': run, **metrics}

    out_path = os.path.join(out_dir, f'{model}_run{run}.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    return out_path


# Shortened labels for confusion matrix axes (full names are too long to display)
_CM_SHORT_LABELS = {
    'Loaded_Language':                      'Loaded Lang.',
    'Name_Calling,Labeling':               'Name Calling',
    'Repetition':                          'Repetition',
    'Doubt':                               'Doubt',
    'Exaggeration,Minimisation':           'Exaggeration',
    'Appeal_to_fear-prejudice':            'Appeal Fear',
    'Flag-Waving':                         'Flag-Waving',
    'Causal_Oversimplification':           'Causal Oversimp.',
    'Appeal_to_Authority':                 'Appeal Auth.',
    'Slogans':                             'Slogans',
    'Whataboutism,Straw_Men,Red_Herring':  'Whataboutism',
    'Black-and-White_Fallacy':             'Black-White',
    'Thought-terminating_Cliches':         'Thought-term.',
    'Bandwagon,Reductio_ad_hitlerum':      'Bandwagon',
}


def save_confusion_matrix(
    task: str,
    model: str,
    run: int,
    gold: list,
    preds: list,
    labels: list,
) -> str:
    """
    Compute a row-normalised confusion matrix and save it as a PNG heatmap.

    Each cell shows the proportion of true-class examples predicted as
    each class (rows sum to 1.0). This normalisation accounts for class
    imbalance — rare classes are not swamped by frequent ones.

    Args:
        task:   'si' or 'tc'
        model:  model name string
        run:    run number
        gold:   list of true label strings
        preds:  list of predicted label strings
        labels: ordered list of class names (sets row/column order)

    Returns:
        Path to the saved PNG file.
    """
    cm = confusion_matrix(gold, preds, labels=labels)

    # Row-normalise (avoid division by zero for empty classes)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums, where=row_sums != 0)

    short = [_CM_SHORT_LABELS.get(l, l) for l in labels]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('True', fontsize=11)
    ax.set_title(f'Confusion Matrix — {model} (run {run}, row-normalised)', fontsize=12)

    # Annotate cells with values ≥ 0.05 to avoid clutter
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            if val >= 0.05:
                colour = 'white' if val > 0.6 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=colour)

    plt.tight_layout()

    out_dir = os.path.join('results', task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{model}_cm_run{run}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path
