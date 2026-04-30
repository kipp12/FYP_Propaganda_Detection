"""
Shared utilities for experiment scripts.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


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


def save_errors(task: str, model: str, run: int, articles: list, gold: list, preds: list) -> str:
    """
    Save misclassified TC spans to a JSON file for error analysis.

    For every span where gold != pred, records the span text, its gold
    technique label, and the model's predicted label. The resulting file
    can be used to find representative examples of common confusions —
    e.g. to illustrate an off-diagonal cell in the confusion matrix.

    Args:
        task:     'tc'
        model:    model name string
        run:      run number
        articles: list of Article objects from the test split
        gold:     flat list of true technique label strings
        preds:    flat list of predicted technique label strings

    Returns:
        Path to the saved JSON file.
    """
    spans = [span for a in articles for span in a.tc_spans]
    texts = [a.text[span.start:span.end] for a in articles for span in a.tc_spans]

    errors = [
        {'text': text, 'gold': g, 'pred': p}
        for text, g, p in zip(texts, gold, preds)
        if g != p
    ]

    out_dir = os.path.join('results', task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{model}_errors_run{run}.json')
    with open(out_path, 'w') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

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


def save_classification_report_figure(
    task: str,
    model: str,
    run: int,
    gold: list,
    preds: list,
    labels: list,
) -> str:
    """
    Generate a per-class classification report heatmap and save as PNG.

    Shows precision, recall, and F1-score for each class as a colour-coded
    heatmap. Classes with no predicted examples show 0.0.

    Args:
        task:   'si' or 'tc'
        model:  model name string
        run:    run number
        gold:   list of true label strings
        preds:  list of predicted label strings
        labels: ordered list of class names

    Returns:
        Path to the saved PNG file.
    """
    report = classification_report(
        gold, preds, labels=labels, output_dict=True, zero_division=0
    )

    short = [_CM_SHORT_LABELS.get(l, l) for l in labels]
    metrics = ['precision', 'recall', 'f1-score']
    data = np.array([[report[l][m] for m in metrics] for l in labels])

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(3))
    ax.set_xticklabels(['Precision', 'Recall', 'F1'], fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(short, fontsize=9)
    ax.set_title(f'Per-class Report — {model} (run {run})', fontsize=12)

    # Annotate every cell with its value
    for i in range(len(labels)):
        for j in range(3):
            val = data[i, j]
            colour = 'white' if val < 0.25 or val > 0.75 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=colour)

    # Add support counts on the right
    supports = [report[l]['support'] for l in labels]
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels([f'n={int(s)}' for s in supports], fontsize=8)

    plt.tight_layout()

    out_dir = os.path.join('results', task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{model}_report_run{run}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return out_path
