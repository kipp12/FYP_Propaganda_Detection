"""
Shared utilities for experiment scripts.
"""

import argparse
import json
import os


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
