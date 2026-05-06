# Propaganda Detection in News Articles

Final Year Project — Cardiff University, 2025/26  
SemEval 2020 Task 11: Detection of Propaganda Techniques in News Articles (PTC corpus)

## Overview

This project implements and evaluates models for two propaganda detection tasks:

- **Task 1 — Span Identification (SI):** Given a news article, identify the character-level spans that contain propaganda.
- **Task 2 — Technique Classification (TC):** Given a propaganda span, classify it into one of 14 propaganda techniques (e.g. Loaded Language, Appeal to Fear, Name Calling).

Four models are implemented for each task, ranging from simple baselines to fine-tuned transformer models.

## Project Structure

```
src/
  data/
    corpus.py         — Article/SISpan/TCSpan dataclasses and corpus loader
    splits.py         — Reproducible train/dev/test split (80/10/10)
    bio.py            — BIO tagging utilities for SI
  models/
    si/
      naive.py            — Predicts the entire article as one propaganda span
      logistic_regression.py — Sliding-window TF-IDF + logistic regression
      roberta_base.py     — Frozen RoBERTa encoder + linear head (BIO tagging)
      roberta_ft.py       — Full fine-tuned RoBERTa (BIO tagging)
    tc/
      naive.py            — Majority class baseline (Loaded Language)
      logistic_regression.py — TF-IDF span representation + logistic regression
      roberta_base.py     — Frozen RoBERTa encoder + linear classification head
      roberta_ft.py       — Full fine-tuned RoBERTa sequence classifier
  evaluation/
    si_eval.py        — Span-level precision, recall, F1 (partial overlap scoring)
    tc_eval.py        — Micro and macro precision, recall, F1

experiments/
  si/
    run_naive.py          — SI naive baseline
    run_lr.py             — SI logistic regression
    run_roberta_base.py   — SI frozen RoBERTa
    run_roberta_ft.py     — SI fine-tuned RoBERTa
    error_analysis.py     — Categorise SI errors (FN / FP / boundary errors)
  tc/
    run_naive.py          — TC naive baseline
    run_lr.py             — TC logistic regression
    run_roberta_base.py   — TC frozen RoBERTa
    run_roberta_ft.py     — TC fine-tuned RoBERTa
  utils.py              — Shared utilities (result saving, confusion matrix, etc.)

data/                   — PTC corpus (not included, see Data section below)
results/                — Output JSONs and figures (generated at runtime)
```

## Data

This project uses the PTC corpus from SemEval 2020 Task 11. The data is **not included** in this repository as it is subject to the competition's terms of use.

To obtain the data:
1. Register at https://propaganda.math.unipd.it/semeval2020task11/
2. Download the training set
3. Place the contents in the `data/` directory so it contains:
   - `data/train-articles/`
   - `data/train-labels-task1-span-identification/`
   - `data/train-labels-task2-technique-classification/`

## Setup

```bash
pip install -r requirements.txt
```

A GPU is strongly recommended for the RoBERTa models. The experiments were run on Google Colab with a T4 GPU.

## Running Experiments

All scripts are run as modules from the project root. Use `--run N` to set the run number (used to name output files for multiple runs).

**Span Identification:**
```bash
python -m experiments.si.run_naive        --run 1
python -m experiments.si.run_lr           --run 1
python -m experiments.si.run_roberta_base --run 1
python -m experiments.si.run_roberta_ft   --run 1
```

**Technique Classification:**
```bash
python -m experiments.tc.run_naive        --run 1
python -m experiments.tc.run_lr           --run 1
python -m experiments.tc.run_roberta_base --run 1
python -m experiments.tc.run_roberta_ft   --run 1
```

**SI Error Analysis:**
```bash
python -m experiments.si.error_analysis
```

Results are saved to `results/si/` and `results/tc/` as JSON files, with confusion matrix and classification report figures saved as PNGs.

## Models

| Model | SI F1 (mean) | TC Macro F1 (mean) |
|---|---|---|
| Naive baseline | 0.000 | 0.033 |
| Logistic Regression | 0.282 | 0.371 |
| Frozen RoBERTa | 0.276 | 0.089 |
| Fine-tuned RoBERTa | 0.324 | 0.454 |

All neural models use class-weighted cross-entropy loss to address class imbalance, with early stopping based on span-level F1 (SI) or macro F1 (TC) on the development set.
