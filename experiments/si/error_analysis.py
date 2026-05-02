"""
SI Error Analysis
-----------------
Trains the fine-tuned RoBERTa SI model and categorises span-level errors
on the test set into three types:

  1. False negatives  — gold spans with no overlapping prediction
  2. False positives  — predicted spans with no overlapping gold span
  3. Boundary errors  — predicted spans that partially overlap a gold span
                        (IoU > 0 but < 1.0)

For each error, saves: article_id, article text, gold span text,
predicted span text, and the IoU overlap score.

Output: results/si/error_analysis.json

Run from the project root:
    python -m experiments.si.error_analysis
"""

import json
import os

from src.data.corpus import load_corpus
from src.data.splits import make_splits
from src.models.si.roberta_finetuned import FinetunedRoBERTaSI

DATA_DIR = os.getenv('DATA_DIR', 'data')


def _iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Intersection-over-union for two character-level spans."""
    intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
    if intersection == 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 0 else 0.0


def _best_overlap(span_start: int, span_end: int, candidates: list) -> float:
    """Return the highest IoU between a span and any candidate span."""
    if not candidates:
        return 0.0
    return max(_iou(span_start, span_end, c[0], c[1]) for c in candidates)


def analyse(articles: list, predictions: list) -> dict:
    """
    Categorise span prediction errors across a list of articles.

    Args:
        articles:    list of Article objects with gold si_spans
        predictions: list of lists of (start, end) tuples, one per article

    Returns:
        Dict with keys 'false_negatives', 'false_positives', 'boundary_errors',
        each containing a list of error dicts.
    """
    false_negatives = []
    false_positives = []
    boundary_errors = []

    for article, pred_spans in zip(articles, predictions):
        gold_spans = [(s.start, s.end) for s in article.si_spans]
        text = article.text

        # --- False negatives: gold spans missed entirely ---
        for g_start, g_end in gold_spans:
            best = _best_overlap(g_start, g_end, pred_spans)
            if best == 0.0:
                false_negatives.append({
                    'article_id':       article.article_id,
                    'gold_span':        text[g_start:g_end],
                    'predicted_span':   None,
                    'overlap':          0.0,
                    'article_text':     text,
                })

        # --- False positives and boundary errors: examine each prediction ---
        for p_start, p_end in pred_spans:
            best = _best_overlap(p_start, p_end, gold_spans)

            if best == 0.0:
                # No overlap with any gold span → false positive
                false_positives.append({
                    'article_id':       article.article_id,
                    'gold_span':        None,
                    'predicted_span':   text[p_start:p_end],
                    'overlap':          0.0,
                    'article_text':     text,
                })
            elif best < 1.0:
                # Partial overlap → boundary error; find the best-matching gold span
                best_gold = max(gold_spans, key=lambda g: _iou(p_start, p_end, g[0], g[1]))
                boundary_errors.append({
                    'article_id':       article.article_id,
                    'gold_span':        text[best_gold[0]:best_gold[1]],
                    'predicted_span':   text[p_start:p_end],
                    'overlap':          round(best, 4),
                    'article_text':     text,
                })

    return {
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'boundary_errors': boundary_errors,
    }


def main():
    print('=== SI Error Analysis (Fine-tuned RoBERTa) ===\n')

    articles = load_corpus('train', DATA_DIR)
    train, dev, test = make_splits(articles, seed=42)
    print(f'Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}\n')

    model = FinetunedRoBERTaSI()
    print(f'Device: {model.device}')
    print(f'Model : {FinetunedRoBERTaSI.MODEL_NAME}\n')
    model.fit(train, dev)

    predictions = model.predict(test)
    errors = analyse(test, predictions)

    fn = errors['false_negatives']
    fp = errors['false_positives']
    be = errors['boundary_errors']

    print(f'False negatives : {len(fn)}')
    print(f'False positives : {len(fp)}')
    print(f'Boundary errors : {len(be)}')
    print(f'Total errors    : {len(fn) + len(fp) + len(be)}\n')

    # Sample prints — first 3 of each type
    if fn:
        print('--- False Negative examples ---')
        for e in fn[:3]:
            print(f'  [{e["article_id"]}] "{e["gold_span"][:80]}"')
    if fp:
        print('\n--- False Positive examples ---')
        for e in fp[:3]:
            print(f'  [{e["article_id"]}] "{e["predicted_span"][:80]}"')
    if be:
        print('\n--- Boundary Error examples ---')
        for e in be[:3]:
            print(f'  [{e["article_id"]}] IoU={e["overlap"]:.3f}')
            print(f'    Gold : "{e["gold_span"][:80]}"')
            print(f'    Pred : "{e["predicted_span"][:80]}"')

    out_dir = os.path.join('results', 'si')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'error_analysis.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)

    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
