def evaluate_si(articles: list, predictions: list) -> dict:
    """
    Evaluate span identification using the official SemEval 2020 Task 11
    span-overlap F1 metric (Da San Martino et al., 2019).

    Precision: (1/|S|) * sum_s  |s ∩ union(T_d)| / |s|
    Recall:    (1/|T|) * sum_t  |t ∩ union(S_d)| / |t|

    Where S is the set of predicted spans and T the set of gold spans.
    Each term is in [0,1]: precision measures what fraction of each
    predicted span is covered by gold, recall measures what fraction of
    each gold span is covered by predictions. Overlapping spans on both
    sides are unioned before computation to prevent double-counting.

    Args:
        articles:    list of Article objects with gold si_spans
        predictions: list of lists of (start, end) character offset tuples,
                     one list per article, in the same order as articles

    Returns:
        dict with keys 'precision', 'recall', 'f1'
    """
    assert len(articles) == len(predictions), (
        f"Number of articles ({len(articles)}) must match predictions ({len(predictions)})"
    )

    prec_sum   = 0.0
    rec_sum    = 0.0
    total_pred = 0
    total_gold = 0

    for article, pred_spans in zip(articles, predictions):
        gold_spans = [(s.start, s.end) for s in article.si_spans]

        # Merge overlapping spans on both sides before computation
        merged_pred = _merge_spans(pred_spans)
        merged_gold = _merge_spans(gold_spans)

        total_pred += len(merged_pred)
        total_gold += len(merged_gold)

        # Precision: for each predicted span, fraction covered by gold
        for s_start, s_end in merged_pred:
            s_len = s_end - s_start
            if s_len > 0:
                overlap = _overlap_with_set(s_start, s_end, merged_gold)
                prec_sum += overlap / s_len

        # Recall: for each gold span, fraction covered by predictions
        for t_start, t_end in merged_gold:
            t_len = t_end - t_start
            if t_len > 0:
                overlap = _overlap_with_set(t_start, t_end, merged_pred)
                rec_sum += overlap / t_len

    precision = prec_sum / total_pred if total_pred > 0 else 0.0
    recall    = rec_sum  / total_gold if total_gold > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {'precision': precision, 'recall': recall, 'f1': f1}


def _overlap_with_set(start: int, end: int, spans: list) -> int:
    """Compute overlap between a single span and a list of merged spans."""
    total = 0
    for s, e in spans:
        total += max(0, min(end, e) - max(start, s))
    return total


def _merge_spans(spans: list) -> list:
    """
    Merge a list of (start, end) spans into non-overlapping spans.
    Returns sorted, merged spans.
    """
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda x: x[0])
    merged = [sorted_spans[0]]

    for start, end in sorted_spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged
