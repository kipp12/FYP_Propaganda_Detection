import pytest
from src.evaluation.si_eval import evaluate_si, _merge_spans


# ---------------------------------------------------------------------------
# _merge_spans
# ---------------------------------------------------------------------------

def test_merge_spans_empty():
    assert _merge_spans([]) == []


def test_merge_spans_no_overlap():
    spans = [(0, 5), (10, 15)]
    assert _merge_spans(spans) == [(0, 5), (10, 15)]


def test_merge_spans_overlapping():
    spans = [(0, 10), (5, 15)]
    assert _merge_spans(spans) == [(0, 15)]


def test_merge_spans_adjacent():
    spans = [(0, 5), (5, 10)]
    assert _merge_spans(spans) == [(0, 10)]


def test_merge_spans_unsorted():
    spans = [(10, 20), (0, 5)]
    assert _merge_spans(spans) == [(0, 5), (10, 20)]


# ---------------------------------------------------------------------------
# evaluate_si
# ---------------------------------------------------------------------------

def test_perfect_predictions(simple_article):
    pred = [[(s.start, s.end) for s in simple_article.si_spans]]
    result = evaluate_si([simple_article], pred)
    assert result['precision'] == pytest.approx(1.0)
    assert result['recall']    == pytest.approx(1.0)
    assert result['f1']        == pytest.approx(1.0)


def test_all_o_baseline(simple_article):
    result = evaluate_si([simple_article], [[]])
    assert result['precision'] == 0.0
    assert result['recall']    == 0.0
    assert result['f1']        == 0.0


def test_partial_overlap_gives_partial_credit(simple_article):
    span = simple_article.si_spans[0]
    mid = (span.start + span.end) // 2
    pred = [[(span.start, mid)]]
    result = evaluate_si([simple_article], pred)
    assert 0.0 < result['f1'] < 1.0


def test_all_propaganda_recall_is_one(simple_article):
    pred = [[(0, len(simple_article.text))]]
    result = evaluate_si([simple_article], pred)
    assert result['recall'] == pytest.approx(1.0)
    assert result['precision'] < 1.0


def test_mismatched_lengths_raises(simple_article):
    with pytest.raises(AssertionError):
        evaluate_si([simple_article], [[], []])


def test_scores_bounded(simple_article):
    pred = [[(s.start, s.end) for s in simple_article.si_spans]]
    result = evaluate_si([simple_article], pred)
    for v in result.values():
        assert 0.0 <= v <= 1.0
