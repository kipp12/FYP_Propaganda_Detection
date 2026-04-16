import pytest
from src.models.si.naive import NaiveSI
from src.models.tc.naive import NaiveTC
from src.evaluation.si_eval import evaluate_si
from src.evaluation.tc_eval import evaluate_tc


# ---------------------------------------------------------------------------
# NaiveSI
# ---------------------------------------------------------------------------

def test_naive_si_returns_empty_spans(articles):
    model = NaiveSI()
    preds = model.predict(articles)
    assert len(preds) == len(articles)
    assert all(p == [] for p in preds)


def test_naive_si_f1_is_zero(articles):
    model = NaiveSI()
    preds = model.predict(articles)
    result = evaluate_si(articles, preds)
    assert result['f1'] == 0.0


def test_naive_si_recall_is_zero(articles):
    model = NaiveSI()
    preds = model.predict(articles)
    result = evaluate_si(articles, preds)
    assert result['recall'] == 0.0


# ---------------------------------------------------------------------------
# NaiveTC
# ---------------------------------------------------------------------------

def test_naive_tc_majority_class(articles):
    model = NaiveTC().fit(articles)
    assert model.majority_class == 'Loaded_Language'


def test_naive_tc_predicts_majority_for_all(articles):
    model = NaiveTC().fit(articles)
    spans = [span for a in articles for span in a.tc_spans]
    preds = model.predict(spans)
    assert all(p == 'Loaded_Language' for p in preds)


def test_naive_tc_predict_before_fit_raises():
    model = NaiveTC()
    with pytest.raises(AssertionError):
        model.predict([])


def test_naive_tc_macro_f1_near_zero(articles):
    model = NaiveTC().fit(articles)
    gold  = [s.technique for a in articles for s in a.tc_spans]
    preds = model.predict(gold)
    result = evaluate_tc(gold, preds)
    assert result['macro_f1'] < 0.5
