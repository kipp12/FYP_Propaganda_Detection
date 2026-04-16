import pytest
from src.evaluation.tc_eval import evaluate_tc, TC_LABELS


def test_perfect_predictions():
    gold = ['Loaded_Language', 'Doubt', 'Flag-Waving']
    result = evaluate_tc(gold, gold)
    assert result['micro_f1'] == pytest.approx(1.0)
    assert result['macro_f1'] == pytest.approx(1.0)


def test_all_wrong():
    gold = ['Loaded_Language', 'Doubt']
    pred = ['Flag-Waving', 'Repetition']
    result = evaluate_tc(gold, pred)
    assert result['micro_f1'] == 0.0
    assert result['macro_f1'] == 0.0


def test_majority_class_micro_higher_than_macro():
    gold = ['Loaded_Language', 'Loaded_Language', 'Loaded_Language', 'Doubt']
    pred = ['Loaded_Language', 'Loaded_Language', 'Loaded_Language', 'Loaded_Language']
    result = evaluate_tc(gold, pred)
    assert result['micro_f1'] > result['macro_f1']


def test_with_per_class_report():
    gold = ['Loaded_Language', 'Doubt']
    pred = ['Loaded_Language', 'Doubt']
    result = evaluate_tc(gold, pred, label_names=TC_LABELS)
    assert 'report' in result
    assert 'Loaded_Language' in result['report']


def test_mismatched_lengths_raises():
    with pytest.raises(AssertionError):
        evaluate_tc(['Loaded_Language'], ['Doubt', 'Flag-Waving'])


def test_scores_bounded():
    gold = ['Loaded_Language', 'Doubt', 'Repetition']
    pred = ['Loaded_Language', 'Loaded_Language', 'Repetition']
    result = evaluate_tc(gold, pred)
    assert 0.0 <= result['micro_f1'] <= 1.0
    assert 0.0 <= result['macro_f1'] <= 1.0


def test_tc_labels_has_14_classes():
    assert len(TC_LABELS) == 14
