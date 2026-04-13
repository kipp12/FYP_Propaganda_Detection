from sklearn.metrics import f1_score, classification_report


def evaluate_tc(gold_labels: list, pred_labels: list, label_names: list = None) -> dict:
    """
    Evaluate technique classification predictions using micro and macro F1,
    the standard metrics for TC in imbalanced multi-class settings.

    Micro F1: aggregates TP/FP/FN across all classes — dominated by frequent
              classes (e.g. Loaded Language). Reflects overall accuracy.

    Macro F1: averages F1 per class unweighted — treats all 14 techniques
              equally regardless of frequency. Primary metric for TC.

    Args:
        gold_labels:  list of true technique label strings
        pred_labels:  list of predicted technique label strings
        label_names:  optional list of class names for the per-class report

    Returns:
        dict with keys 'micro_f1', 'macro_f1', and optionally 'report'
    """
    assert len(gold_labels) == len(pred_labels), (
        f"Length mismatch: gold={len(gold_labels)}, pred={len(pred_labels)}"
    )

    micro_f1 = f1_score(gold_labels, pred_labels, average='micro', zero_division=0)
    macro_f1 = f1_score(gold_labels, pred_labels, average='macro', zero_division=0)

    result = {'micro_f1': micro_f1, 'macro_f1': macro_f1}

    if label_names is not None:
        result['report'] = classification_report(
            gold_labels, pred_labels, labels=label_names, zero_division=0
        )

    return result


# Canonical list of 14 TC classes in the PTC corpus
TC_LABELS = [
    'Loaded_Language',
    'Name_Calling,Labeling',
    'Repetition',
    'Doubt',
    'Exaggeration,Minimisation',
    'Appeal_to_fear-prejudice',
    'Flag-Waving',
    'Causal_Oversimplification',
    'Appeal_to_Authority',
    'Slogans',
    'Whataboutism,Straw_Men,Red_Herring',
    'Black-and-White_Fallacy',
    'Thought-terminating_Cliches',
    'Bandwagon,Reductio_ad_hitlerum',
]
