from __future__ import annotations

from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def false_positive_rate(
    y_true,
    y_pred,
):
    """
    False Positive Rate.
    """

    tn, fp, _, _ = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    denominator = fp + tn

    if denominator == 0:
        return 0.0

    return float(fp / denominator)


def false_negative_rate(
    y_true,
    y_pred,
):
    """
    False Negative Rate.
    """

    _, _, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    denominator = fn + tp

    if denominator == 0:
        return 0.0

    return float(fn / denominator)


def true_positive_rate(
    y_true,
    y_pred,
):
    """
    True Positive Rate (Recall).
    """

    _, _, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    denominator = tp + fn

    if denominator == 0:
        return 0.0

    return float(tp / denominator)


def compute_fairness_metrics(
    y_true,
    y_pred,
    sensitive_features,
):
    """
    Compute overall fairness metrics.
    """

    return {
        "Demographic Parity Difference": float(
            demographic_parity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
        ),
        "Demographic Parity Ratio": float(
            demographic_parity_ratio(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
        ),
        "Equal Opportunity Difference": float(
            equal_opportunity_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
        ),
        "Equalized Odds Difference": float(
            equalized_odds_difference(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
        ),
    }


def compute_metric_frame(
    y_true,
    y_pred,
    sensitive_features,
):
    """
    Compute group fairness metrics.
    """

    metrics = {
        "Count": count,
        "Accuracy": accuracy_score,
        "Precision": precision_score,
        "Recall": recall_score,
        "F1": f1_score,
        "Selection Rate": selection_rate,
        "True Positive Rate": true_positive_rate,
        "False Positive Rate": false_positive_rate,
        "False Negative Rate": false_negative_rate,
    }

    frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    return frame
