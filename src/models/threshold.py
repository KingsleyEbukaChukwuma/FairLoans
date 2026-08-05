from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)


class DecisionThresholdOptimizer:
    """
    Learn and apply an optimal probability threshold for
    binary classification.
    """

    @staticmethod
    def optimize(
        y_true,
        y_prob,
        metric: str = "balanced_accuracy",
    ):
        """
        Search for the threshold that maximizes the chosen
        evaluation metric.

        Parameters
        ----------
        y_true
            True labels.

        y_prob
            Predicted probabilities for the positive class.

        metric
            One of:
                - balanced_accuracy
                - f1
                - mcc
        """

        thresholds = np.arange(
            0.01,
            1.00,
            0.01,
        )

        best_threshold = 0.50
        best_score = -1.0

        for threshold in thresholds:

            y_pred = (
                y_prob >= threshold
            ).astype(int)

            if metric == "balanced_accuracy":

                score = balanced_accuracy_score(
                    y_true,
                    y_pred,
                )

            elif metric == "f1":

                score = f1_score(
                    y_true,
                    y_pred,
                )

            elif metric == "mcc":

                score = matthews_corrcoef(
                    y_true,
                    y_pred,
                )

            else:

                raise ValueError(
                    f"Unsupported metric: {metric}"
                )

            if score > best_score:

                best_score = score
                best_threshold = threshold

        return (
            float(best_threshold),
            float(best_score),
        )

    @staticmethod
    def predict(
        y_prob,
        threshold: float,
    ):
        """
        Convert probabilities into predictions using
        a custom decision threshold.
        """

        return (
            y_prob >= threshold
        ).astype(int)