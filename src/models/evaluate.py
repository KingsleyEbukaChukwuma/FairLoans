from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    brier_score_loss,
    roc_curve,
    confusion_matrix,
)


class ModelEvaluator:
    """Evaluate binary classification models."""

    @staticmethod
    def ks_statistic(y_true, y_prob):
        """
        Compute the Kolmogorov-Smirnov statistic.
        """

        df = pd.DataFrame(
            {
                "target": y_true,
                "prob": y_prob,
            }
        ).sort_values("prob")

        bad = df["target"].sum()
        good = len(df) - bad

        df["cum_bad"] = df["target"].cumsum() / bad
        df["cum_good"] = (1 - df["target"]).cumsum() / good

        return float(np.max(
            np.abs(
                df["cum_bad"] - df["cum_good"]
            ))
        )

    @staticmethod
    def classification_metrics(
        y_true,
        y_pred,
        y_prob,
    ):

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

        return {

            "Accuracy":
                accuracy_score(
                    y_true,
                    y_pred,
                ),

            "Balanced Accuracy":
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                ),

            "Precision":
                precision_score(
                    y_true,
                    y_pred,
                ),

            "Recall":
                recall_score(
                    y_true,
                    y_pred,
                ),

            "F1":
                f1_score(
                    y_true,
                    y_pred,
                ),

            "ROC AUC":
                auc,

            "PR AUC":
                average_precision_score(
                    y_true,
                    y_prob,
                ),

            "Log Loss":
                log_loss(
                    y_true,
                    y_prob,
                ),

            "Brier Score":
                brier_score_loss(
                    y_true,
                    y_prob,
                ),

            "MCC":
                matthews_corrcoef(
                    y_true,
                    y_pred,
                ),

            "Cohen Kappa":
                cohen_kappa_score(
                    y_true,
                    y_pred,
                ),
        }

    @staticmethod
    def credit_metrics(
        y_true,
        y_prob,
    ):

        auc = roc_auc_score(
            y_true,
            y_prob,
        )

        return {

            "Gini":
                (2 * auc) - 1,

            "KS":
                ModelEvaluator.ks_statistic(
                    y_true,
                    y_prob,
                ),

        }

    @staticmethod
    def evaluate(
        model,
        X_test,
        y_test,
    ):

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

        metrics = {}

        metrics.update(

            ModelEvaluator.classification_metrics(
                y_test,
                y_pred,
                y_prob,
            )

        )

        metrics.update(

            ModelEvaluator.credit_metrics(
                y_test,
                y_prob,
            )

        )

        metrics["Confusion Matrix"] = confusion_matrix(
            y_test,
            y_pred,
        ).tolist()

        return metrics

    @staticmethod
    def roc(
        y_test,
        y_prob,
    ):
        return roc_curve(
            y_test,
            y_prob,
        )