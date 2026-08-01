from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


class ModelEvaluator:
    """Evaluate binary classification models."""

    @staticmethod
    def ks_statistic(y_true, y_prob):
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

        return np.max(np.abs(df["cum_bad"] - df["cum_good"]))

    @staticmethod
    def evaluate(model, X_test, y_test):

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC AUC": auc,
            "Gini": (2 * auc) - 1,
            "KS": ModelEvaluator.ks_statistic(y_test, y_prob),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
        }

    @staticmethod
    def roc(y_test, y_prob):
        return roc_curve(y_test, y_prob)