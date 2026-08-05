from __future__ import annotations

from typing import ClassVar

import pandas as pd


class FairnessComparison:
    """
    Compare the original model with a bias-mitigated model.
    """

    # Higher values are better
    HIGHER_IS_BETTER: ClassVar[set[str]] = {
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC",
        "PR AUC",
        "MCC",
        "Cohen Kappa",
        "Gini",
        "KS",
        "Demographic Parity Ratio",
    }

    # Lower values are better
    LOWER_IS_BETTER: ClassVar[set[str]] = {
        "Log Loss",
        "Brier Score",
        "Demographic Parity Difference",
        "Equal Opportunity Difference",
        "Equalized Odds Difference",
    }

    @staticmethod
    def compare(
        baseline_performance: dict,
        mitigated_performance: dict,
        baseline_fairness: dict,
        mitigated_fairness: dict,
    ) -> pd.DataFrame:
        """
        Compare model performance before and after
        bias mitigation.
        """

        rows = []

        performance_metrics = [
            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC",
            "PR AUC",
            "Log Loss",
            "Brier Score",
            "MCC",
            "Cohen Kappa",
            "Gini",
            "KS",
        ]

        fairness_metrics = [
            "Demographic Parity Difference",
            "Demographic Parity Ratio",
            "Equal Opportunity Difference",
            "Equalized Odds Difference",
        ]

        for metric in performance_metrics:

            before = baseline_performance.get(metric)
            after = mitigated_performance.get(metric)

            rows.append(
                FairnessComparison._build_row(
                    "Performance",
                    metric,
                    before,
                    after,
                )
            )

        for metric in fairness_metrics:

            before = baseline_fairness.get(metric)
            after = mitigated_fairness.get(metric)

            rows.append(
                FairnessComparison._build_row(
                    "Fairness",
                    metric,
                    before,
                    after,
                )
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _build_row(
        category,
        metric,
        before,
        after,
    ):

        if before is None or after is None:

            return {
                "Category": category,
                "Metric": metric,
                "Before": before,
                "After": after,
                "Change": None,
                "Outcome": "Not Available",
            }

        change = after - before

        if abs(change) < 1e-6:

            outcome = "Unchanged"

        elif metric in FairnessComparison.HIGHER_IS_BETTER:

            outcome = "Improved" if after > before else "Worse"

        elif metric in FairnessComparison.LOWER_IS_BETTER:

            outcome = "Improved" if after < before else "Worse"

        else:

            outcome = "Unknown"

        return {
            "Category": category,
            "Metric": metric,
            "Before": before,
            "After": after,
            "Change": change,
            "Outcome": outcome,
        }

    @staticmethod
    def summary(
        comparison: pd.DataFrame,
    ) -> dict:
        """
        Summarize comparison results.
        """

        performance = comparison[comparison["Category"] == "Performance"]

        fairness = comparison[comparison["Category"] == "Fairness"]

        return {
            "Performance Metrics Improved": int(
                (performance["Outcome"] == "Improved").sum()
            ),
            "Performance Metrics Worse": int((performance["Outcome"] == "Worse").sum()),
            "Fairness Metrics Improved": int((fairness["Outcome"] == "Improved").sum()),
            "Fairness Metrics Worse": int((fairness["Outcome"] == "Worse").sum()),
        }
