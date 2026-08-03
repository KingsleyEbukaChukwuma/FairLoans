from __future__ import annotations

import pandas as pd

from configs.config import (
    MAX_ROC_AUC_DEGRADATION,
    MAX_F1_DEGRADATION,
)


class GovernanceReport:
    """
    Generate governance reports comparing the baseline
    and bias-mitigated models.
    """

    @staticmethod
    def generate(
        comparison: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate the full governance report table.
        """

        return comparison[
            [
                "Category",
                "Metric",
                "Before",
                "After",
                "Change",
                "Outcome",
            ]
        ].copy()

    @staticmethod
    def executive_summary(
        comparison: pd.DataFrame,
    ) -> dict:
        """
        Generate an evidence-based governance summary.
        """

        comparison = comparison.copy()

        fairness = comparison[
            comparison["Category"] == "Fairness"
        ]

        performance = comparison[
            comparison["Category"] == "Performance"
        ]

        comparison = comparison.set_index("Metric")

        def get(metric: str):

            row = comparison.loc[metric]

            return {
                "before": row["Before"],
                "after": row["After"],
                "change": row["Change"],
                "outcome": row["Outcome"],
            }

        roc = get("ROC AUC")
        f1 = get("F1")
        brier = get("Brier Score")

        dp = get(
            "Demographic Parity Difference"
        )

        eo = get(
            "Equal Opportunity Difference"
        )

        performance_summary = (

            f"ROC AUC {roc['outcome'].lower()} "

            f"from {roc['before']:.3f} "

            f"to {roc['after']:.3f} "

            f"({roc['change']:+.3f}). "

            f"F1 {f1['outcome'].lower()} "

            f"from {f1['before']:.3f} "

            f"to {f1['after']:.3f} "

            f"({f1['change']:+.3f})."

        )

        calibration_summary = (

            f"Brier Score {brier['outcome'].lower()} "

            f"from {brier['before']:.3f} "

            f"to {brier['after']:.3f} "

            f"({brier['change']:+.3f})."

        )

        fairness_summary = (

            f"Demographic Parity Difference "

            f"{dp['outcome'].lower()} "

            f"from {dp['before']:.3f} "

            f"to {dp['after']:.3f}. "

            f"Equal Opportunity Difference "

            f"{eo['outcome'].lower()} "

            f"from {eo['before']:.3f} "

            f"to {eo['after']:.3f}."

        )

        roc_drop = roc["before"] - roc["after"]

        f1_drop = f1["before"] - f1["after"]

        fairness_improved = (

            dp["outcome"] == "Improved"

            and

            eo["outcome"] == "Improved"

        )

        if (

            fairness_improved

            and

            roc_drop <= MAX_ROC_AUC_DEGRADATION

            and

            f1_drop <= MAX_F1_DEGRADATION

        ):

            recommendation = (

                "Adopt the mitigated model because "

                "fairness improved substantially while "

                "predictive performance remained within "

                "the acceptable governance thresholds."

            )

        else:

            recommendation = (

                "Retain the baseline model because "

                "the fairness improvements do not justify "

                "the observed degradation in predictive "

                "performance."

            )

        return {

            "Performance Summary":

                performance_summary,

            "Calibration Summary":

                calibration_summary,

            "Fairness Summary":

                fairness_summary,

            "Performance Metrics Improved":

                int(
                    (
                        performance["Outcome"]
                        == "Improved"
                    ).sum()
                ),

            "Performance Metrics Worse":

                int(
                    (
                        performance["Outcome"]
                        == "Worse"
                    ).sum()
                ),

            "Fairness Metrics Improved":

                int(
                    (
                        fairness["Outcome"]
                        == "Improved"
                    ).sum()
                ),

            "Fairness Metrics Worse":

                int(
                    (
                        fairness["Outcome"]
                        == "Worse"
                    ).sum()
                ),

            "Recommendation":

                recommendation,

        }