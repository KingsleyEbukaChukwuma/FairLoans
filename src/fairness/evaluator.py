from __future__ import annotations

import pandas as pd

from src.fairness.metrics import (
    compute_fairness_metrics,
    compute_metric_frame,
)

from configs.config import (
    FAIRNESS_DIR,
    FAIRNESS_METRICS_FILENAME,
    FAIRNESS_BY_GROUP_FILENAME,
    FAIRNESS_DIFFERENCE_FILENAME,
    WORST_GROUP_FILENAME,
    BEST_GROUP_FILENAME,
    FAIRNESS_PLOTS_DIR,
    BASELINE_DIR,
    MITIGATED_DIR,
)

from src.models.save import (
    save_json,
    save_dataframe,
)

from src.fairness.plots import FairnessPlots


class FairnessEvaluator:
    """
    Generate a complete fairness evaluation report.
    """

    @staticmethod
    def evaluate(
        y_true,
        y_pred,
        y_prob,
        sensitive_features,
    ):

        overall_metrics = compute_fairness_metrics(
            y_true,
            y_pred,
            y_prob,
            sensitive_features,
        )

        metric_frame = compute_metric_frame(
            y_true,
            y_pred,
            y_prob,
            sensitive_features,
        )

        by_group = (
            metric_frame.by_group
            .reset_index()
        )

        by_group.rename(
            columns={
                by_group.columns[0]: "Group"
            },
            inplace=True,
        )

        differences = (
            metric_frame.difference()
            .rename("Difference")
            .reset_index()
            .rename(
                columns={
                    "index": "Metric"
                }
            )
        )

        return {

            "overall": overall_metrics,

            "by_group": by_group,

            "difference": differences,

            "worst_group": metric_frame.group_min().to_dict(),

            "best_group": metric_frame.group_max().to_dict(),
        }

    @staticmethod
    def save(
        report,
        stage: str,
    ):
        """
        Save fairness artifacts.

        Parameters
        ----------
        stage

            baseline
    
            mitigated
        """

        if stage == "baseline":

            output_dir = BASELINE_DIR

        elif stage == "mitigated":

            output_dir = MITIGATED_DIR

        else:

            raise ValueError(
                f"Unknown stage: {stage}"
            )

        save_json(
            report["overall"],
            output_dir / FAIRNESS_METRICS_FILENAME,
        )

        save_dataframe(
            report["by_group"],
            output_dir / FAIRNESS_BY_GROUP_FILENAME,
        )

        save_dataframe(
            report["difference"],
            output_dir / FAIRNESS_DIFFERENCE_FILENAME,
        )

        save_json(
            report["worst_group"],
            output_dir / WORST_GROUP_FILENAME,
        )

        save_json(
            report["best_group"],
            output_dir / BEST_GROUP_FILENAME,
        )

        FairnessPlots.generate_all(
            report["by_group"],
            output_dir / "plots",
        )