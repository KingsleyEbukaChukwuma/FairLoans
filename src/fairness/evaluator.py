from __future__ import annotations

from configs.config import (
    BASELINE_DIR,
    BEST_GROUP_FILENAME,
    FAIRNESS_BY_GROUP_FILENAME,
    FAIRNESS_DIFFERENCE_FILENAME,
    FAIRNESS_METRICS_FILENAME,
    MITIGATED_DIR,
    WORST_GROUP_FILENAME,
)
from src.fairness.metrics import (
    compute_fairness_metrics,
    compute_metric_frame,
)
from src.fairness.plots import FairnessPlots
from src.models.save import (
    save_dataframe,
    save_json,
)


class FairnessEvaluator:
    """
    Generate a complete fairness evaluation report.
    """

    @staticmethod
    def evaluate(
        y_true,
        y_pred,
        sensitive_features,
    ):

        overall_metrics = compute_fairness_metrics(
            y_true,
            y_pred,
            sensitive_features,
        )

        metric_frame = compute_metric_frame(
            y_true,
            y_pred,
            sensitive_features,
        )

        by_group = metric_frame.by_group.reset_index()

        by_group.rename(
            columns={by_group.columns[0]: "Group"},
            inplace=True,
        )

        differences = (
            metric_frame.difference()
            .rename("Difference")
            .reset_index()
            .rename(columns={"index": "Metric"})
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

            raise ValueError(f"Unknown stage: {stage}")

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
