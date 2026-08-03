from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class FairnessPlots:
    """
    Generate fairness visualizations.
    """

    @staticmethod
    def plot_metric(
        by_group: pd.DataFrame,
        metric: str,
        output_path: Path,
    ):
        """
        Plot a fairness metric by protected group.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(figsize=(6, 4))

        plt.bar(
            by_group["Group"],
            by_group[metric],
        )

        plt.title(
            f"{metric} by Group"
        )

        plt.xlabel(
            "Protected Group"
        )

        plt.ylabel(
            metric
        )

        plt.grid(
            axis="y",
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    @staticmethod
    def generate_all(
        by_group: pd.DataFrame,
        output_dir: Path,
    ):
        """
        Generate all fairness charts.
        """

        metrics = [

            "Selection Rate",

            "Accuracy",

            "Precision",

            "Recall",

            "F1",

            "True Positive Rate",

            "False Positive Rate",

            "False Negative Rate",
        ]

        for metric in metrics:

            filename = (
                metric.lower()
                .replace(" ", "_")
                + ".png"
            )

            FairnessPlots.plot_metric(
                by_group,
                metric,
                output_dir / filename,
            )