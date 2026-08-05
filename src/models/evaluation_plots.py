from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from configs.config import (
    PERFORMANCE_DIR,
    ROC_CURVE_FILENAME,
    CONFUSION_MATRIX_FILENAME,
    CALIBRATION_CURVE_FILENAME,
    RELIABILITY_DIAGRAM_FILENAME,
)


class EvaluationPlots:
    """
    Generate model evaluation plots.
    """

    @staticmethod
    def roc_curve(
        y_true,
        y_prob,
    ):

        PERFORMANCE_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        RocCurveDisplay.from_predictions(
            y_true,
            y_prob,
        )

        plt.tight_layout()

        plt.savefig(
            PERFORMANCE_DIR
            / ROC_CURVE_FILENAME,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    @staticmethod
    def confusion_matrix(
        y_true,
        y_pred,
    ):

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
        )

        plt.tight_layout()

        plt.savefig(
            PERFORMANCE_DIR
            / CONFUSION_MATRIX_FILENAME,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    @staticmethod
    def calibration_curve(
        y_true,
        y_prob,
    ):

        prob_true, prob_pred = calibration_curve(
            y_true,
            y_prob,
            n_bins=10,
        )

        plt.figure(figsize=(6, 6))

        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
        )

        plt.plot(
            [0, 1],
            [0, 1],
            "--",
        )

        plt.xlabel(
            "Predicted Probability"
        )

        plt.ylabel(
            "Observed Frequency"
        )

        plt.title(
            "Calibration Curve"
        )

        plt.tight_layout()

        plt.savefig(
            PERFORMANCE_DIR
            / CALIBRATION_CURVE_FILENAME,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    @staticmethod
    def reliability_diagram(
        y_true,
        y_prob,
    ):

        plt.figure(figsize=(6, 6))

        plt.hist(
            y_prob,
            bins=10,
        )

        plt.xlabel(
            "Predicted Probability"
        )

        plt.ylabel(
            "Count"
        )

        plt.title(
            "Reliability Diagram"
        )

        plt.tight_layout()

        plt.savefig(
            PERFORMANCE_DIR
            / RELIABILITY_DIAGRAM_FILENAME,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    @classmethod
    def build(
        cls,
        y_true,
        y_pred,
        y_prob,
    ):

        cls.roc_curve(
            y_true,
            y_prob,
        )

        cls.confusion_matrix(
            y_true,
            y_pred,
        )

        cls.calibration_curve(
            y_true,
            y_prob,
        )

        cls.reliability_diagram(
            y_true,
            y_prob,
        )