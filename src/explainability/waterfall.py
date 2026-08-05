from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import shap

from configs.config import (
    EXPLAINABILITY_DIR,
    SHAP_MAX_DISPLAY,
    WATERFALL_FILENAME,
)


class WaterfallPlot:
    """
    Generate a SHAP waterfall plot for a single prediction.
    """

    @staticmethod
    def save(
        shap_values,
        instance: int = 0,
    ) -> Path:
        """
        Save a waterfall plot for one observation.
        """

        EXPLAINABILITY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(
            figsize=(10, 8),
        )

        shap.plots.waterfall(
            shap_values[instance],
            max_display=SHAP_MAX_DISPLAY,
            show=False,
        )

        path = EXPLAINABILITY_DIR / WATERFALL_FILENAME

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return path

    @classmethod
    def build(
        cls,
        shap_values,
        instance: int = 0,
    ) -> Path:
        """
        Build and save the waterfall plot.
        """

        return cls.save(
            shap_values,
            instance,
        )
