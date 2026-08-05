from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import shap

from configs.config import (
    BEESWARM_FILENAME,
    EXPLAINABILITY_DIR,
    SHAP_MAX_DISPLAY,
)


class BeeswarmPlot:
    """
    Generate and save a SHAP beeswarm plot.
    """

    @staticmethod
    def save(
        shap_values,
    ) -> Path:
        """
        Save the SHAP beeswarm plot.
        """

        EXPLAINABILITY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(
            figsize=(12, 8),
        )

        shap.plots.beeswarm(
            shap_values,
            max_display=SHAP_MAX_DISPLAY,
            show=False,
        )

        plt.tight_layout()

        path = EXPLAINABILITY_DIR / BEESWARM_FILENAME

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
    ) -> Path:
        """
        Build the beeswarm plot.
        """

        return cls.save(
            shap_values,
        )
