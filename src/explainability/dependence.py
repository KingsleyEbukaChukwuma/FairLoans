from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import shap

from src.configs.config import (
    DEPENDENCE_FILENAME,
    EXPLAINABILITY_DIR,
)


class DependencePlot:
    """
    Generate SHAP dependence plots.
    """

    @staticmethod
    def save(
        shap_values,
        feature: str,
    ) -> Path:
        """
        Save a SHAP dependence plot.
        """

        EXPLAINABILITY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        feature_names = list(shap_values.feature_names)

        if feature not in feature_names:
            raise ValueError(f"{feature} not found in SHAP feature names.")

        feature_index = feature_names.index(feature)

        plt.figure(
            figsize=(10, 7),
        )

        shap.plots.scatter(
            shap_values[:, feature_index],
            show=False,
        )

        plt.tight_layout()

        path = EXPLAINABILITY_DIR / DEPENDENCE_FILENAME.format(
            feature=feature,
        )

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
        feature: str,
    ) -> Path:
        """
        Build a dependence plot.
        """

        return cls.save(
            shap_values,
            feature,
        )
