from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from configs.config import (
    BAR_IMPORTANCE_FILENAME,
    EXPLAINABILITY_DIR,
    GLOBAL_IMPORTANCE_FILENAME,
    SHAP_MAX_DISPLAY,
)


class GlobalFeatureImportance:
    """
    Generate global SHAP feature importance.
    """

    @staticmethod
    def importance_dataframe(
        explainer,
        shap_values,
    ) -> pd.DataFrame:
        """
        Compute mean absolute SHAP values for each feature.
        """

        importance = pd.DataFrame(
            {
                "Feature": explainer.feature_names,
                "Importance": (abs(shap_values.values).mean(axis=0)),
            }
        )

        importance = importance.sort_values("Importance", ascending=False).head(SHAP_MAX_DISPLAY).reset_index(drop=True)
        return importance

    @staticmethod
    def save_csv(
        importance: pd.DataFrame,
    ) -> Path:

        EXPLAINABILITY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = EXPLAINABILITY_DIR / GLOBAL_IMPORTANCE_FILENAME

        importance.to_csv(
            path,
            index=False,
        )

        return path

    @staticmethod
    def save_plot(
        shap_values,
    ) -> Path:
        """
        Save SHAP global importance bar plot.
        """

        EXPLAINABILITY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure()

        shap.plots.bar(
            shap_values,
            show=False,
        )

        path = EXPLAINABILITY_DIR / BAR_IMPORTANCE_FILENAME

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
        explainer,
        shap_values,
    ) -> pd.DataFrame:
        """
        Build and save global feature importance.
        """

        importance = cls.importance_dataframe(
            explainer,
            shap_values,
        )

        cls.save_csv(
            importance,
        )

        cls.save_plot(
            shap_values,
        )

        return importance
