from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.configs.config import (
    EXPLAINABILITY_DIR,
    INTERACTION_FILENAME,
    INTERACTION_VALUES_FILENAME,
    TOP_INTERACTIONS,
)
from src.models.save import (
    save_dataframe,
)


class InteractionPlot:
    """
    Generate SHAP interaction artifacts.

    Produces:

    - interaction_heatmap.png
    - interaction_values.csv
    """

    @staticmethod
    def interaction_matrix(
        explainer,
        interaction_values,
    ) -> pd.DataFrame:
        """
        Compute the mean absolute SHAP interaction matrix.
        """

        matrix = np.abs(interaction_values).mean(axis=0)

        return pd.DataFrame(
            matrix,
            index=explainer.feature_names,
            columns=explainer.feature_names,
        )

    @staticmethod
    def save_heatmap(
        matrix: pd.DataFrame,
    ) -> Path:

        EXPLAINABILITY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(
            figsize=(14, 12),
        )

        plt.imshow(
            matrix,
            aspect="auto",
        )

        plt.colorbar()

        plt.xticks(
            range(len(matrix.columns)),
            matrix.columns,
            rotation=90,
            fontsize=8,
        )

        plt.yticks(
            range(len(matrix.index)),
            matrix.index,
            fontsize=8,
        )

        plt.title("Mean Absolute SHAP Interaction Values")

        plt.tight_layout()

        path = EXPLAINABILITY_DIR / INTERACTION_FILENAME

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return path

    @staticmethod
    def save_ranking(
        matrix: pd.DataFrame,
    ):
        """
        Save the strongest feature interactions.
        """

        pairs = []

        n_features = len(matrix.columns)

        for i in range(n_features):

            for j in range(
                i + 1,
                n_features,
            ):

                pairs.append(
                    {
                        "Feature 1": matrix.index[i],
                        "Feature 2": matrix.columns[j],
                        "Mean Interaction": matrix.iloc[i, j],
                    }
                )

        ranking = (
            pd.DataFrame(
                pairs,
            )
            .sort_values(
                "Mean Interaction",
                ascending=False,
            )
            .head(
                TOP_INTERACTIONS,
            )
            .reset_index(
                drop=True,
            )
        )

        save_dataframe(
            ranking,
            EXPLAINABILITY_DIR / INTERACTION_VALUES_FILENAME,
        )

        return ranking

    @classmethod
    def build(
        cls,
        explainer,
        interaction_values,
    ):

        matrix = cls.interaction_matrix(
            explainer,
            interaction_values,
        )

        cls.save_heatmap(
            matrix,
        )

        cls.save_ranking(
            matrix,
        )

        return matrix
