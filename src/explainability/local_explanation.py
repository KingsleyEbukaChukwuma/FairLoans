from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.configs.config import (
    EXPLAINABILITY_DIR,
    LOCAL_EXPLANATION_FILENAME,
)


class LocalExplanation:
    """
    Generate a local explanation for a single prediction.
    """

    @staticmethod
    def explain(
        explainer,
        shap_values,
        instance: int = 0,
    ) -> pd.DataFrame:
        """
        Return SHAP contributions for one observation.
        """

        values = shap_values.values[instance]

        explanation = pd.DataFrame(
            {
                "Feature": explainer.feature_names,
                "Feature Value": shap_values.data[instance],
                "SHAP Value": values,
                "Absolute SHAP": abs(values),
            }
        )

        explanation = explanation.sort_values(
            "Absolute SHAP",
            ascending=False,
        ).reset_index(drop=True)

        return explanation

    @staticmethod
    def save(
        explanation: pd.DataFrame,
    ) -> Path:
        """
        Save explanation as JSON.
        """

        EXPLAINABILITY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = EXPLAINABILITY_DIR / LOCAL_EXPLANATION_FILENAME

        explanation.to_json(
            path,
            orient="records",
            indent=4,
        )

        return path

    @classmethod
    def build(
        cls,
        explainer,
        shap_values,
        instance: int = 0,
    ) -> pd.DataFrame:
        """
        Build and save a local explanation.
        """

        explanation = cls.explain(
            explainer,
            shap_values,
            instance,
        )

        cls.save(
            explanation,
        )

        return explanation
