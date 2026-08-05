from __future__ import annotations

from typing import Any

import pandas as pd
import shap
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class SHAPExplainer:
    """
    Build and reuse a SHAP explainer for a trained pipeline.

    The explainer is created once and SHAP values are cached
    in memory to avoid repeated computation.
    """

    def __init__(
        self,
        pipeline: Pipeline,
    ):

        self.pipeline = pipeline

        self.preprocessor = None
        self.model = None
        self.explainer = None

        self._X_processed = None
        self._shap_values = None
        self._interaction_values = None

    def fit(
        self,
        X_train,
    ) -> SHAPExplainer:
        """
        Build the SHAP explainer from a fitted pipeline.
        """

        self.preprocessor = self.pipeline.named_steps["preprocessor"]

        self.model = self.pipeline.named_steps["classifier"]

        self._X_processed = self._transform_to_dataframe(X_train)

        if isinstance(
            self.model,
            (
                RandomForestClassifier,
                ExtraTreesClassifier,
                HistGradientBoostingClassifier,
                LGBMClassifier,
                CatBoostClassifier,
                ExplainableBoostingClassifier,
            ),
        ):

            self.explainer = shap.TreeExplainer(
                self.model,
                data=self._X_processed,
            )

        elif isinstance(
            self.model,
            LogisticRegression,
        ):

            self.explainer = shap.LinearExplainer(
                self.model,
                self._X_processed,
            )

        else:

            self.explainer = shap.Explainer(
                self.model.predict,
                self._X_processed,
            )

        return self

    def _transform_to_dataframe(
        self,
        X,
    ) -> pd.DataFrame:

        X_processed = self.preprocessor.transform(
            X,
        )

        if hasattr(
            X_processed,
            "toarray",
        ):
            X_processed = X_processed.toarray()

        return pd.DataFrame(
            X_processed,
            columns=self.feature_names,
        )

    def shap_values(
        self,
        X,
    ) -> Any:
        """
        Compute SHAP values once and cache them.
        """

        if self._shap_values is None:

            X_processed = self._transform_to_dataframe(X)

            self._shap_values = self.explainer(X_processed)

            self._shap_values.feature_names = list(self.feature_names)
        return self._shap_values

    def transform(
        self,
        X,
    ):
        """
        Transform features using the fitted preprocessor.
        """

        return self._transform_to_dataframe(
            X,
        )

    @property
    def feature_names(self):
        """
        Return transformed feature names.
        """

        return self.preprocessor.get_feature_names_out()

    @property
    def expected_value(self):
        """
        Return the SHAP expected value.

        Used by waterfall and force plots.
        """

        if hasattr(
            self.explainer,
            "expected_value",
        ):
            return self.explainer.expected_value

        return None

    def interaction_values(
        self,
        X,
    ):

        if self._interaction_values is not None:
            return self._interaction_values

        if not hasattr(
            self.explainer,
            "shap_interaction_values",
        ):
            return None

        X_processed = self._transform_to_dataframe(X)

        self._interaction_values = self.explainer.shap_interaction_values(
            X_processed,
        )

        return self._interaction_values

    def clear_cache(self):
        """
        Clear cached SHAP values.
        """

        self._shap_values = None
        self._interaction_values = None
        self._X_processed = None
