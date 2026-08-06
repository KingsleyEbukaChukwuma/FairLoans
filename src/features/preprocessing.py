from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from src.configs.config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PERSONAL_STATUS_COLUMN,
    SENSITIVE_FEATURE,
    TARGET_COLUMN,
    TARGET_RAW_COLUMN,
)


class CreditPreprocessor:
    """
    Prepare the German Credit dataset for
    training and inference.
    """

    def __init__(self):

        self.transformer = None

    @staticmethod
    def extract_gender(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extract gender from the personal status field.
        """

        df = df.copy()

        df[SENSITIVE_FEATURE] = df[PERSONAL_STATUS_COLUMN].apply(lambda x: "male" if str(x).lower().startswith("male") else "female")

        return df

    @staticmethod
    def create_target(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create the binary target variable.
        """

        df = df.copy()

        df[TARGET_COLUMN] = (df[TARGET_RAW_COLUMN] == "bad").astype(int)

        return df

    @staticmethod
    def validate_schema(
        df: pd.DataFrame,
    ) -> None:
        """
        Validate the dataset schema.
        """

        required_columns = MODEL_FEATURES + [TARGET_RAW_COLUMN]

        missing_columns = [column for column in required_columns if column not in df.columns]

        if missing_columns:

            raise ValueError(f"Missing required columns: {missing_columns}")

        duplicate_columns = df.columns[df.columns.duplicated()]

        if not duplicate_columns.empty:

            raise ValueError(f"Duplicate columns detected: {list(duplicate_columns)}")

        for column in NUMERIC_FEATURES:

            if not pd.api.types.is_numeric_dtype(df[column]):

                raise TypeError(f"Column '{column}' must be numeric.")

    def prepare_data(
        self,
        df: pd.DataFrame,
    ):
        """
        Prepare the modelling dataset.
        """

        self.validate_schema(df)

        df = self.extract_gender(df)

        df = self.create_target(df)

        X = df[MODEL_FEATURES].copy()

        y = df[TARGET_COLUMN].copy()

        sensitive = df[SENSITIVE_FEATURE].copy()

        return X, y, sensitive

    def build_transformer(
        self,
    ) -> ColumnTransformer:
        """
        Build the preprocessing pipeline.
        """

        numeric_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_transformer,
                    NUMERIC_FEATURES,
                ),
                (
                    "categorical",
                    categorical_transformer,
                    CATEGORICAL_FEATURES,
                ),
            ],
            remainder="drop",
        )

        transformer.set_output(
            transform="pandas",
        )

        self.transformer = transformer

        return transformer

    @staticmethod
    def get_input_schema(
        transformer: ColumnTransformer,
    ) -> dict:
        """
        Extract the application input schema
        from a fitted transformer.
        """

        if not hasattr(
            transformer,
            "transformers_",
        ):
            raise RuntimeError("Transformer must be fitted first.")

        encoder = transformer.named_transformers_["categorical"].named_steps["encoder"]

        schema = {
            "numeric": list(NUMERIC_FEATURES),
            "categorical": {},
        }

        for feature, categories in zip(
            CATEGORICAL_FEATURES,
            encoder.categories_,
        ):

            schema["categorical"][feature] = list(categories)

        return schema
