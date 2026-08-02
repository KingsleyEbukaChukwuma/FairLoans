import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from configs.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    TARGET_COLUMN,
    TARGET_RAW_COLUMN,
    SENSITIVE_FEATURE,
    PERSONAL_STATUS_COLUMN,
)


class CreditPreprocessor:

    @staticmethod
    def extract_gender(df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df[SENSITIVE_FEATURE] = df[PERSONAL_STATUS_COLUMN].apply(
            lambda x: "male"
            if str(x).lower().startswith("male")
            else "female"
        )

        return df

    @staticmethod
    def create_target(df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        df[TARGET_COLUMN] = (
            df[TARGET_RAW_COLUMN] == "bad"
        ).astype(int)

        return df

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> None:
        """
        Validate the input dataset before preprocessing.
        """

        required_columns = MODEL_FEATURES + [TARGET_RAW_COLUMN]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        duplicate_columns = df.columns[df.columns.duplicated()]

        if not duplicate_columns.empty:
            raise ValueError(
                f"Duplicate columns detected: {list(duplicate_columns)}"
            )

        for column in NUMERIC_FEATURES:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise TypeError(
                    f"Column '{column}' must be numeric."
                )

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare the dataset for modelling.

        Returns
        -------
        X : pd.DataFrame
            Model features.

        y : pd.Series
            Binary target.

        sensitive : pd.Series
            Protected attribute used for fairness evaluation.
        """

        self.validate_schema(df)
        df = self.extract_gender(df)
        df = self.create_target(df)

        X = df[MODEL_FEATURES].copy()
        y = df[TARGET_COLUMN].copy()
        sensitive = df[SENSITIVE_FEATURE].copy()

        return X, y, sensitive

    @staticmethod
    def build_transformer() -> ColumnTransformer:
        """
        Build the preprocessing pipeline for numeric and categorical features.
        """

        numeric_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
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

        return ColumnTransformer(
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

        return transformer