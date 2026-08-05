from sklearn.pipeline import Pipeline

from src.models.registry import MODELS


def build_pipeline(preprocessor, model_name):

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", MODELS[model_name]),
        ]
    )
