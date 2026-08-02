from sklearn.linear_model import LogisticRegression

from src.features.preprocessing import CreditPreprocessor
from src.models.build_pipeline import build_pipeline


def test_pipeline_fit(sample_data):

    preprocessor = CreditPreprocessor()

    X, y, _ = preprocessor.prepare_data(
        sample_data
    )

    transformer = preprocessor.build_transformer()

    pipeline = build_pipeline(
        transformer,
        "logistic",
    )

    pipeline.fit(
        X,
        y,
    )

    predictions = pipeline.predict(X)

    assert len(predictions) == len(y)


def test_pipeline_predict_proba(sample_data):

    preprocessor = CreditPreprocessor()

    X, y, _ = preprocessor.prepare_data(
        sample_data
    )

    transformer = preprocessor.build_transformer()

    pipeline = build_pipeline(
        transformer,
        "logistic",
    )

    pipeline.fit(
        X,
        y,
    )

    probabilities = pipeline.predict_proba(X)

    assert probabilities.shape[1] == 2