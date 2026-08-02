from src.features.preprocessing import CreditPreprocessor
from src.models.build_pipeline import build_pipeline
from src.models.evaluate import ModelEvaluator


def test_evaluate_returns_metrics(sample_data):

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

    metrics = ModelEvaluator.evaluate(
        pipeline,
        X,
        y,
    )

    expected = [

        "Accuracy",

        "Balanced Accuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC AUC",

        "PR AUC",

        "Log Loss",

        "Brier Score",

        "MCC",

        "Cohen Kappa",

        "Gini",

        "KS",

        "Confusion Matrix",
    ]

    for metric in expected:

        assert metric in metrics


def test_auc_range(sample_data):

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

    metrics = ModelEvaluator.evaluate(
        pipeline,
        X,
        y,
    )

    assert 0 <= metrics["ROC AUC"] <= 1