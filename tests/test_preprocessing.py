from src.features.preprocessing import CreditPreprocessor


def test_prepare_data(sample_data):

    preprocessor = CreditPreprocessor()

    X, y, sensitive = preprocessor.prepare_data(sample_data)

    assert X.shape[0] == len(sample_data)

    assert len(y) == len(sample_data)

    assert len(sensitive) == len(sample_data)

    assert "gender" in sensitive.name.lower()


def test_transformer_builds():

    preprocessor = CreditPreprocessor()

    transformer = preprocessor.build_transformer()

    assert transformer is not None
