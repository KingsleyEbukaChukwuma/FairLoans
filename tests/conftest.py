import pytest

from src.data.loader import DataLoader
from src.features.preprocessing import CreditPreprocessor
from configs.config import (
    RAW_DATA_DIR,
    DATASET_FILENAME,
    RANDOM_STATE,
)


@pytest.fixture(scope="session")
def sample_data():

    loader = DataLoader()

    preprocessor = CreditPreprocessor()

    df = loader.load_arff(
        RAW_DATA_DIR / DATASET_FILENAME
    )

    # Random sample of 50 rows
    df = df.sample(
        n=50,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return df


@pytest.fixture(scope="session")
def prepared_data(sample_data):

    preprocessor = CreditPreprocessor()

    return preprocessor.prepare_data(sample_data)