from pathlib import Path

import joblib
import pandas as pd

from configs.config import (
    FEATURE_NAMES_FILENAME,
    METRICS_FILENAME,
    MODEL_DIR,
    PIPELINE_FILENAME,
    TRAINING_SUMMARY_FILENAME,
)
from src.models.train import TrainingManager


def assert_exists(path: Path) -> None:
    """
    Raise an error if an expected artifact is missing.
    """

    assert path.exists(), f"Missing artifact: {path.name}"

    print(f"✓ {path.name}")


def main():

    print("=" * 60)
    print("Running FairLoans Smoke Test")
    print("=" * 60)

    #
    # 1. Train the complete pipeline
    #

    manager = TrainingManager()

    manager.run()

    #
    # 2. Verify artifacts
    #

    print("\nChecking saved artifacts...")

    pipeline_path = MODEL_DIR / PIPELINE_FILENAME
    feature_path = MODEL_DIR / FEATURE_NAMES_FILENAME
    metrics_path = MODEL_DIR / METRICS_FILENAME
    summary_path = MODEL_DIR / TRAINING_SUMMARY_FILENAME

    assert_exists(pipeline_path)
    assert_exists(feature_path)
    assert_exists(metrics_path)
    assert_exists(summary_path)

    #
    # 3. Load pipeline
    #

    print("\nLoading trained pipeline...")

    pipeline = joblib.load(pipeline_path)

    print("✓ Pipeline loaded")

    #
    # 4. Load feature names
    #

    feature_names = joblib.load(feature_path)

    print(f" Loaded {len(feature_names)} feature names")

    #
    # 5. Dummy prediction
    #

    dummy = pd.DataFrame([{feature: 0 for feature in feature_names}])

    try:

        pipeline.predict(dummy)

        pipeline.predict_proba(dummy)

        print("✓ Prediction successful")

    except Exception as e:

        raise RuntimeError(f"Prediction failed: {e}")

    print("\n" + "=" * 60)
    print("✓ SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":

    main()
