import json
from functools import lru_cache

import pandas as pd

from configs.config import (
    BAR_IMPORTANCE_FILENAME,
    BASELINE_DIR,
    BEESWARM_FILENAME,
    CALIBRATION_CURVE_FILENAME,
    COMPARISON_DIR,
    CONFUSION_MATRIX_FILENAME,
    DATASET_FILENAME,
    EXECUTIVE_SUMMARY_FILENAME,
    EXPLAINABILITY_DIR,
    FAIRNESS_BY_GROUP_FILENAME,
    FAIRNESS_COMPARISON_FILENAME,
    FAIRNESS_DIR,
    GOVERNANCE_REPORT_FILENAME,
    INPUT_SCHEMA_FILENAME,
    INTERACTION_FILENAME,
    LOCAL_EXPLANATION_FILENAME,
    METRICS_FILENAME,
    MITIGATED_DIR,
    MODEL_DIR,
    PERFORMANCE_DIR,
    RAW_DATA_DIR,
    ROC_CURVE_FILENAME,
    THRESHOLD_FILENAME,
    TRAINING_SUMMARY_FILENAME,
    WATERFALL_FILENAME,
)
from src.data.loader import DataLoader
from src.inference.predictor import CreditPredictor


@lru_cache(maxsize=1)
def get_predictor():
    """
    Load the trained predictor once.
    """

    return CreditPredictor()


@lru_cache(maxsize=1)
def get_schema():
    """
    Load the saved input schema.
    """

    with open(
        MODEL_DIR / INPUT_SCHEMA_FILENAME,
        "r",
    ) as f:

        return json.load(f)


@lru_cache(maxsize=1)
def get_dataset():

    loader = DataLoader()

    return loader.load_arff(RAW_DATA_DIR / DATASET_FILENAME)


@lru_cache(maxsize=1)
def get_fairness_dashboard():
    """
    Load every artifact required by the
    Responsible AI dashboard.
    """

    with open(
        COMPARISON_DIR / EXECUTIVE_SUMMARY_FILENAME,
        "r",
    ) as f:

        summary = json.load(f)

    baseline = pd.read_csv(BASELINE_DIR / FAIRNESS_BY_GROUP_FILENAME)

    mitigated = pd.read_csv(MITIGATED_DIR / FAIRNESS_BY_GROUP_FILENAME)

    comparison = pd.read_csv(FAIRNESS_DIR / FAIRNESS_COMPARISON_FILENAME)

    governance = pd.read_csv(COMPARISON_DIR / GOVERNANCE_REPORT_FILENAME)

    return {
        "summary": summary,
        "baseline": baseline,
        "mitigated": mitigated,
        "comparison": comparison,
        "governance": governance,
    }


@lru_cache(maxsize=1)
def get_explainability_dashboard():
    """
    Load all explainability artifacts.
    """

    with open(
        EXPLAINABILITY_DIR / LOCAL_EXPLANATION_FILENAME,
        "r",
    ) as f:

        local = json.load(f)

    return {
        "global_importance": EXPLAINABILITY_DIR / BAR_IMPORTANCE_FILENAME,
        "beeswarm": EXPLAINABILITY_DIR / BEESWARM_FILENAME,
        "waterfall": EXPLAINABILITY_DIR / WATERFALL_FILENAME,
        "interaction": EXPLAINABILITY_DIR / INTERACTION_FILENAME,
        "local": local,
    }


@lru_cache(maxsize=1)
def get_performance_dashboard():
    """
    Load every artifact required by the
    Model Performance dashboard.
    """

    #
    # Training summary
    #

    with open(
        MODEL_DIR / TRAINING_SUMMARY_FILENAME,
        "r",
    ) as f:

        summary = json.load(f)

    #
    # Evaluation metrics
    #

    with open(
        MODEL_DIR / METRICS_FILENAME,
        "r",
    ) as f:

        metrics = json.load(f)

    #
    # Optimized decision threshold
    #

    with open(
        MODEL_DIR / THRESHOLD_FILENAME,
        "r",
    ) as f:

        threshold = json.load(f)["threshold"]

    return {
        "summary": summary,
        "metrics": metrics,
        "best_model": {
            "Model": metrics["Model"],
        },
        "overall_score": metrics["Overall Score"],
        "threshold": threshold,
        "roc": PERFORMANCE_DIR / ROC_CURVE_FILENAME,
        "confusion": PERFORMANCE_DIR / CONFUSION_MATRIX_FILENAME,
        "calibration": PERFORMANCE_DIR / CALIBRATION_CURVE_FILENAME,
    }
