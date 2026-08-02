from pathlib import Path

# =============================================================================
# Project
# =============================================================================

APP_NAME = "FairLoans"
VERSION = "1.0.0"

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
ARTIFACT_DIR = ROOT_DIR / "artifacts"
STUDY_DIR = MODEL_DIR / "studies"
LOG_DIR = ROOT_DIR / "logs"
TRAINING_LOG = "training.log"
PREDICTION_LOG = "prediction.log"
METRICS_DIR = ARTIFACT_DIR / "metrics"

# =============================================================================
# Dataset
# =============================================================================

DATASET_NAME = "German Credit"

DATASET_FILENAME = "dataset_31_credit-g.arff"

TARGET_RAW_COLUMN = "class"
TARGET_COLUMN = "target"

SENSITIVE_FEATURE = "gender"
PERSONAL_STATUS_COLUMN = "personal_status"

POSITIVE_CLASS = "bad"
NEGATIVE_CLASS = "good"

# =============================================================================
# Feature Lists
# =============================================================================

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# =============================================================================
# Train / Test
# =============================================================================

TEST_SIZE = 0.30

RANDOM_STATE = 123

CV_FOLDS = 5

N_JOBS = -1

N_JOBS2 = 1

LOGISTIC_MAX_ITER = 3000

# =============================================================================
# Model Selection
# =============================================================================

MODEL_SELECTION_WEIGHTS = {
    "ROC AUC": 0.20,
    "KS": 0.15,
    "MCC": 0.15,
    "F1": 0.10,
    "PR AUC": 0.10,
    "Log Loss": 0.15,
    "Brier Score": 0.15,
}
# =============================================================================
# Hyperparameter Tuning
# =============================================================================

OPTUNA_TRIALS = 50

OPTUNA_TIMEOUT = 900      # seconds

OPTUNA_DIRECTION = "maximize"

OPTIMIZATION_METRIC = "roc_auc"

OPTUNA_RANDOM_STATE = RANDOM_STATE

# =============================================================================
# Calibration
# =============================================================================

CALIBRATION_METHOD = "sigmoid"

CALIBRATION_CV = 5

# =============================================================================
# Inference
# =============================================================================

APPROVAL_THRESHOLD = 0.50

HIGH_CONFIDENCE = 0.80

LOW_CONFIDENCE = 0.20

# =============================================================================
# Evaluation
# =============================================================================

PRIMARY_METRIC = "ROC AUC"

SECONDARY_METRIC = "F1"

# =============================================================================
# Persistence
# =============================================================================

PIPELINE_FILENAME = "fairloans_pipeline.joblib"

METRICS_FILENAME = "metrics.json"

FEATURE_NAMES_FILENAME = "feature_names.pkl"

TRAINING_SUMMARY_FILENAME = "training_summary.json"

RESULTS_FILENAME = "model_results.csv"

THRESHOLD_FILENAME = "threshold.json"

STUDY_FILENAME = "optuna_study.pkl"

# =============================================================================
# Logging
# =============================================================================

LOG_LEVEL = "INFO"

LOG_FILE = LOG_DIR / "training.log"


