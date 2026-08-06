from pathlib import Path

# =============================================================================
# Project
# =============================================================================

APP_NAME = "FairLoans"
VERSION = "1.0.0"

ROOT_DIR = Path(__file__).resolve().parents[2]

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
FAIRNESS_DIR = ARTIFACT_DIR / "fairness"
FAIRNESS_METRICS_FILENAME = "fairness_metrics.json"
FAIRNESS_BY_GROUP_FILENAME = "fairness_by_group.csv"
FAIRNESS_DIFFERENCE_FILENAME = "fairness_difference.csv"
WORST_GROUP_FILENAME = "worst_group.json"
BEST_GROUP_FILENAME = "best_group.json"
FAIRNESS_PLOTS_DIR = FAIRNESS_DIR / "plots"
BASELINE_DIR = FAIRNESS_DIR / "baseline"
MITIGATED_DIR = FAIRNESS_DIR / "mitigated"
COMPARISON_DIR = FAIRNESS_DIR / "comparison"
FAIRNESS_SUMMARY_FILENAME = "fairness_summary.json"
FAIRNESS_COMPARISON_FILENAME = "comparison.csv"
COMPARISON_SUMMARY_FILENAME = "comparison_summary.json"
GOVERNANCE_REPORT_FILENAME = "governance_report.csv"
EXECUTIVE_SUMMARY_FILENAME = "executive_summary.json"
INPUT_SCHEMA_FILENAME = "input_schema.json"

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

N_JOBS = 1

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

OPTUNA_TIMEOUT = 900  # seconds

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

THRESHOLD_OPTIMIZATION_METRIC = "balanced_accuracy"

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

# =============================================================================
# Fairness
# =============================================================================

FAIRNESS_CONSTRAINT = "equalized_odds"

FAIRNESS_OBJECTIVE = "balanced_accuracy_score"

FAIRNESS_MITIGATION_METHOD = "threshold_optimizer"

# =============================================================================
# Governance
# =============================================================================

MAX_ROC_AUC_DEGRADATION = 0.01

MAX_F1_DEGRADATION = 0.02

# =============================================================================
# Explainability
# =============================================================================

EXPLAINABILITY_DIR = ARTIFACT_DIR / "explainability"

GLOBAL_IMPORTANCE_FILENAME = "global_feature_importance.csv"

LOCAL_EXPLANATION_FILENAME = "local_explanation.json"

SHAP_VALUES_FILENAME = "shap_values.pkl"

BEESWARM_FILENAME = "beeswarm.png"

BAR_IMPORTANCE_FILENAME = "global_importance.png"

WATERFALL_FILENAME = "waterfall.png"

FORCE_PLOT_FILENAME = "force_plot.html"

INTERACTION_FILENAME = "interaction_heatmap.png"

INTERACTION_VALUES_FILENAME = "interaction_values.csv"

TOP_INTERACTIONS = 20

MAX_EXPLANATIONS = 100

SHAP_EXPLAINER_FILENAME = "shap_explainer.pkl"

SHAP_INTERACTION_VALUES_FILENAME = "shap_interaction_values.pkl"

SHAP_MAX_DISPLAY = 15

DEPENDENCE_FEATURES = [
    "numeric__age",
    "numeric__credit_amount",
    "numeric__duration",
]

DEPENDENCE_FILENAME = "dependence_{feature}.png"


# =============================================================================
# Performance
# =============================================================================

PERFORMANCE_DIR = ARTIFACT_DIR / "performance"

ROC_CURVE_FILENAME = "roc_curve.png"

CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"

CALIBRATION_CURVE_FILENAME = "calibration_curve.png"


# =============================================================================
# App Fields
# =============================================================================


# ============================================================
# Field Labels
# ============================================================

FIELD_LABELS = {
    # Financial
    "checking_status": "Checking Account Status",
    "credit_amount": "Credit Amount",
    "savings_status": "Savings Account Status",
    # Loan
    "duration": "Loan Duration (Months)",
    "purpose": "Loan Purpose",
    "installment_commitment": "Installment Commitment (1-4)",
    # Applicant
    "age": "Age (Years)",
    "employment": "Employment Duration",
    "personal_status": "Personal Status",
    "job": "Employment Type",
    "foreign_worker": "Foreign Worker",
    "own_telephone": "Telephone",
    # Credit
    "credit_history": "Credit History",
    "existing_credits": "Existing Credits",
    "other_payment_plans": "Other Payment Plans",
    # Residence
    "housing": "Housing",
    "property_magnitude": "Primary Asset",
    "residence_since": "Current Residence (Years)",
    "other_parties": "Co-applicant / Guarantor",
    "num_dependents": "Number of Dependents",
}


# ============================================================
# Numeric Input Configuration
# ============================================================

NUMERIC_CONFIG = {
    "age": {
        "min": 18,
        "max": 100,
        "value": 18,
        "step": 1,
    },
    "duration": {
        "min": 1,
        "max": 72,
        "value": 1,
        "step": 1,
    },
    "credit_amount": {
        "min": 100,
        "value": 1000,
        "step": 100,
    },
    "installment_commitment": {
        "min": 1,
        "max": 4,
        "value": 1,
        "step": 1,
    },
    "residence_since": {
        "min": 1,
        "max": 4,
        "value": 1,
        "step": 1,
    },
    "existing_credits": {
        "min": 1,
        "max": 4,
        "value": 1,
        "step": 1,
    },
    "num_dependents": {
        "min": 1,
        "max": 2,
        "value": 1,
        "step": 1,
    },
}


# ============================================================
# Display Names for Categories
# ============================================================

CATEGORY_DISPLAY_MAPS = {
    "checking_status": {
        "<0": "Negative Balance",
        "0<=X<200": "Low Balance (0 - 199)",
        ">=200": "High Balance (200+)",
        "no checking": "No Checking Account",
    },
    "credit_history": {
        "no credits/all paid": "No Previous Credit / All Paid",
        "all paid": "All Previous Credits Paid",
        "existing paid": "Existing Credits Paid on Time",
        "delayed previously": "Previously Delayed Payments",
        "critical/other existing credit": "Critical Credit History",
    },
    "purpose": {
        "new car": "Purchase New Car",
        "used car": "Purchase Used Car",
        "furniture/equipment": "Furniture / Equipment",
        "radio/tv": "Electronics",
        "domestic appliance": "Domestic Appliance",
        "repairs": "Repairs",
        "education": "Education",
        "business": "Business",
        "vacation": "Vacation",
        "retraining": "Retraining",
        "other": "Other",
    },
    "savings_status": {
        "<100": "Less than 100",
        "100<=X<500": "100 - 499",
        "500<=X<1000": "500 - 999",
        ">=1000": "1000 or More",
        "no known savings": "No Savings Account",
    },
    "employment": {
        "unemployed": "Unemployed",
        "<1": "Less than 1 Year",
        "1<=X<4": "1 to 3 Years",
        "4<=X<7": "4 to 6 Years",
        ">=7": "7 Years or More",
    },
    "personal_status": {
        "male single": "Single Male",
        "male mar/wid": "Married/Widowed Male",
        "male div/sep": "Divorced/Separated Male",
        "female div/dep/mar": "Female",
    },
    "other_parties": {
        "none": "None",
        "co applicant": "Co-applicant",
        "guarantor": "Guarantor",
    },
    "property_magnitude": {
        "real estate": "Real Estate",
        "life insurance": "Life Insurance",
        "car": "Vehicle",
        "no known property": "No Property",
    },
    "other_payment_plans": {
        "none": "None",
        "bank": "Bank",
        "stores": "Store",
    },
    "housing": {
        "own": "Own Home",
        "rent": "Rent",
        "for free": "Living Free",
    },
    "job": {
        "high qualif/self emp/mgmt": "Management / Self-employed",
        "skilled": "Skilled Worker",
        "unskilled resident": "Unskilled Worker (Resident)",
        "unemp/unskilled non res": "Unemployed / Non-resident",
    },
    "own_telephone": {
        "none": "No Telephone",
        "yes": "Own Telephone",
    },
    "foreign_worker": {
        "yes": "Yes",
        "no": "No",
    },
}


# ============================================================
# Dataset Visualization Configuration
# ============================================================


DEFAULT_PLOT_FEATURES = [
    "age",
    "credit_amount",
    "duration",
    "credit_history",
    "checking_status",
    "purpose",
]


HISTOGRAM_FEATURES = [
    "age",
    "credit_amount",
]


DISCRETE_NUMERIC_FEATURES = [
    "duration",
    "installment_commitment",
    "residence_since",
    "existing_credits",
    "num_dependents",
]


CATEGORY_ORDER = {
    "checking_status": [
        "Negative Balance",
        "Low Balance (0 - 199)",
        "High Balance (200+)",
        "No Checking Account",
    ],
    "savings_status": [
        "Less than 100",
        "100 - 499",
        "500 - 999",
        "1000 or More",
        "No Savings Account",
    ],
    "employment": [
        "Unemployed",
        "Less than 1 Year",
        "1 to 3 Years",
        "4 to 6 Years",
        "7 Years or More",
    ],
    "duration": sorted(range(1, 73)),
    "installment_commitment": [1, 2, 3, 4],
    "residence_since": [1, 2, 3, 4],
    "existing_credits": [1, 2, 3, 4],
    "num_dependents": [1, 2],
}
