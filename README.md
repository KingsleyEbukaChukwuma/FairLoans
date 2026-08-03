# FairLoans: Responsible AI for Credit Risk Assessment

## Overview

FairLoans is an end-to-end machine learning system for credit risk assessment built with a strong emphasis on **Responsible AI**, **model governance**, and **production-ready engineering practices**.

The project demonstrates the complete machine learning lifecycle, from data ingestion and preprocessing to hyperparameter optimization, calibration, fairness evaluation, bias mitigation, explainability, and governance reporting.

FairLoan evaluates models across multiple dimensions including accuracy, calibration, fairness, and operational governance.

---

## Key Features

### Machine Learning Pipeline

* Automated data loading and preprocessing
* Schema and data validation
* Feature engineering
* Train/test splitting
* Scikit-learn pipelines
* Multiple benchmark models
* Hyperparameter optimization using Optuna
* Probability calibration
* Threshold optimization
* Model selection based on multiple evaluation metrics

---

### Models Benchmarked

* Logistic Regression
* Random Forest
* Extra Trees
* HistGradientBoosting
* LightGBM
* CatBoost
* Explainable Boosting Machine (EBM)

---

### Model Evaluation

The project evaluates models using both traditional classification metrics and credit risk metrics.

#### Classification Metrics

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1 Score
* ROC AUC
* PR AUC
* Log Loss
* Brier Score
* Matthews Correlation Coefficient (MCC)
* Cohen's Kappa

#### Credit Risk Metrics

* Gini Coefficient
* Kolmogorov-Smirnov (KS) Statistic

---

### Responsible AI

FairLoans includes a dedicated Responsible AI pipeline built using Fairlearn.

#### Fairness Metrics

* Demographic Parity Difference
* Demographic Parity Ratio
* Equal Opportunity Difference
* Equalized Odds Difference
* Selection Rate
* False Positive Rate
* False Negative Rate
* True Positive Rate
* Accuracy by Protected Group

#### Bias Mitigation

Post-processing mitigation is implemented using Fairlearn's Threshold Optimizer to compare baseline and fairness-aware decision policies.

---

### Governance

The project automatically generates governance artifacts including:

* Baseline fairness report
* Mitigated fairness report
* Fairness comparison
* Executive governance summary
* Model metrics
* Training summary

Governance decisions are based on configurable thresholds that balance predictive performance against fairness improvements.

---

## Project Structure

```text
FairLoans/

├── configs/
│   └── config.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── fairloans_pipeline.joblib
│   └── studies/
│
├── reports/
│
├── artifacts/
│   ├── metrics/
│   ├── fairness/
│   │   ├── baseline/
│   │   ├── mitigated/
│   │   ├── comparison/
│   │   └── plots/
│
├── logs/
│
├── tests/
│
└── src/
    ├── data/
    ├── evaluation/
    ├── fairness/
    ├── features/
    ├── models/
    └── utils/
```

---

## Training Workflow

```text
Load Data
    ↓
Validate Dataset
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
Preprocessing Pipeline
    ↓
Hyperparameter Optimization
    ↓
Model Training
    ↓
Model Evaluation
    ↓
Model Selection
    ↓
Probability Calibration
    ↓
Responsible AI Pipeline
    ↓
Bias Mitigation
    ↓
Governance Reporting
    ↓
Save Artifacts
```

---

## Generated Artifacts

### Models

* Trained pipeline
* Best hyperparameters
* Optuna studies

### Metrics

* Model benchmark results
* Training summary
* Feature names
* Threshold configuration

### Fairness

* Baseline fairness metrics
* Mitigated fairness metrics
* Fairness comparison
* Governance report
* Executive summary

### Logs

* Training logs
* Prediction logs

---

## Technologies

### Machine Learning

* Scikit-learn
* LightGBM
* CatBoost
* InterpretML (EBM)
* Optuna

### Responsible AI

* Fairlearn

### Data Processing

* Pandas
* NumPy

### Visualization

* Matplotlib

### Model Persistence

* Joblib
* JSON
* Pickle

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python -m src.models.train
```

The training pipeline will:

1. Load and validate the dataset.
2. Train and tune all benchmark models.
3. Select the best-performing model.
4. Calibrate predicted probabilities.
5. Evaluate fairness.
6. Apply bias mitigation.
7. Generate governance reports.
8. Save all trained models and artifacts.

---

## Current Status

### Completed

* End-to-end training pipeline
* Hyperparameter optimization
* Model calibration
* Threshold optimization
* Comprehensive evaluation metrics
* Fairness evaluation
* Bias mitigation
* Governance reporting
* Artifact persistence
* Logging
* Unit testing framework

### In Progress

* SHAP explainability
* Streamlit dashboard
* REST API deployment
* Docker support
* CI/CD pipeline

---

## Future Improvements

* SHAP global and local explanations
* Feature interaction analysis
* Streamlit monitoring dashboard
* REST API with FastAPI
* Docker containerization
* Continuous integration and testing
* Automated model monitoring
* Population Stability Index (PSI)
* Drift detection
* Explainability dashboards

---

## License

This project is intended for educational and research. Please verify all fairness, governance, and regulatory requirements before using it in a production credit decisioning environment.
