# 🏦 FairLoans

**Responsible AI Credit Risk Assessment Platform**

FairLoans is an end-to-end machine learning application for credit
risk prediction that combines predictive modelling with Responsible
AI practices. The platform integrates model training, fairness
evaluation, explainability, governance reporting, and interactive
deployment into a single Streamlit application.

> **Production Model:** CatBoost\
> **Primary Optimisation Metric:** ROC AUC\
> **Deployment:** Artifact-driven Streamlit dashboard

------------------------------------------------------------------------

# Features

## 🤖 Machine Learning

-   End-to-end credit risk prediction
-   Benchmarking of seven classification models
-   Hyperparameter optimisation with Optuna
-   Probability calibration
-   Decision threshold optimisation
-   Automated model selection

## ⚖️ Responsible AI

-   Fairlearn fairness evaluation
-   Bias mitigation
-   Governance reporting
-   Executive deployment recommendations

## 🔍 Explainability

-   Global feature importance
-   SHAP Beeswarm
-   SHAP Waterfall
-   Local prediction explanations

## 📈 Model Performance

-   Classification metrics
-   ROC Curve
-   Confusion Matrix
-   Calibration Curve
-   Production model summary

------------------------------------------------------------------------

# Architecture

``` text
German Credit Dataset
        │
        ▼
 Data Validation
        │
        ▼
 Feature Engineering
        │
        ▼
 Model Benchmarking
        │
        ▼
 Hyperparameter Optimisation
        │
        ▼
 Best Model Selection
        │
        ▼
 Calibration
        │
        ▼
 Threshold Optimisation
        │
        ▼
 Fairness Assessment
        │
        ▼
 SHAP Explainability
        │
        ▼
 Persist Artifacts
        │
        ▼
 Streamlit Dashboard
```

The application follows a train once, deploy many architecture, all
computationally intensive tasks are performed during training and
persisted as artifacts. The Streamlit application loads these artifacts
at runtime, providing fast, reproducible, and deterministic dashboards.

------------------------------------------------------------------------

# Streamlit Application

  Page                    Purpose
  ----------------------- ----------------------------------------------
  🏠 Home                 Project overview
  💳 Prediction           Credit risk prediction and local explanation
  📊 Dataset Explorer     Explore the German Credit dataset
  ⚖️ Fairness Dashboard   Compare baseline and mitigated fairness
  🔍 Explainability       Global and local SHAP explanations
  📈 Model Performance    Evaluation metrics and diagnostics

------------------------------------------------------------------------

# Results

## Production Model

  Metric                       Value
  ------------------- --------------
  Selected Model        **CatBoost**
  ROC AUC                  **0.802**
  Accuracy                     0.763
  Balanced Accuracy            0.682
  F1 Score                     0.548
  PR AUC                       0.664
  Brier Score                  0.159
  Gini                         0.604
  KS Statistic                 0.463

------------------------------------------------------------------------

# Installation

``` bash
git clone https://github.com/KingsleyEbukaChukwuma/FairLoans.git
cd FairLoans
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run Home.py
```
or install FairLoans in editable mode:

```bash
pip install -e .
```

------------------------------------------------------------------------

# Reproducing the Training Pipeline

``` bash
python -m src.models.train
```

This regenerates the trained models, evaluation metrics, fairness
reports, explainability artifacts, and governance summaries consumed by
the Streamlit application.

------------------------------------------------------------------------

# License

MIT License.

NB: This project is intended for educational and research purposes, Verify
fairness, governance, and regulatory requirements before using the
workflow in production.
