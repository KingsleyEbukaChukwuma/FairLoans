# 🏦 FairLoans: Responsible AI Credit Risk Assessment Platform

[![FairLoans CI](https://github.com/KingsleyEbukaChukwuma/FairLoans/actions/workflows/ci.yml/badge.svg)](https://github.com/KingsleyEbukaChukwuma/FairLoans/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

> **An end-to-end Responsible AI platform for credit risk assessment that combines predictive modelling, fairness evaluation, explainability, governance, and interactive deployment.**

---

# Overview

FairLoans is an end-to-end machine learning platform for credit risk assessment built to demonstrate how high-performing predictive models can be combined with Responsible AI principles in a production-oriented workflow.

The project implements the complete machine learning lifecycle, from data ingestion and preprocessing through feature engineering, hyperparameter optimization, probability calibration, threshold optimization, fairness evaluation, bias mitigation, explainability, governance reporting, and deployment as an interactive Streamlit application.

Rather than focusing solely on predictive performance, FairLoans emphasizes the broader requirements of trustworthy AI by integrating fairness analysis, model transparency, explainability, reproducibility, and governance into a unified system. All dashboards are powered by precomputed artifacts generated during training, ensuring fast, reproducible, and consistent reporting without recomputing models during deployment.

The current production model is **CatBoost**, selected after benchmarking seven machine learning algorithms using **ROC AUC** as the primary optimization metric. The final model uses calibrated probabilities, 5-fold cross-validation, and Optuna-based hyperparameter optimization to support reliable credit risk predictions.

---

# Key Features

### 🤖 Machine Learning

* End-to-end credit risk prediction pipeline
* Automated data validation and preprocessing
* Feature engineering using Scikit-learn pipelines
* Benchmarking of seven classification algorithms
* Hyperparameter optimization with Optuna
* Probability calibration using sigmoid calibration
* Decision threshold optimization
* Automated model selection based on multiple evaluation metrics

### ⚖️ Responsible AI

* Fairness assessment using Fairlearn
* Baseline and post-mitigation fairness evaluation
* Bias mitigation with Threshold Optimizer
* Fairness comparison across protected groups
* Governance reporting and executive summaries

### 🔍 Explainability

* Global feature importance
* SHAP beeswarm visualization
* Local prediction explanations
* Feature interaction heatmaps
* Explainability dashboard powered by saved artifacts

### 📊 Model Evaluation

* Comprehensive classification metrics
* Credit risk metrics including Gini and KS Statistic
* ROC Curve
* Confusion Matrix
* Calibration Curve
* Production model summary and deployment diagnostics

### 🖥️ Interactive Application

* Credit Risk Prediction
* Dataset Explorer
* Fairness Dashboard
* Explainability Dashboard
* Model Performance Dashboard

### 🛠️ Engineering Practices

* Modular Python architecture
* Artifact-driven deployment
* GitHub Actions continuous integration
* Automated testing with Pytest
* Code quality checks using Ruff and Black
* Configuration-driven project structure
* Reproducible machine learning pipeline


# System Architecture

FairLoans follows a modular, artifact-driven architecture that separates model development from application deployment. The machine learning pipeline is executed once during training to produce validated models, evaluation metrics, fairness reports, explainability artifacts, and governance documents. The Streamlit application then loads these saved artifacts to provide fast, reproducible, and interactive reporting without retraining the model.

```text
                              German Credit Dataset
                                       │
                                       ▼
                          Data Loading & Validation
                                       │
                                       ▼
                         Feature Engineering Pipeline
                                       │
                                       ▼
                        Train / Test Data Splitting
                                       │
                                       ▼
                    Scikit-learn Preprocessing Pipeline
                                       │
                                       ▼
                       Benchmark Machine Learning Models
                                       │
                                       ▼
                     Hyperparameter Optimization (Optuna)
                                       │
                                       ▼
                         Best Model Selection (CatBoost)
                                       │
                                       ▼
                  Probability Calibration (Sigmoid Method)
                                       │
                                       ▼
                     Decision Threshold Optimization
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
           Model Performance Evaluation         Responsible AI Pipeline
                     │                                   │
                     ▼                                   ▼
          ROC • Confusion Matrix              Fairness Evaluation
          Calibration • Metrics               Bias Mitigation
                     │                         Governance Reports
                     └───────────────┬────────────────────────────┘
                                     ▼
                           SHAP Explainability
                    Global • Local • Interaction Analysis
                                     │
                                     ▼
                         Persist Models & Artifacts
                                     │
                                     ▼
                    Interactive Streamlit Application
                                     │
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        ▼              ▼              ▼              ▼              ▼
   Prediction     Dataset        Fairness     Explainability   Performance
                   Explorer       Dashboard      Dashboard       Dashboard
```

## Architecture Highlights

The FairLoans architecture is designed around the principle of **train once, deploy many**. All computationally intensive tasks—including model training, hyperparameter optimization, fairness assessment, probability calibration, and SHAP explainability—are executed during the training phase.

The training pipeline produces a collection of persistent artifacts, including trained models, evaluation metrics, fairness reports, governance summaries, explainability outputs, and diagnostic plots. These artifacts are stored and later consumed by the Streamlit application, allowing the dashboard to provide interactive insights without recomputing machine learning results.

This separation between training and deployment provides several advantages:

* Faster application performance through artifact-based loading.
* Reproducible and deterministic dashboards.
* Clear separation of development and inference workflows.
* Simplified deployment and maintenance.
* Improved scalability for future model updates.

# Application Documentation

The FairLoans Streamlit application provides an interactive interface for exploring the complete machine learning workflow. Rather than retraining models, the application loads precomputed artifacts generated during training, ensuring consistent, reproducible, and responsive analysis.

---

# Credit Risk Prediction

The **Prediction** page enables users to evaluate the credit risk of a new loan applicant using the selected production model.

Users enter applicant information through a structured form that mirrors the German Credit dataset features. After submission, the calibrated CatBoost model estimates the probability of default and classifies the applicant according to the optimized decision threshold.

The prediction interface includes:

* Applicant data entry form
* Credit risk classification
* Probability of default estimation
* Decision threshold evaluation
* Prediction confidence
* Local SHAP explanation for individual predictions

This page demonstrates how trained machine learning models can be deployed for interactive decision support while maintaining model transparency.

---

# Fairness Dashboard

The **Fairness Dashboard** evaluates whether model performance differs across protected demographic groups.

FairLoans uses Fairlearn to measure fairness before and after applying bias mitigation using the Threshold Optimizer. The dashboard compares baseline and mitigated results to illustrate the trade-offs between predictive performance and fairness.

The dashboard includes:

* Executive fairness summary
* Fairness metrics by protected group
* Baseline versus mitigated comparison
* Governance recommendations
* Responsible AI reporting
* Interactive fairness visualizations

Key fairness metrics include:

* Demographic Parity
* Equal Opportunity
* Equalized Odds
* Selection Rate
* True Positive Rate
* False Positive Rate
* False Negative Rate
* Accuracy by protected group

This component demonstrates how fairness evaluation can become an integral part of the machine learning lifecycle rather than an afterthought.

---

# Explainability Dashboard

The **Explainability Dashboard** provides model transparency using SHAP (SHapley Additive Explanations).

Instead of generating explanations during application runtime, FairLoans loads precomputed explainability artifacts produced during model training. This significantly improves application responsiveness while ensuring reproducible explanations.

Available visualizations include:

* Global Feature Importance
* SHAP Beeswarm Plot
* Waterfall Plot
* Feature Interaction Heatmap
* Local Prediction Explanation

These visualizations help users understand both the overall behaviour of the production model and the factors influencing individual predictions.

---

# Model Performance Dashboard

The **Model Performance Dashboard** summarizes the predictive quality of the selected production model using saved evaluation artifacts.

The dashboard combines numerical metrics with graphical diagnostics to provide a comprehensive assessment of model performance.

Included evaluations are:

* Production model summary
* ROC Curve
* Confusion Matrix
* Calibration Curve
* Classification metrics
* Credit risk metrics
* Training summary
* Decision threshold
* Deployment recommendation

The dashboard highlights the selected CatBoost model together with its evaluation metrics, calibration status, and optimized decision threshold.

---

# Dataset Explorer

The **Dataset Explorer** allows users to interactively examine the German Credit dataset used throughout the project.

The explorer provides descriptive statistics and filtering capabilities to support exploratory data analysis without requiring programming knowledge.

Features include:

* Interactive dataset preview
* Summary statistics
* Variable distributions
* Feature filtering
* Missing value inspection
* Data exploration tools

This page enables users to better understand the structure and characteristics of the dataset before reviewing model predictions or fairness analyses.

# The Streamlit Features

The FairLoans application is built as an interactive Streamlit dashboard that allows users to explore every stage of the Responsible AI workflow, from credit risk prediction to fairness evaluation and model explainability.

---

## 🏠 Home

The landing page introduces FairLoans, summarizes the project objectives, and provides an overview of the machine learning workflow, Responsible AI capabilities, and available application modules.

---

## 💳 Credit Risk Prediction

The Prediction page enables users to evaluate the credit risk of an individual applicant using the deployed CatBoost model. Predictions include the estimated probability of default, decision threshold evaluation, and a local SHAP explanation to improve transparency.


---

## ⚖️ Fairness Dashboard

The Fairness Dashboard compares model performance before and after bias mitigation using Fairlearn's Threshold Optimizer. It includes fairness metrics, group-level comparisons, governance reports, and executive summaries to support Responsible AI evaluation.

---

## 🔍 Explainability Dashboard

The Explainability Dashboard presents precomputed SHAP artifacts that explain both global model behaviour and individual predictions. Users can explore feature importance, SHAP beeswarm plots, waterfall plots, interaction heatmaps, and local explanations.

---

## 📈 Model Performance Dashboard

The Model Performance Dashboard summarizes the predictive quality of the production model using saved evaluation artifacts. It includes classification metrics, ROC Curve, Confusion Matrix, Calibration Curve, and deployment recommendations.


# Installation

Follow the steps below to set up FairLoans locally.

## 1. Clone the Repository

```bash
git clone https://github.com/KingsleyEbukaChukwuma/FairLoans.git

cd FairLoans
```

---

## 2. Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv .venv

source .venv/bin/activate
```

---

## 3. Install Dependencies

Install all required Python packages.

```bash
pip install --upgrade pip

pip install -r requirements.txt
```

---

## 4. Launch the Application

Start the Streamlit application.

```bash
streamlit run Home.py
```

Once the application starts, Streamlit will display a local URL similar to:

```text
Local URL: http://localhost:8501
```

Open the URL in your web browser to access the FairLoans dashboard.

---

# Usage

The Streamlit application consists of five interactive modules:

| Module                | Description                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| 🏠 Home               | Overview of the platform and Responsible AI workflow                        |
| 💳 Prediction         | Predict credit risk for a new applicant using the production CatBoost model |
| 📊 Dataset Explorer   | Explore the German Credit dataset with interactive summaries                |
| ⚖️ Fairness Dashboard | Compare fairness metrics before and after bias mitigation                   |
| 🔍 Explainability     | Visualize SHAP-based explanations and feature importance                    |
| 📈 Model Performance  | Review model metrics, calibration, diagnostics, and deployment summary      |

---

## Training the Models

To reproduce the complete machine learning pipeline from scratch:

```bash
python -m src.models.train
```

The training pipeline will:

1. Load and validate the German Credit dataset.
2. Perform feature engineering and preprocessing.
3. Train and benchmark seven machine learning models.
4. Optimize hyperparameters using Optuna.
5. Select the best-performing model.
6. Apply probability calibration.
7. Optimize the decision threshold.
8. Evaluate fairness and apply bias mitigation.
9. Generate explainability and governance artifacts.
10. Persist models, metrics, reports, and visualizations for deployment.

> **Note:** The Streamlit application loads the saved artifacts generated during training. It does not retrain models or recompute SHAP explanations at runtime, ensuring fast, reproducible, and consistent results.


# Results

The FairLoans training pipeline benchmarked **seven** machine learning algorithms and selected the production model using **ROC AUC** as the primary optimization metric. The selected model was subsequently calibrated, evaluated for fairness, and integrated into the Streamlit application.

---

# Production Model Summary

| Metric                          |              Value |
| ------------------------------- | -----------------: |
| **Selected Model**              |       **CatBoost** |
| **Models Benchmarked**          |                  7 |
| **Primary Optimization Metric** |            ROC AUC |
| **ROC AUC**                     |          **0.802** |
| **Accuracy**                    |          **0.763** |
| **Balanced Accuracy**           |          **0.682** |
| **Precision**                   |          **0.642** |
| **Recall**                      |          **0.478** |
| **F1 Score**                    |          **0.548** |
| **PR AUC**                      |          **0.664** |
| **Brier Score**                 |          **0.159** |
| **Gini Coefficient**            |          **0.604** |
| **KS Statistic**                |          **0.463** |
| **Cross Validation**            |             5-Fold |
| **Hyperparameter Optimization** | Optuna (50 Trials) |
| **Probability Calibration**     |            Sigmoid |
| **Dataset Size**                |      1,000 Records |

---

# Model Benchmark

The project evaluated seven supervised learning algorithms before selecting the production model.

| Model                        |  ROC AUC  |  F1 Score | Brier Score |
| ---------------------------- | :-------: | :-------: | :---------: |
| **CatBoost** ⭐               | **0.802** |   0.548   |  **0.159**  |
| Logistic Regression          |   0.800   |   0.525   |    0.164    |
| Extra Trees                  |   0.795   |   0.538   |    0.164    |
| LightGBM                     |   0.794   |   0.537   |    0.164    |
| HistGradientBoosting         |   0.787   | **0.596** |    0.206    |
| Random Forest                |   0.785   |   0.353   |    0.169    |
| Explainable Boosting Machine |   0.776   |   0.540   |    0.179    |

---

# Responsible AI Outcomes

FairLoans extends traditional model evaluation by incorporating fairness analysis and bias mitigation into the model development lifecycle.

The Responsible AI pipeline provides:

* Fairness evaluation across protected demographic groups.
* Baseline and post-mitigation comparisons.
* Bias mitigation using Fairlearn's Threshold Optimizer.
* Governance reports summarizing fairness and performance trade-offs.
* Executive summaries to support responsible deployment decisions.

These artifacts enable users to assess both predictive performance and fairness characteristics before deploying the model.

---

# Deployment Readiness

The production model satisfies the project's deployment workflow by combining predictive performance with engineering and Responsible AI practices.

**Deployment highlights:**

* ✅ Production model selected through benchmark comparison.
* ✅ Hyperparameters optimized using Optuna.
* ✅ Probabilities calibrated using sigmoid calibration.
* ✅ Decision threshold optimized.
* ✅ Fairness evaluated and bias mitigation applied.
* ✅ SHAP explainability artifacts generated.
* ✅ Governance reports produced.
* ✅ Interactive Streamlit dashboard implemented.
* ✅ Automated testing and GitHub Actions CI configured.

---

## License

This project is intended for educational and research. Please verify all fairness, governance, and regulatory requirements before using it in a production credit decisioning environment.

