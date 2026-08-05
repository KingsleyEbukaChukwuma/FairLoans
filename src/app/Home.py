from __future__ import annotations

import streamlit as st

from src.configs.config import APP_NAME

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏦",
    layout="wide",
)

# ==========================================================
# Header
# ==========================================================

st.title("🏦 FairLoans")

st.subheader("Responsible AI Credit Risk Assessment Platform")

st.markdown("""
FairLoans is an end-to-end machine learning application for
**credit risk prediction** with an emphasis on **Responsible AI**.

The platform combines predictive modelling, fairness evaluation,
bias mitigation and explainability into a single interactive
dashboard.
""")

st.divider()

# ==========================================================
# Platform Overview
# ==========================================================

st.header("Platform Overview")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
### Core Capabilities

- Credit Risk Prediction
- Probability Estimation
- Threshold Optimization
- Probability Calibration
- Model Selection
- Hyperparameter Optimization
""")

with col2:

    st.markdown("""
### Responsible AI

- Fairness Evaluation
- Bias Mitigation
- SHAP Explainability
- Model Governance
- Model Diagnostics
- Deployment Reporting
""")

st.divider()

# ==========================================================
# Navigation
# ==========================================================

st.header("Application Pages")

pages = [
    (
        "🏠 Prediction",
        "Predict credit risk for an applicant and view the model explanation.",
    ),
    (
        "📊 Dataset Explorer",
        "Explore the German Credit dataset using interactive filters and summary statistics.",
    ),
    (
        "⚖️ Fairness Dashboard",
        "Compare fairness metrics before and after bias mitigation and review governance recommendations.",
    ),
    (
        "🔍 Explainability",
        "Inspect SHAP-based explanations including global importance, beeswarm, waterfall and interaction plots.",
    ),
    (
        "📈 Model Performance",
        "Review evaluation metrics, calibration, confusion matrix, ROC curve and model training details.",
    ),
]

for page, description in pages:

    st.markdown(f"""
### {page}

{description}
""")

st.divider()

# ==========================================================
# Machine Learning Pipeline
# ==========================================================

st.header("Machine Learning Pipeline")

st.code(
    """
Raw Data
      │
      ▼
Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Hyperparameter Optimization
      │
      ▼
Model Selection
      │
      ▼
Probability Calibration
      │
      ▼
Threshold Optimization
      │
      ▼
Fairness Evaluation
      │
      ▼
Bias Mitigation
      │
      ▼
SHAP Explainability
      │
      ▼
Deployment Dashboard
""",
    language="text",
)
