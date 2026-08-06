from __future__ import annotations

import streamlit as st

from src.configs.config import APP_NAME

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏦",
    layout="wide",
)

#
# -------------------------------------------------------
# Header
# -------------------------------------------------------
#

st.title("🏦 FairLoans")

st.caption("A Responsible AI platform for credit risk prediction, fairness assessment, and model explainability.")

st.divider()

#
# -------------------------------------------------------
# Platform Overview
# -------------------------------------------------------
#

st.subheader("Platform Overview")

st.write("""
FairLoans combines machine learning with Responsible AI practices to
predict credit risk, evaluate fairness, explain model decisions, and
support transparent deployment.
""")

st.divider()

#
# -------------------------------------------------------
# Explore the Platform
# -------------------------------------------------------
#

st.subheader("Explore the Platform")

col1, col2 = st.columns(2)

with col1:

    st.info("""
### 🏠 Prediction

Predict the credit risk of an applicant and
view the factors influencing the decision.
""")

    st.info("""
### 📊 Dataset Explorer

Explore the German Credit dataset through
interactive summaries and visualisations.
""")

with col2:

    st.info("""
### ⚖️ Fairness Dashboard

Assess fairness metrics, compare bias
mitigation results, and review governance
recommendations.
""")

    st.info("""
### 🔍 Explainability & 📈 Performance

Understand model behaviour using SHAP
explanations and review evaluation metrics,
diagnostic plots, and deployment readiness.
""")

st.divider()

st.success("""
**Use the navigation menu on the left to explore each dashboard.**
""")
