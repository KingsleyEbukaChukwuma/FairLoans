from __future__ import annotations

import streamlit as st


def explainability_overview():

    st.subheader("Explainability Overview")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Global Analysis",
            "2",
            help="Feature Importance and SHAP Feature Impact",
        )

    with c2:

        st.metric(
            "Local Analysis",
            "2",
            help="Waterfall Plot and Individual Prediction Summary",
        )

    with c3:

        st.metric(
            "Feature Relationships",
            "1",
            help="SHAP Feature Interaction Analysis",
        )

    st.info("""
This dashboard explains **why** the model makes its predictions.

The visualizations are generated from explainability artifacts created during model training and rendered interactively within the application.
""")
