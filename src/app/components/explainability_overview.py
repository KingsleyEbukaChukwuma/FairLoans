import streamlit as st


def explainability_overview():

    c1, c2, c3 = st.columns(3)

    c1.metric("Global Explanations", "✔")

    c2.metric("Local Explanations", "✔")

    c3.metric("SHAP", "Enabled")

    st.info("""
This dashboard displays precomputed SHAP artifacts generated
during model training.

No SHAP computation occurs within the Streamlit application,
ensuring fast loading and consistent explanations.
""")
