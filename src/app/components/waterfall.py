from __future__ import annotations

import matplotlib.pyplot as plt
import shap
import streamlit as st

from src.app.helpers.feature_names import (
    clean_feature_name,
)


def waterfall(
    shap_values,
    index: int = 0,
):
    """
    Display a SHAP waterfall plot for an individual prediction.
    """

    st.subheader("Individual Prediction Breakdown")

    st.caption("Shows how each feature contributes to the prediction for a single applicant.")

    #
    # Use business-friendly feature names
    #

    display_values = shap.Explanation(
        values=shap_values.values,
        base_values=shap_values.base_values,
        data=shap_values.data,
        feature_names=[
            clean_feature_name(
                name,
                short=True,
            )
            for name in shap_values.feature_names
        ],
    )

    #
    # Build the SHAP figure
    #

    plt.figure(
        figsize=(10, 6),
    )

    shap.plots.waterfall(
        display_values[index],
        show=False,
    )

    st.pyplot(
        plt.gcf(),
        clear_figure=True,
    )

    plt.close()
