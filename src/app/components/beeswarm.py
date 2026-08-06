from __future__ import annotations

import matplotlib.pyplot as plt
import shap
import streamlit as st

from src.app.helpers.feature_names import clean_feature_name
from src.configs.config import SHAP_MAX_DISPLAY


def beeswarm(
    shap_values,
):
    """
    Display the SHAP beeswarm plot.
    """

    st.subheader("Feature Impact (SHAP Beeswarm)")

    st.caption(
        "This visualization shows how each feature influences model predictions "
        "across all applicants. Features are ordered by overall importance, while "
        "each point represents one applicant. Color indicates whether the feature "
        "value is relatively low or high."
    )

    #
    # Use friendly feature names
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

    plt.figure(
        figsize=(10, 6),
    )

    shap.plots.beeswarm(
        display_values,
        max_display=SHAP_MAX_DISPLAY,
        show=False,
    )

    st.pyplot(
        plt.gcf(),
        clear_figure=True,
    )

    plt.close()
