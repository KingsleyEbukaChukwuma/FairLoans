from __future__ import annotations

from datetime import datetime

import streamlit as st

from configs.config import (
    APP_NAME,
    VERSION,
)


class Footer:
    """
    Display application metadata and model
    provenance information.
    """

    @staticmethod
    def show(
        dashboard: dict,
    ) -> None:

        summary = dashboard["summary"]

        best_model = dashboard["best_model"]

        st.caption("---")

        left, right = st.columns([3, 2])

        #
        # Application
        #

        with left:

            st.caption(f"""
**{APP_NAME}** • Version {VERSION}

Responsible AI Credit Risk Assessment Platform

Generated from saved model artifacts.
""")

        #
        # Model metadata
        #

        with right:

            st.caption(f"**Model:** {best_model['Model']}")

            st.caption(f"**Training Samples:** {summary['dataset_size']}")

            st.caption(
                f"**Cross Validation:** {summary['cross_validation_folds']}-Fold"
            )

        st.caption(f"""
Generated on **{datetime.now().strftime("%Y-%m-%d %H:%M")}**

This dashboard displays precomputed evaluation,
fairness and explainability artifacts. No model
training or SHAP computation occurs inside the
Streamlit application.
""")


def footer(
    dashboard: dict,
) -> None:

    Footer.show(
        dashboard,
    )
