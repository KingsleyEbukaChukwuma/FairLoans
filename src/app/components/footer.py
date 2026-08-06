from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st

from src.configs.config import (
    APP_NAME,
    VERSION,
)


class Footer:
    """
    Display application and model metadata.
    """

    @staticmethod
    def show(
        dashboard: dict,
    ) -> None:

        summary = dashboard["summary"]

        best_model = dashboard["best_model"]

        st.divider()

        left, right = st.columns(2)

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
        # Model
        #

        with right:

            st.caption(f"""
**Production Model:** {best_model["Model"]}

**Training Samples:** {summary["dataset_size"]:,}

**Generated:** {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}
""")

        st.caption("""
This dashboard visualizes precomputed evaluation,
fairness, and explainability artifacts. No model
training or SHAP computations are performed within
the Streamlit application.
""")


def footer(
    dashboard: dict,
) -> None:

    Footer.show(
        dashboard,
    )
