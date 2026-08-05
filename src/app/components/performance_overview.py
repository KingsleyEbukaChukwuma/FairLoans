from __future__ import annotations

import streamlit as st


class PerformanceOverview:
    """
    Display key performance indicators for
    the trained model.
    """

    @staticmethod
    def show(
        dashboard: dict,
    ):

        summary = dashboard["summary"]

        metrics = dashboard["metrics"]

        best_model = dashboard["best_model"]

        st.subheader("Model Overview")

        c1, c2, c3 = st.columns(3)

        #
        # Row 1
        #

        with c1:

            st.metric(
                label="Best Model",
                value=best_model["Model"],
                help="Selected after model comparison.",
            )

        with c2:

            st.metric(
                label="Primary Metric",
                value=summary["optimization_metric"].upper(),
            )

        with c3:

            st.metric(
                label="Calibration",
                value="Sigmoid" if summary["calibrated"] else "None",
            )

        st.divider()

        #
        # Row 2
        #

        c4, c5, c6 = st.columns(3)

        with c4:

            st.metric(
                label="ROC AUC",
                value=f"{metrics['ROC AUC']:.3f}",
            )

        with c5:

            st.metric(
                label="Brier Score",
                value=f"{metrics['Brier Score']:.3f}",
            )

        with c6:

            st.metric(
                label="Decision Threshold",
                value=f"{dashboard['threshold']:.3f}",
            )

        st.divider()

        #
        # Training Information
        #

        c7, c8, c9 = st.columns(3)

        with c7:

            st.metric(
                label="Cross Validation",
                value=summary["cross_validation_folds"],
            )

        with c8:

            st.metric(
                label="Optuna Trials",
                value=summary["optuna_trials"],
            )

        with c9:

            st.metric(
                label="Training Samples",
                value=summary["dataset_size"],
            )

        st.info("""
The selected model achieved the best balance
between predictive performance and calibration.
All evaluation metrics shown below are loaded
from saved training artifacts.
""")


def performance_overview(
    dashboard: dict,
):

    PerformanceOverview.show(
        dashboard,
    )
