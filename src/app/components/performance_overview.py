from __future__ import annotations

import streamlit as st


class PerformanceOverview:
    """
    Display the key performance indicators
    for the selected model.
    """

    @staticmethod
    def show(
        dashboard: dict,
    ):

        metrics = dashboard["metrics"]

        best_model = dashboard["best_model"]

        summary = dashboard["summary"]

        st.subheader("Performance Overview")

        #
        # Primary KPIs
        #

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Selected Model",
                best_model["Model"],
            )

        with col2:

            st.metric(
                "ROC AUC",
                f"{metrics['ROC AUC']:.3f}",
            )

        with col3:

            st.metric(
                "F1 Score",
                f"{metrics['F1']:.3f}",
            )

        with col4:

            st.metric(
                "Brier Score",
                f"{metrics['Brier Score']:.3f}",
            )

        st.divider()

        #
        # Deployment KPIs
        #

        col5, col6, col7 = st.columns(3)

        with col5:

            st.metric(
                "Decision Threshold",
                f"{dashboard['threshold']:.3f}",
            )

        with col6:

            st.metric(
                "Primary Metric",
                summary["optimization_metric"].upper(),
            )

        with col7:

            st.metric(
                "Calibration",
                "Enabled" if summary["calibrated"] else "Disabled",
            )


def performance_overview(
    dashboard: dict,
):

    PerformanceOverview.show(
        dashboard,
    )
