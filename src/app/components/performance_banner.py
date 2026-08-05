from __future__ import annotations

import streamlit as st


class PerformanceBanner:
    """
    Executive banner for the selected model.
    """

    @staticmethod
    def show(
        dashboard: dict,
    ) -> None:

        summary = dashboard["summary"]

        best_model = dashboard["best_model"]

        metrics = dashboard["metrics"]

        st.success(f"""
### 🏆 Selected Model

**{best_model["Model"]}** was selected as the production model
using **{summary["optimization_metric"].upper()}**
as the primary optimization metric.
""")

        st.subheader("Deployment Summary")

        col1, col2, col3, col4 = st.columns(4)

        #
        # Model
        #

        with col1:

            st.metric(
                "Model",
                best_model["Model"],
            )

        #
        # ROC AUC
        #

        with col2:

            st.metric(
                "ROC AUC",
                f'{metrics["ROC AUC"]:.3f}',
            )

        #
        # Threshold
        #

        with col3:

            st.metric(
                "Decision Threshold",
                f'{dashboard["threshold"]:.3f}',
            )

        #
        # Calibration
        #

        with col4:

            st.metric(
                "Calibration",
                "Sigmoid" if summary["calibrated"] else "None",
            )

        st.info(f"""
**Cross Validation:** {summary["cross_validation_folds"]}-Fold

**Optuna Trials:** {summary["optuna_trials"]}

**Training Samples:** {summary["dataset_size"]}

**Model Selection Metric:** {summary["optimization_metric"].upper()}
""")


def performance_banner(
    dashboard: dict,
) -> None:

    PerformanceBanner.show(
        dashboard,
    )
