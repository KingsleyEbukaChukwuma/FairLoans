from __future__ import annotations

import streamlit as st


class PerformanceBestModel:
    """
    Display the selected production model
    and its deployment information.
    """

    @staticmethod
    def show(
        dashboard: dict,
    ) -> None:

        best = dashboard["best_model"]

        summary = dashboard["summary"]

        metrics = dashboard["metrics"]

        threshold = dashboard["threshold"]

        st.subheader("🏆 Selected Model")

        st.success(f"""
The **{best["Model"]}** model achieved the
highest overall score and was selected
for deployment.
""")

        #
        # -------------------------------------------------------
        # Model Information
        # -------------------------------------------------------
        #

        st.markdown("### Model Information")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Model",
                best["Model"],
            )

        with c2:

            st.metric(
                "Primary Metric",
                summary["optimization_metric"].upper(),
            )

        with c3:

            st.metric(
                "Decision Threshold",
                f"{threshold:.3f}",
            )

        st.divider()

        #
        # -------------------------------------------------------
        # Performance
        # -------------------------------------------------------
        #

        st.markdown("### Performance")

        p1, p2, p3 = st.columns(3)

        with p1:

            st.metric(
                "ROC AUC",
                f'{metrics["ROC AUC"]:.3f}',
            )

        with p2:

            st.metric(
                "F1 Score",
                f'{metrics["F1"]:.3f}',
            )

        with p3:

            st.metric(
                "Brier Score",
                f'{metrics["Brier Score"]:.3f}',
            )

        st.divider()

        #
        # -------------------------------------------------------
        # Training Configuration
        # -------------------------------------------------------
        #

        st.markdown("### Training Configuration")

        t1, t2, t3 = st.columns(3)

        with t1:

            st.metric(
                "Cross Validation",
                summary["cross_validation_folds"],
            )

        with t2:

            st.metric(
                "Optuna Trials",
                summary["optuna_trials"],
            )

        with t3:

            st.metric(
                "Training Samples",
                summary["dataset_size"],
            )

        st.divider()

        #
        # -------------------------------------------------------
        # Calibration
        # -------------------------------------------------------
        #

        st.markdown("### Calibration")

        if summary["calibrated"]:

            st.success("✔ Model probabilities are calibrated " "using **Sigmoid Calibration**.")

        else:

            st.warning("No probability calibration applied.")

        st.divider()

        #
        # -------------------------------------------------------
        # Deployment Status
        # -------------------------------------------------------
        #

        st.markdown("### Deployment Status")

        st.info(f"""
**Production Model:** {best["Model"]}

**Optimization Metric:** {summary["optimization_metric"].upper()}

**Calibration:** {"Sigmoid" if summary["calibrated"] else "None"}

**Decision Threshold:** {threshold:.3f}

This model is the recommended production
candidate based on the completed
training and evaluation pipeline.
""")


def performance_best_model(
    dashboard: dict,
) -> None:

    PerformanceBestModel.show(
        dashboard,
    )
