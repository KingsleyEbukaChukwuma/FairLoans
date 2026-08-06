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

        st.subheader("🏆 Production Model")

        st.success(f"""
The **{best["Model"]}** model achieved the highest
**{summary["optimization_metric"].upper()}** during model
selection and is recommended for deployment.
""")

        st.divider()

        #
        # Model Card
        #

        st.subheader("Model Card")

        left, right = st.columns(2)

        with left:

            st.write(f"**Model:** {best['Model']}")

            st.write(f"**Optimization Metric:** {summary['optimization_metric'].upper()}")

            st.write(f"**Decision Threshold:** {threshold:.3f}")

            st.write(f"**Calibration:** {'Sigmoid' if summary['calibrated'] else 'None'}")

        with right:

            st.write(f"**Training Samples:** {summary['dataset_size']:,}")

            st.write(f"**Cross-Validation:** {summary['cross_validation_folds']}-Fold")

            st.write(f"**Optuna Trials:** {summary['optuna_trials']}")

        st.divider()

        #
        # Key Performance
        #

        st.subheader("Key Performance Metrics")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "ROC AUC",
                f'{metrics["ROC AUC"]:.3f}',
            )

        with c2:

            st.metric(
                "F1 Score",
                f'{metrics["F1"]:.3f}',
            )

        with c3:

            st.metric(
                "Brier Score",
                f'{metrics["Brier Score"]:.3f}',
            )

        st.divider()

        #
        # Deployment Notes
        #

        st.subheader("Deployment Notes")

        notes = []

        if summary["calibrated"]:

            notes.append("✔ Probability estimates are calibrated using Sigmoid Calibration.")

        else:

            notes.append("• Probability calibration was not applied.")

        notes.append(f"✔ Decision threshold set to **{threshold:.3f}** for production inference.")

        notes.append(f"✔ Model selected using **{summary['optimization_metric'].upper()}**.")

        for note in notes:

            st.write(note)


def performance_best_model(
    dashboard: dict,
) -> None:

    PerformanceBestModel.show(
        dashboard,
    )
