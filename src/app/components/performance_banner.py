from __future__ import annotations

import streamlit as st


class PerformanceBanner:
    """
    Display the deployment recommendation.
    """

    @staticmethod
    def show(
        dashboard: dict,
    ) -> None:

        summary = dashboard["summary"]

        best_model = dashboard["best_model"]

        st.success(f"""
### 🏆 Deployment Recommendation

Deploy **{best_model["Model"]}** as the production credit risk model.

The model achieved the highest **{summary["optimization_metric"].upper()}**
during hyperparameter optimisation while maintaining strong predictive
performance and calibration.
""")


def performance_banner(
    dashboard: dict,
) -> None:

    PerformanceBanner.show(
        dashboard,
    )
