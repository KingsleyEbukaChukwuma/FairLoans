from __future__ import annotations

import pandas as pd
import streamlit as st


class GovernanceDashboard:
    """
    Display Responsible AI governance assessment.
    """

    @staticmethod
    def checklist(
        report: pd.DataFrame,
    ):

        fairness = report[report["Category"] == "Fairness"]

        performance = report[report["Category"] == "Performance"]

        fairness_pass = (fairness["Outcome"] == "Improved").sum() >= (fairness["Outcome"] == "Worse").sum()

        performance_pass = (performance["Outcome"] == "Worse").sum() <= (performance["Outcome"] == "Improved").sum()

        st.subheader("Deployment Readiness")

        checks = [
            (
                "Fairness objectives achieved",
                fairness_pass,
            ),
            (
                "Performance degradation acceptable",
                performance_pass,
            ),
            (
                "Governance assessment completed",
                True,
            ),
        ]

        for label, passed in checks:

            if passed:

                st.success(f"✔ {label}")

            else:

                st.error(f"✖ {label}")

        return fairness_pass and performance_pass

    @staticmethod
    def overview(
        report: pd.DataFrame,
    ):

        st.subheader("Governance Assessment")

        numeric = report.select_dtypes(
            include="number",
        ).columns

        st.dataframe(
            report.style.format({column: "{:.3f}" for column in numeric}),
            use_container_width=True,
            hide_index=True,
        )

    @staticmethod
    def decision(
        deploy: bool,
    ):

        st.subheader("Governance Decision")

        if deploy:

            st.success("""
### Approved for Deployment

The mitigated model satisfies the governance
criteria for fairness and predictive performance.

The assessment indicates that the fairness gains
justify the observed performance trade-offs.
""")

        else:

            st.warning("""
### Further Review Recommended

The current mitigation does not yet satisfy the
governance criteria for deployment.

Additional model development or bias mitigation
is recommended before production deployment.
""")

    @classmethod
    def show(
        cls,
        report: pd.DataFrame,
    ):

        st.caption("Review governance compliance and deployment readiness.")

        deploy = cls.checklist(
            report,
        )

        st.divider()

        cls.overview(
            report,
        )

        st.divider()

        cls.decision(
            deploy,
        )


def governance(
    report: pd.DataFrame,
):

    GovernanceDashboard.show(
        report,
    )
