from __future__ import annotations

import pandas as pd
import streamlit as st


class GovernanceDashboard:
    """
    Display the Responsible AI governance report.
    """

    @staticmethod
    def style(
        report: pd.DataFrame,
    ):

        def colour(value):

            if value == "Improved":

                return "background-color:#d4edda;" "color:#155724;" "font-weight:bold;"

            if value == "Worse":

                return "background-color:#f8d7da;" "color:#721c24;" "font-weight:bold;"

            return "background-color:#fff3cd;" "color:#856404;"

        numeric = report.select_dtypes(
            include="number",
        ).columns

        return (
            report.style.format({column: "{:.3f}" for column in numeric})
            .background_gradient(
                cmap="Blues",
                subset=["Before", "After"],
            )
            .background_gradient(
                cmap="RdYlGn",
                subset=["Change"],
            )
            .map(
                colour,
                subset=["Outcome"],
            )
        )

    @staticmethod
    def deployment_decision(
        report: pd.DataFrame,
    ):

        improved = (report["Outcome"] == "Improved").sum()

        worse = (report["Outcome"] == "Worse").sum()

        if improved >= worse:

            st.success("""
### ✅ Deployment Recommendation

Deploy the **Mitigated Model**.

The governance assessment indicates that
fairness improvements outweigh the
performance degradation.
""")

        else:

            st.warning("""
### ⚠ Deployment Recommendation

Retain the **Baseline Model**.

The fairness improvements are insufficient
to justify the observed performance loss.
""")

    @staticmethod
    def trade_offs(
        report: pd.DataFrame,
    ):

        st.subheader("Trade-off Summary")

        fairness = report[report["Category"] == "Fairness"]

        performance = report[report["Category"] == "Performance"]

        col1, col2 = st.columns(2)

        with col1:

            st.info(f"""
### Fairness

Improved Metrics

**{(fairness["Outcome"]=="Improved").sum()}**

Worse Metrics

**{(fairness["Outcome"]=="Worse").sum()}**
""")

        with col2:

            st.info(f"""
### Performance

Improved Metrics

**{(performance["Outcome"]=="Improved").sum()}**

Worse Metrics

**{(performance["Outcome"]=="Worse").sum()}**
""")

    @classmethod
    def show(
        cls,
        report: pd.DataFrame,
    ):

        st.subheader("Governance Report")

        st.caption("Assessment of fairness and " "performance trade-offs.")

        st.dataframe(
            cls.style(
                report,
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        cls.trade_offs(
            report,
        )

        st.divider()

        cls.deployment_decision(
            report,
        )


def governance(
    report: pd.DataFrame,
):

    GovernanceDashboard.show(
        report,
    )
