from __future__ import annotations

import streamlit as st


class ExecutiveSummary:
    """
    Display an executive summary of the
    Responsible AI assessment.
    """

    @staticmethod
    def recommendation(summary):

        recommendation = summary["Recommendation"]

        if recommendation == "Mitigated Model":

            st.success(f"""
## ✅ Recommendation

**Deploy the {recommendation}.**

The fairness improvements substantially
outweigh the observed performance
degradation.
""")

        else:

            st.warning(f"""
## ⚠ Recommendation

**Retain the {recommendation}.**

The reduction in predictive performance
does not justify deployment of the
mitigated model.
""")

    @staticmethod
    def metrics(summary):

        st.subheader("Assessment Summary")

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Fairness Metrics Improved",
                summary["Fairness Metrics Improved"],
            )

            st.metric(
                "Fairness Metrics Worse",
                summary["Fairness Metrics Worse"],
            )

        with c2:

            st.metric(
                "Performance Metrics Improved",
                summary["Performance Metrics Improved"],
            )

            st.metric(
                "Performance Metrics Worse",
                summary["Performance Metrics Worse"],
            )

    @staticmethod
    def deployment(summary):

        recommendation = summary["Recommendation"]

        st.subheader("Deployment Decision")

        if recommendation == "Mitigated Model":

            st.info("""
The mitigated model satisfies the
governance requirements for deployment.

The observed fairness improvements are
considered sufficient while maintaining
acceptable predictive performance.
""")

        else:

            st.info("""
The baseline model should remain in
production.

Additional bias mitigation or model
development is recommended before
deployment.
""")

    @staticmethod
    def key_takeaways(summary):

        st.subheader("Key Takeaways")

        bullets = [
            f"• Fairness metrics improved: {summary['Fairness Metrics Improved']}",
            f"• Fairness metrics worsened: {summary['Fairness Metrics Worse']}",
            f"• Performance metrics improved: {summary['Performance Metrics Improved']}",
            f"• Performance metrics worsened: {summary['Performance Metrics Worse']}",
            f"• Recommended model: {summary['Recommendation']}",
        ]

        for bullet in bullets:

            st.write(bullet)

    @classmethod
    def show(
        cls,
        summary,
    ):

        cls.recommendation(
            summary,
        )

        st.divider()

        cls.metrics(
            summary,
        )

        st.divider()

        cls.deployment(
            summary,
        )

        st.divider()

        cls.key_takeaways(
            summary,
        )


def executive_summary(
    summary,
):

    ExecutiveSummary.show(
        summary,
    )
