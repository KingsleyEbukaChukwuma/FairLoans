from __future__ import annotations

import streamlit as st


class ExecutiveSummary:
    """
    Display a concise executive summary of the
    Responsible AI assessment.
    """

    @staticmethod
    def summary(summary):

        recommendation = summary["Recommendation"]

        fairness_improved = summary["Fairness Metrics Improved"]

        performance_worse = summary["Performance Metrics Worse"]

        if recommendation == "Mitigated Model":

            st.success(f"""
### Executive Summary

The **mitigated model** is recommended for deployment.

The bias mitigation strategy improved **{fairness_improved}** fairness metric(s) while maintaining acceptable predictive performance. Overall, the fairness gains outweigh the observed performance trade-offs, making the mitigated model suitable for deployment.
""")

        else:

            st.warning(f"""
### Executive Summary

The **baseline model** is recommended for deployment.

Although the mitigation improved **{fairness_improved}** fairness metric(s), the resulting degradation in **{performance_worse}** performance metric(s) outweighs the fairness gains. Further model refinement or bias mitigation is recommended before deployment.
""")

    @classmethod
    def show(
        cls,
        summary,
    ):

        cls.summary(
            summary,
        )


def executive_summary(summary):

    ExecutiveSummary.show(
        summary,
    )
