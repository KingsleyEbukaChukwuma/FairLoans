from __future__ import annotations

import streamlit as st


def fairness_overview(
    summary: dict,
):
    """
    Display a high-level overview of the Responsible AI assessment.
    """

    st.subheader("Assessment Overview")

    #
    # KPI Cards
    #

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Fairness Improved",
            summary["Fairness Metrics Improved"],
            help="Number of fairness metrics that improved after mitigation.",
        )

    with col2:

        st.metric(
            "Fairness Worse",
            summary["Fairness Metrics Worse"],
        )

    with col3:

        st.metric(
            "Performance Improved",
            summary["Performance Metrics Improved"],
        )

    with col4:

        st.metric(
            "Performance Worse",
            summary["Performance Metrics Worse"],
        )
