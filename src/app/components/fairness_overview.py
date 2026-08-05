from __future__ import annotations

import streamlit as st


def fairness_overview(
    summary: dict,
):
    """
    Display the Responsible AI dashboard overview.
    """

    st.subheader("Overview")

    col1, col2, col3 = st.columns(3)

    #
    # Fairness
    #

    with col1:

        st.metric(
            label="Fairness Metrics Improved",
            value=summary["Fairness Metrics Improved"],
            help=("Number of fairness metrics that " "improved after mitigation."),
        )

    #
    # Performance
    #

    with col2:

        st.metric(
            label="Performance Metrics Improved",
            value=summary["Performance Metrics Improved"],
            help=("Number of predictive performance " "metrics that improved."),
        )

    #
    # Recommendation
    #

    with col3:

        recommendation = summary["Recommendation"]

        icon = "✅" if recommendation == "Mitigated Model" else "⚠️"

        st.metric(
            label="Recommended Model",
            value=f"{icon} {recommendation}",
        )

    st.divider()

    #
    # Summary Cards
    #

    left, right = st.columns(2)

    with left:

        st.success(f"""
### Fairness Summary

- Improved Metrics:
  **{summary['Fairness Metrics Improved']}**

- Worse Metrics:
  **{summary['Fairness Metrics Worse']}**
""")

    with right:

        st.info(f"""
### Performance Summary

- Improved Metrics:
  **{summary['Performance Metrics Improved']}**

- Worse Metrics:
  **{summary['Performance Metrics Worse']}**
""")

    st.divider()

    #
    # Recommendation Banner
    #

    recommendation = summary["Recommendation"]

    if recommendation == "Mitigated Model":

        st.success("""
### Recommendation

Deploy the **Mitigated Model**.

The bias mitigation process substantially
improved fairness while maintaining
acceptable predictive performance.
""")

    else:

        st.warning("""
### Recommendation

Retain the **Baseline Model**.

The mitigation process introduced a
performance degradation that outweighs
the fairness improvements.
""")
