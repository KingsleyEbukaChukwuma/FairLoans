from __future__ import annotations

import pandas as pd
import streamlit as st


class FairnessMetricFrame:
    """
    Display MetricFrame results for the
    baseline and mitigated models.
    """

    @staticmethod
    def _style(
        df: pd.DataFrame,
    ):

        numeric = df.select_dtypes(
            include="number",
        ).columns

        return df.style.format({column: "{:.3f}" for column in numeric}).background_gradient(
            cmap="Blues",
            subset=numeric,
        )

    @classmethod
    def show(
        cls,
        baseline: pd.DataFrame,
        mitigated: pd.DataFrame,
    ):

        st.subheader("Fairness by Protected Group")

        st.caption("Compare fairness metrics before and after " "bias mitigation.")

        tab1, tab2 = st.tabs(
            [
                "Baseline",
                "Mitigated",
            ]
        )

        #
        # Baseline
        #

        with tab1:

            st.markdown("### Baseline Model")

            st.dataframe(
                cls._style(
                    baseline,
                ),
                use_container_width=True,
                hide_index=True,
            )

        #
        # Mitigated
        #

        with tab2:

            st.markdown("### Mitigated Model")

            st.dataframe(
                cls._style(
                    mitigated,
                ),
                use_container_width=True,
                hide_index=True,
            )

        #
        # Side-by-side comparison
        #

        st.divider()

        st.subheader("Side-by-Side Comparison")

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### Baseline")

            st.dataframe(
                cls._style(
                    baseline,
                ),
                use_container_width=True,
                hide_index=True,
            )

        with col2:

            st.markdown("#### Mitigated")

            st.dataframe(
                cls._style(
                    mitigated,
                ),
                use_container_width=True,
                hide_index=True,
            )


def metricframe(
    baseline: pd.DataFrame,
    mitigated: pd.DataFrame,
):

    FairnessMetricFrame.show(
        baseline,
        mitigated,
    )
