from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


class FairnessComparison:
    """
    Display fairness comparison between the
    baseline and mitigated models.
    """

    @staticmethod
    def style(
        comparison: pd.DataFrame,
    ):

        def colour_outcome(value):

            if value == "Improved":
                return "background-color:#d4edda;" "color:#155724;" "font-weight:bold;"

            if value == "Worse":
                return "background-color:#f8d7da;" "color:#721c24;" "font-weight:bold;"

            return "background-color:#fff3cd;" "color:#856404;"

        numeric = comparison.select_dtypes(
            include="number",
        ).columns

        return (
            comparison.style.format({column: "{:.3f}" for column in numeric})
            .background_gradient(
                cmap="Blues",
                subset=["Before", "After"],
            )
            .background_gradient(
                cmap="RdYlGn",
                subset=["Change"],
            )
            .map(
                colour_outcome,
                subset=["Outcome"],
            )
        )

    @staticmethod
    def plot(
        comparison: pd.DataFrame,
    ):

        fig, ax = plt.subplots(
            figsize=(10, 5),
        )

        x = range(len(comparison))

        width = 0.35

        ax.bar(
            [i - width / 2 for i in x],
            comparison["Before"],
            width,
            label="Baseline",
        )

        ax.bar(
            [i + width / 2 for i in x],
            comparison["After"],
            width,
            label="Mitigated",
        )

        ax.set_xticks(
            list(x),
        )

        ax.set_xticklabels(
            comparison["Metric"],
            rotation=45,
            ha="right",
        )

        ax.set_ylabel(
            "Metric Value",
        )

        ax.set_title("Fairness Metrics Before vs After")

        ax.legend()

        plt.tight_layout()

        st.pyplot(fig)

    @classmethod
    def show(
        cls,
        comparison: pd.DataFrame,
    ):

        st.subheader("Before vs After")

        st.caption("Compare fairness metrics before " "and after mitigation.")

        tab1, tab2 = st.tabs(
            [
                "Comparison Table",
                "Visual Comparison",
            ]
        )

        #
        # Table
        #

        with tab1:

            st.dataframe(
                cls.style(
                    comparison,
                ),
                use_container_width=True,
                hide_index=True,
            )

        #
        # Chart
        #

        with tab2:

            cls.plot(
                comparison,
            )


def comparison(
    comparison_df: pd.DataFrame,
):

    FairnessComparison.show(
        comparison_df,
    )
