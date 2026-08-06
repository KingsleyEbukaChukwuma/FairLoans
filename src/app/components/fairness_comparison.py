from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


class FairnessComparison:
    """
    Compare fairness metrics before and after mitigation.
    """

    @staticmethod
    def summary(
        comparison: pd.DataFrame,
    ):

        improved = (comparison["Outcome"] == "Improved").sum()

        worse = (comparison["Outcome"] == "Worse").sum()

        unchanged = (comparison["Outcome"] == "Unchanged").sum()

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Metrics Improved",
            improved,
        )

        c2.metric(
            "Metrics Worse",
            worse,
        )

        c3.metric(
            "Unchanged",
            unchanged,
        )

    @staticmethod
    def table(
        comparison: pd.DataFrame,
    ):

        st.subheader("Metric Comparison")

        display = comparison.copy()

        #
        # Format numeric values
        #

        for column in [
            "Before",
            "After",
        ]:

            display[column] = display[column].map(lambda x: f"{x:.3f}")

        display["Improvement (%)"] = display["Improvement (%)"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")

        display = display[
            [
                "Category",
                "Metric",
                "Before",
                "After",
                "Improvement (%)",
                "Outcome",
            ]
        ]

        #
        # Outcome colouring
        #

        def colour_outcome(value):

            if value == "Improved":

                return "background-color:#d4edda;" "color:#155724;" "font-weight:bold;"

            if value == "Worse":

                return "background-color:#fdebd0;" "color:#9c640c;" "font-weight:bold;"

            return "background-color:#ecf0f1;" "color:#566573;" "font-weight:bold;"

        st.dataframe(
            display.style.map(
                colour_outcome,
                subset=["Outcome"],
            ),
            use_container_width=True,
            hide_index=True,
        )

    @staticmethod
    def chart(
        comparison: pd.DataFrame,
    ):

        st.subheader("Improvement by Metric")

        chart = comparison.sort_values(
            "Improvement (%)",
            ascending=False,
        )

        fig = px.bar(
            chart,
            x="Improvement (%)",
            y="Metric",
            orientation="h",
            color="Outcome",
            text="Improvement (%)",
            color_discrete_map={
                "Improved": "#2ca02c",
                "Worse": "#f39c12",
                "Unchanged": "#95a5a6",
            },
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        fig.update_layout(
            template="plotly_white",
            height=650,
            title="Impact of Bias Mitigation",
            title_x=0.5,
            xaxis_title="Improvement (%)",
            yaxis_title="",
            legend_title="Outcome",
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    @classmethod
    def show(
        cls,
        comparison: pd.DataFrame,
    ):

        st.subheader("Mitigation Impact")

        st.caption("Compare fairness and performance before and after bias mitigation.")

        cls.summary(
            comparison,
        )

        st.divider()

        cls.table(
            comparison,
        )

        st.divider()

        cls.chart(
            comparison,
        )


def comparison(
    comparison_df: pd.DataFrame,
):

    FairnessComparison.show(
        comparison_df,
    )
