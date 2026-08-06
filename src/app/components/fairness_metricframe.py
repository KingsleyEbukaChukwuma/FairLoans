from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


class FairnessMetricFrame:
    """
    Display fairness metrics by protected group.
    """

    @staticmethod
    def table(df: pd.DataFrame):

        numeric = df.select_dtypes(
            include="number",
        ).columns

        st.dataframe(
            df.style.format({column: "{:.3f}" for column in numeric}),
            use_container_width=True,
            hide_index=True,
        )

    @classmethod
    def show(
        cls,
        baseline: pd.DataFrame,
        mitigated: pd.DataFrame,
    ):

        st.subheader("Fairness by Protected Group")

        st.caption("Compare fairness metrics across protected groups before and after bias mitigation.")

        #
        # Dashboard controls
        #

        col1, col2 = st.columns(2)

        with col1:

            model = st.radio(
                "Model",
                [
                    "Baseline",
                    "Mitigated",
                ],
                horizontal=True,
            )

        data = baseline if model == "Baseline" else mitigated

        metric_columns = [
            column
            for column in data.columns
            if column
            not in [
                "Group",
                "Count",
            ]
        ]

        with col2:

            metric = st.selectbox(
                "Metric",
                metric_columns,
            )

        #
        # Summary cards
        #

        c1, c2 = st.columns(2)

        c1.metric(
            "Protected Groups",
            len(data),
        )

        c2.metric(
            "Applicants",
            int(data["Count"].sum()),
        )

        st.divider()

        #
        # Plot
        #

        fig = px.bar(
            data,
            x=metric,
            y="Group",
            orientation="h",
            color=metric,
            text_auto=".3f",
            color_continuous_scale="Blues",
        )

        fig.update_layout(
            template="plotly_white",
            height=450,
            title=f"{metric} ({model})",
            title_x=0.5,
            xaxis_title=metric,
            yaxis_title="Protected Group",
            coloraxis_showscale=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        #
        # Raw data
        #

        with st.expander(
            "View Data Table",
        ):

            cls.table(
                data,
            )


def metricframe(
    baseline: pd.DataFrame,
    mitigated: pd.DataFrame,
):

    FairnessMetricFrame.show(
        baseline,
        mitigated,
    )
