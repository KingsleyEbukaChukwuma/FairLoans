from __future__ import annotations

import pandas as pd
import streamlit as st


class PerformanceMetrics:
    """
    Display model evaluation metrics.
    """

    @staticmethod
    def show(
        dashboard: dict,
    ) -> None:

        metrics = dashboard["metrics"]

        #
        # Key Performance Indicators
        #

        st.subheader("Key Performance Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "ROC AUC",
                f"{metrics['ROC AUC']:.3f}",
            )

        with col2:

            st.metric(
                "F1 Score",
                f"{metrics['F1']:.3f}",
            )

        with col3:

            st.metric(
                "Balanced Accuracy",
                f"{metrics['Balanced Accuracy']:.3f}",
            )

        with col4:

            st.metric(
                "Brier Score",
                f"{metrics['Brier Score']:.3f}",
            )

        st.divider()

        #
        # Complete Metrics Table
        #

        st.subheader("Complete Evaluation Metrics")

        table = pd.DataFrame(
            metrics.items(),
            columns=[
                "Metric",
                "Value",
            ],
        )

        table["Value"] = table["Value"].apply(
            lambda value: (
                round(value, 4)
                if isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
                else value
            )
        )

        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
        )


def performance_metrics(
    dashboard: dict,
) -> None:

    PerformanceMetrics.show(
        dashboard,
    )
