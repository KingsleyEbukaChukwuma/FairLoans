from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.configs.config import FIELD_LABELS


def dataset_missing(df: pd.DataFrame):

    st.subheader("Data Quality")

    #
    # Dataset quality metrics
    #

    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    col1, col2 = st.columns(2)

    col1.metric(
        "Missing Values",
        total_missing,
    )

    col2.metric(
        "Duplicate Records",
        duplicate_rows,
    )

    st.divider()

    #
    # Missing values by feature
    #

    missing = pd.DataFrame(
        {
            "Feature": [
                FIELD_LABELS.get(
                    column,
                    column,
                )
                for column in df.columns
            ],
            "Missing Values": df.isna().sum().values,
            "Missing (%)": (df.isna().mean() * 100).round(2).values,
        }
    )

    #
    # Display summary table
    #

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True,
    )

    #
    # Only plot if missing values exist
    #

    if total_missing > 0:

        chart = missing[missing["Missing Values"] > 0]

        fig = px.bar(
            chart,
            x="Missing Values",
            y="Feature",
            orientation="h",
            text_auto=True,
            title="Missing Values by Feature",
        )

        fig.update_layout(
            template="plotly_white",
            height=450,
            margin=dict(
                l=20,
                r=20,
                t=50,
                b=20,
            ),
            title_x=0.5,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.success("✅ No missing values were found in the dataset.")

    #
    # Duplicate records
    #

    if duplicate_rows == 0:

        st.success("✅ No duplicate records were found.")

    else:

        st.warning(f"⚠️ {duplicate_rows} duplicate records detected.")
