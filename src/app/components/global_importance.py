from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.app.helpers.feature_names import clean_feature_name


def global_importance(
    importance: pd.DataFrame,
):
    """
    Display the global feature importance as an
    interactive Plotly chart.
    """

    st.subheader("Feature Importance")

    st.caption("Features ranked by their overall contribution to the model's predictions.")

    #
    # Copy dataframe
    #

    df = importance.copy()

    #
    # Clean feature names
    #

    df["Feature"] = df["Feature"].apply(
        clean_feature_name,
    )

    #
    # Sort so the largest feature appears at the top
    #

    df = df.sort_values(
        "Importance",
        ascending=True,
    )

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=".3f",
    )

    fig.update_layout(
        template="plotly_white",
        height=700,
        title="Global Feature Importance",
        title_x=0.5,
        xaxis_title="Importance",
        yaxis_title="",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )
