from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.configs.config import (
    CATEGORY_DISPLAY_MAPS,
    CATEGORY_ORDER,
    DEFAULT_PLOT_FEATURES,
    DISCRETE_NUMERIC_FEATURES,
    FIELD_LABELS,
    HISTOGRAM_FEATURES,
)


def dataset_plots(df: pd.DataFrame):

    st.subheader("Feature Distributions")

    feature_map = {
        FIELD_LABELS.get(
            column,
            column.replace("_", " ").title(),
        ): column
        for column in df.columns
    }

    selected_features = DEFAULT_PLOT_FEATURES.copy()

    with st.expander(
        "Customize or Add More Visualizations",
        expanded=False,
    ):

        selected_labels = st.multiselect(
            "Select Features",
            options=list(feature_map.keys()),
            default=[FIELD_LABELS[f] for f in DEFAULT_PLOT_FEATURES],
        )

        if selected_labels:

            selected_features = [feature_map[label] for label in selected_labels]

    cols = st.columns(2)

    for i, feature in enumerate(selected_features):

        label = FIELD_LABELS.get(
            feature,
            feature.replace("_", " ").title(),
        )

        with cols[i % 2]:

            #
            # Continuous numeric
            #

            if feature in HISTOGRAM_FEATURES:

                fig = px.histogram(
                    df,
                    x=feature,
                    nbins=20,
                    title=label,
                    labels={
                        feature: label,
                        "count": "Applicants",
                    },
                )

            #
            # Small-range integers
            #

            elif feature in DISCRETE_NUMERIC_FEATURES:

                counts = df[feature].value_counts().sort_index().rename_axis(feature).reset_index(name="Count")

                fig = px.bar(
                    counts,
                    x=feature,
                    y="Count",
                    title=label,
                    text_auto=True,
                    labels={
                        feature: label,
                        "Count": "Applicants",
                    },
                )

            #
            # Categorical
            #

            else:

                counts = df[feature].value_counts().rename_axis(feature).reset_index(name="Count")

                display_map = CATEGORY_DISPLAY_MAPS.get(
                    feature,
                    {},
                )

                counts[feature] = counts[feature].replace(display_map)

                if feature in CATEGORY_ORDER:

                    counts[feature] = pd.Categorical(
                        counts[feature],
                        categories=CATEGORY_ORDER[feature],
                        ordered=True,
                    )

                    counts = counts.sort_values(
                        feature,
                    )

                fig = px.bar(
                    counts,
                    x="Count",
                    y=feature,
                    orientation="h",
                    title=label,
                    text_auto=True,
                    labels={
                        feature: label,
                        "Count": "Applicants",
                    },
                )

            fig.update_layout(
                template="plotly_white",
                height=400,
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
