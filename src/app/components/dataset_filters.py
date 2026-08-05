from __future__ import annotations

import pandas as pd
import streamlit as st


def apply_filters(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply interactive filters.
    """

    filtered = df.copy()

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):

            minimum = float(df[column].min())

            maximum = float(df[column].max())

            values = st.slider(
                label=column,
                min_value=minimum,
                max_value=maximum,
                value=(minimum, maximum),
            )

            filtered = filtered[
                filtered[column].between(
                    values[0],
                    values[1],
                )
            ]

        else:

            options = sorted(df[column].dropna().unique())

            values = st.multiselect(
                label=column,
                options=options,
                default=options,
            )

            filtered = filtered[filtered[column].isin(values)]

    return filtered
