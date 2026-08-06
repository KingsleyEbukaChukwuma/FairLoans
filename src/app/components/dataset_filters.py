from __future__ import annotations

import pandas as pd
import streamlit as st

from src.configs.config import (
    CATEGORY_DISPLAY_MAPS,
    FIELD_LABELS,
)

FILTER_FIELDS = [
    "purpose",
    "credit_amount",
    "duration",
    "checking_status",
    "credit_history",
    "employment",
    "age",
    "housing",
]


def apply_filters(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply interactive dataset filters.
    """

    filtered = df.copy()

    st.caption("Use the filters below to explore different groups of applicants.")

    cols = st.columns(2)

    for i, column in enumerate(FILTER_FIELDS):

        with cols[i % 2]:

            label = FIELD_LABELS.get(
                column,
                column.replace("_", " ").title(),
            )

            #
            # Numeric filters
            #

            if pd.api.types.is_numeric_dtype(df[column]):

                minimum = int(df[column].min())
                maximum = int(df[column].max())

                values = st.slider(
                    label,
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

            #
            # Categorical filters
            #

            else:

                raw_options = sorted(df[column].dropna().unique().tolist())

                display_map = CATEGORY_DISPLAY_MAPS.get(
                    column,
                    {},
                )

                display_options = [
                    display_map.get(
                        option,
                        option,
                    )
                    for option in raw_options
                ]

                selected_display = st.multiselect(
                    label,
                    options=display_options,
                    default=display_options,
                )

                reverse_map = {display_map.get(option, option): option for option in raw_options}

                selected_raw = [reverse_map[value] for value in selected_display]

                filtered = filtered[filtered[column].isin(selected_raw)]

    st.divider()

    st.metric(
        "Matching Records",
        f"{len(filtered):,}",
        help="Number of records matching the selected filters.",
    )

    return filtered
