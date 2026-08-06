from __future__ import annotations

import pandas as pd
import streamlit as st

from src.configs.config import FIELD_LABELS


def dataset_table(df: pd.DataFrame):

    st.subheader("Data Preview")

    #
    # Controls
    #

    col1, col2 = st.columns([3, 1])

    with col1:

        search = st.text_input(
            "Search",
            placeholder="Search any value...",
        )

    with col2:

        rows = st.selectbox(
            "Rows",
            options=[10, 25, 50, 100, "All"],
            index=1,
        )

    #
    # Apply search
    #

    filtered = df.copy()

    if search:

        mask = filtered.astype(str).apply(
            lambda column: column.str.contains(
                search,
                case=False,
                na=False,
            )
        )

        filtered = filtered[mask.any(axis=1)]

    #
    # Friendly column names
    #

    display = filtered.rename(columns=FIELD_LABELS)

    #
    # Limit displayed rows
    #

    if rows != "All":

        display = display.head(rows)

    st.caption(f"Showing **{len(display):,}** of **{len(filtered):,}** matching records.")

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )
