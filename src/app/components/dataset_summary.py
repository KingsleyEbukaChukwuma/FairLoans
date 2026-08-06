from __future__ import annotations

import pandas as pd
import streamlit as st

from src.configs.config import FIELD_LABELS


def dataset_summary(df: pd.DataFrame):

    st.subheader("Summary Statistics")

    #
    # Overview metrics
    #

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Records",
        f"{len(df):,}",
    )

    col2.metric(
        "Features",
        len(df.columns),
    )

    col3.metric(
        "Missing Values",
        int(df.isna().sum().sum()),
    )

    col4.metric(
        "Duplicate Records",
        int(df.duplicated().sum()),
    )

    st.divider()

    #
    # Numeric summary
    #

    st.markdown("#### Numeric Features")

    numeric = df.select_dtypes(include="number")

    if not numeric.empty:

        numeric_summary = numeric.describe().T.round(2)

        numeric_summary.index = [FIELD_LABELS.get(col, col) for col in numeric_summary.index]

        st.dataframe(
            numeric_summary,
            use_container_width=True,
        )

    #
    # Categorical summary
    #

    st.markdown("#### Categorical Features")

    categorical = df.select_dtypes(exclude="number")

    if not categorical.empty:

        summary = pd.DataFrame(
            {
                "Unique Values": categorical.nunique(),
                "Most Frequent": categorical.mode().iloc[0],
                "Frequency": categorical.apply(lambda x: x.value_counts().iloc[0]),
            }
        )

        summary.index = [FIELD_LABELS.get(col, col) for col in summary.index]

        st.dataframe(
            summary,
            use_container_width=True,
        )

    st.divider()

    #
    # Data types
    #

    st.markdown("#### Feature Information")

    info = pd.DataFrame(
        {
            "Feature": [FIELD_LABELS.get(col, col) for col in df.columns],
            "Data Type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
        }
    )

    st.dataframe(
        info,
        use_container_width=True,
        hide_index=True,
    )
