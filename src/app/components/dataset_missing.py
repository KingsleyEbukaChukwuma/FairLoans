import pandas as pd
import streamlit as st


def dataset_missing(df):

    st.subheader("Missing Values")

    missing = pd.DataFrame(
        {
            "Missing": df.isna().sum(),
            "Percent": (df.isna().mean() * 100).round(2),
        }
    )

    st.dataframe(
        missing,
        use_container_width=True,
    )
