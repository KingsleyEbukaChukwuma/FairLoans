import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def dataset_plots(df):

    st.subheader("Feature Distribution")

    feature = st.selectbox(
        "Select Feature",
        df.columns,
    )

    fig, ax = plt.subplots(
        figsize=(8, 4),
    )

    if pd.api.types.is_numeric_dtype(df[feature]):

        ax.hist(
            df[feature],
            bins=20,
        )

    else:

        df[feature].value_counts().plot.bar(
            ax=ax,
        )

    ax.set_title(feature)

    st.pyplot(fig)
