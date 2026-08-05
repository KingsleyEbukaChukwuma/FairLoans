import streamlit as st


def dataset_table(df):

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
