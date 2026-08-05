import pandas as pd
import streamlit as st


def local_explanation(
    explanation: dict,
):

    st.subheader("Local Explanation")

    st.json(
        explanation,
        expanded=False,
    )

    if "features" in explanation:

        st.divider()

        st.dataframe(
            pd.DataFrame(explanation["features"]),
            use_container_width=True,
        )
