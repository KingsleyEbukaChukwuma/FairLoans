from pathlib import Path

import streamlit as st


def beeswarm(
    image: Path,
):

    st.subheader("SHAP Beeswarm")

    st.image(
        image,
        use_container_width=True,
    )
