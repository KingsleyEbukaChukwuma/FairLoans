from pathlib import Path

import streamlit as st


def waterfall(
    image: Path,
):

    st.subheader("Waterfall Plot")

    st.image(
        image,
        use_container_width=True,
    )
