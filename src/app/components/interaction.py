from pathlib import Path

import streamlit as st


def interaction(
    image: Path,
):

    st.subheader("Interaction Heatmap")

    st.image(
        image,
        use_container_width=True,
    )
