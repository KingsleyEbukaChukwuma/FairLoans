from pathlib import Path

import streamlit as st


def global_importance(
    image: Path,
):

    st.subheader("Global Feature Importance")

    st.image(
        image,
        use_container_width=True,
    )
