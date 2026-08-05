from __future__ import annotations

from pathlib import Path

import streamlit as st


class Diagnostics:
    """
    Display model diagnostic plots.
    """

    @staticmethod
    def image(
        path: Path,
        title: str,
    ) -> None:
        """
        Display an image if it exists.
        """

        st.subheader(
            title,
        )

        if path.exists():

            st.image(
                path,
                use_container_width=True,
            )

        else:

            st.warning(f"{title} not found.")

    @classmethod
    def show(
        cls,
        dashboard: dict,
    ) -> None:
        """
        Display all diagnostic plots.
        """

        #
        # Classification diagnostics
        #

        st.subheader("Classification Diagnostics")

        col1, col2 = st.columns(2)

        with col1:

            cls.image(
                dashboard["roc"],
                "ROC Curve",
            )

        with col2:

            cls.image(
                dashboard["confusion"],
                "Confusion Matrix",
            )

        st.divider()

        #
        # Calibration diagnostics
        #

        _, center, _ = st.columns([1, 2, 1])

        with center:

            cls.image(
                dashboard["calibration"],
                "Calibration Curve",
            )


def diagnostics(
    dashboard: dict,
) -> None:
    """
    Display all model diagnostics.
    """

    Diagnostics.show(
        dashboard,
    )
