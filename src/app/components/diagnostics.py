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
        caption: str,
    ) -> None:
        """
        Display a diagnostic image if it exists.
        """

        st.markdown(f"### {title}")

        if path.exists():

            st.image(
                path,
                width="stretch",
            )

            st.caption(
                caption,
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
        # Classification Diagnostics
        #

        st.subheader("Classification Diagnostics")

        col1, col2 = st.columns(2)

        with col1:

            cls.image(
                dashboard["roc"],
                "ROC Curve",
                ("Shows the trade-off between the true positive rate " "and false positive rate across all classification thresholds."),
            )

        with col2:

            cls.image(
                dashboard["confusion"],
                "Confusion Matrix",
                ("Summarises the number of correct and incorrect " "predictions for each class."),
            )

        st.divider()

        #
        # Calibration
        #

        st.subheader("Probability Calibration")

        _, center, _ = st.columns([1, 2, 1])

        with center:

            cls.image(
                dashboard["calibration"],
                "Calibration Curve",
                ("Compares predicted probabilities with observed outcomes " "to assess how well the model is calibrated."),
            )


def diagnostics(
    dashboard: dict,
) -> None:
    """
    Display model diagnostics.
    """

    Diagnostics.show(
        dashboard,
    )
