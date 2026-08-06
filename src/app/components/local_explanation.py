from __future__ import annotations

import pandas as pd
import streamlit as st

from src.app.helpers.feature_names import clean_feature_name


def local_explanation(
    explanation: dict,
):
    """
    Display a local prediction explanation.
    """

    st.subheader("Prediction Summary")

    #
    # Prediction information
    #

    if "prediction" in explanation:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction",
                explanation["prediction"],
            )

        with col2:

            probability = explanation.get(
                "probability",
            )

            if probability is not None:

                st.metric(
                    "Confidence",
                    f"{probability:.1%}",
                )

    #
    # Feature contributions
    #

    if "features" in explanation:

        df = pd.DataFrame(
            explanation["features"],
        )

        #
        # Clean feature names
        #

        if "Feature" in df.columns:

            df["Feature"] = df["Feature"].apply(
                clean_feature_name,
            )

        #
        # Positive / Negative contributions
        #

        if "Contribution" in df.columns:

            positive = df[df["Contribution"] > 0].sort_values(
                "Contribution",
                ascending=False,
            )

            negative = df[df["Contribution"] < 0].sort_values(
                "Contribution",
            )

            left, right = st.columns(2)

            with left:

                st.success("### Factors Supporting the Prediction")

                st.dataframe(
                    positive,
                    hide_index=True,
                    width="stretch",
                )

            with right:

                st.warning("### Factors Increasing Risk")

                st.dataframe(
                    negative,
                    hide_index=True,
                    width="stretch",
                )

        st.divider()

        st.subheader("Applicant Feature Values")

        st.dataframe(
            df,
            hide_index=True,
            width="stretch",
        )

    #
    # Raw explanation
    #

    with st.expander(
        "View Raw Explanation",
        expanded=False,
    ):

        st.json(
            explanation,
            expanded=False,
        )
