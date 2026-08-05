from __future__ import annotations

import streamlit as st

from src.app.field_groups import FIELD_GROUPS
from src.app.utils import get_schema


def pretty_name(feature: str) -> str:
    """
    Convert feature names into readable labels.
    """

    return feature.replace("_", " ").title()


def applicant_form():

    schema = get_schema()

    applicant = {}

    with st.form(
        "prediction_form",
        clear_on_submit=False,
    ):

        st.header("Applicant Information")

        #
        # Loop through business sections
        #

        for section, fields in FIELD_GROUPS.items():

            st.subheader(section)

            cols = st.columns(2)

            for i, feature in enumerate(fields):

                with cols[i % 2]:

                    #
                    # Numeric feature
                    #

                    if feature in schema["numeric"]:

                        applicant[feature] = st.number_input(
                            label=pretty_name(feature),
                            value=0.0,
                        )

                    #
                    # Categorical feature
                    #

                    else:

                        applicant[feature] = st.selectbox(
                            label=pretty_name(feature),
                            options=schema["categorical"][feature],
                        )

            st.divider()

        submitted = st.form_submit_button(
            "Predict Credit Risk",
            use_container_width=True,
        )

    return submitted, applicant
