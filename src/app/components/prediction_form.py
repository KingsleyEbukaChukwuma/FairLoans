from __future__ import annotations

import streamlit as st

from src.app.field_groups import FIELD_GROUPS
from src.app.utils import get_schema
from src.configs.config import (
    CATEGORY_DISPLAY_MAPS,
    FIELD_LABELS,
    NUMERIC_CONFIG,
)


def applicant_form():

    schema = get_schema()

    applicant = {}

    with st.form(
        "prediction_form",
        clear_on_submit=False,
    ):

        for section, fields in FIELD_GROUPS.items():

            st.subheader(section)

            cols = st.columns(2)

            for i, feature in enumerate(fields):

                with cols[i % 2]:

                    label = FIELD_LABELS.get(
                        feature,
                        feature.replace("_", " ").title(),
                    )

                    #
                    # Numeric features
                    #

                    if feature in schema["numeric"]:

                        cfg = NUMERIC_CONFIG.get(
                            feature,
                            {},
                        )

                        applicant[feature] = st.number_input(
                            label=label,
                            min_value=cfg.get("min", 0),
                            max_value=cfg.get("max"),
                            value=cfg.get("value", 0),
                            step=cfg.get("step", 1),
                            help=cfg.get("help"),
                        )

                    #
                    # Categorical features
                    #

                    else:

                        raw_options = schema["categorical"][feature]

                        display_map = CATEGORY_DISPLAY_MAPS.get(
                            feature,
                            {},
                        )

                        #
                        # Display values shown to user
                        #

                        display_options = [
                            display_map.get(
                                option,
                                option,
                            )
                            for option in raw_options
                        ]

                        selected_display = st.selectbox(
                            label=label,
                            options=display_options,
                        )

                        #
                        # Convert displayed value back to
                        # original value expected by the model
                        #

                        reverse_map = {display_map.get(option, option): option for option in raw_options}

                        applicant[feature] = reverse_map[selected_display]

            st.divider()

        submitted = st.form_submit_button(
            "Predict Credit Risk",
            type="primary",
            use_container_width=True,
        )

    return submitted, applicant
