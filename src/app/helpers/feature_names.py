from __future__ import annotations

from src.configs.config import (
    CATEGORY_DISPLAY_MAPS,
    FIELD_LABELS,
)


def clean_feature_name(
    feature: str,
    short: bool = False,
) -> str:
    """
    Convert encoded pipeline feature names into
    user-friendly labels.

    Parameters
    ----------
    short
        If True, return only the category value for
        one-hot encoded features.

    Examples
    --------
    numeric__duration
        -> Loan Duration (Months)

    categorical__checking_status_<0
        -> Checking Account Status: Negative Balance

    short=True
        -> Negative Balance
    """

    #
    # Remove sklearn prefixes
    #

    feature = feature.replace(
        "numeric__",
        "",
    ).replace(
        "categorical__",
        "",
    )

    #
    # Numeric feature
    #

    if feature in FIELD_LABELS:

        return FIELD_LABELS[feature]

    #
    # One-hot encoded categorical feature
    #

    for column in CATEGORY_DISPLAY_MAPS:

        prefix = f"{column}_"

        if feature.startswith(prefix):

            value = feature.replace(
                prefix,
                "",
                1,
            )

            label = FIELD_LABELS.get(
                column,
                column.replace(
                    "_",
                    " ",
                ).title(),
            )

            display_value = CATEGORY_DISPLAY_MAPS[column].get(
                value,
                value.replace(
                    "_",
                    " ",
                ).title(),
            )

            #
            # Short labels for plots
            #

            if short:

                return display_value

            #
            # Full labels for tables
            #

            return f"{label}: {display_value}"

    #
    # Fallback
    #

    return feature.replace(
        "_",
        " ",
    ).title()
