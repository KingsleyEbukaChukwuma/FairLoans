from __future__ import annotations

import pandas as pd

from configs.config import MODEL_SELECTION_WEIGHTS


HIGHER_IS_BETTER = [
    "ROC AUC",
    "KS",
    "MCC",
    "F1",
    "PR AUC",
]

LOWER_IS_BETTER = [
    "Log Loss",
    "Brier Score",
]


def normalize(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Min-max normalize a metric to the range [0, 1].
    """

    minimum = series.min()
    maximum = series.max()

    if minimum == maximum:
        return pd.Series(
            1.0,
            index=series.index,
        )

    if higher_is_better:
        return (series - minimum) / (maximum - minimum)

    return (maximum - series) / (maximum - minimum)


def score_models(results: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a weighted overall score for every model.
    """

    results = results.copy()

    results["Overall Score"] = 0.0

    for metric in HIGHER_IS_BETTER:

        normalized = normalize(
            results[metric],
            higher_is_better=True,
        )

        results["Overall Score"] += (
            normalized
            * MODEL_SELECTION_WEIGHTS[metric]
        )

    for metric in LOWER_IS_BETTER:

        normalized = normalize(
            results[metric],
            higher_is_better=False,
        )

        results["Overall Score"] += (
            normalized
            * MODEL_SELECTION_WEIGHTS[metric]
        )

    return results


def select_best(results):
    """
    Select the highest-scoring model.
    """

    df = pd.DataFrame(results)

    df = score_models(df)

    df = df.sort_values(
        "Overall Score",
        ascending=False,
    )

    return df.iloc[0]