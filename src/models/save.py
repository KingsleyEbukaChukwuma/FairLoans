from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd


def _prepare_path(path: str | Path) -> Path:
    """
    Create parent directories if they do not exist.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def save_model(model, path: str | Path) -> None:
    """
    Save a trained model or pipeline.
    """

    path = _prepare_path(path)

    joblib.dump(
        model,
        path,
    )

    print(f"Model saved to {path}")


def save_study(study, path: str | Path) -> None:
    """
    Save an Optuna study.
    """

    path = _prepare_path(path)

    joblib.dump(
        study,
        path,
    )

    print(f"Study saved to {path}")


def save_json(data: dict, path: str | Path) -> None:
    """
    Save a dictionary as JSON.
    """

    path = _prepare_path(path)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
        )

    print(f"JSON saved to {path}")


def save_dataframe(
    dataframe: pd.DataFrame,
    path: str | Path,
    index: bool = False,
) -> None:
    """
    Save a DataFrame as CSV.
    """

    path = _prepare_path(path)

    dataframe.to_csv(
        path,
        index=index,
    )

    print(f"DataFrame saved to {path}")