"""
Data loading utilities for FairLoans.

Supports:
- ARFF (German Credit dataset)
- CSV (future datasets)
"""

from pathlib import Path

import pandas as pd
from scipy.io import arff


class DataLoader:
    """Load datasets into pandas DataFrames."""

    @staticmethod
    def load_arff(path: str | Path) -> pd.DataFrame:
        """
        Load an ARFF dataset and decode byte strings.

        Parameters
        ----------
        path : str | Path
            Path to ARFF file.

        Returns
        -------
        pd.DataFrame
        """

        data, _ = arff.loadarff(path)

        df = pd.DataFrame(data)

        object_columns = df.select_dtypes(include="object").columns

        for col in object_columns:
            df[col] = df[col].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )

        return df

    @staticmethod
    def load_csv(path: str | Path) -> pd.DataFrame:
        return pd.read_csv(path)
