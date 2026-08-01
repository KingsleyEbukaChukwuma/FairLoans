from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from configs.config import (
    CALIBRATION_METHOD,
    CALIBRATION_CV,
)


class ModelTrainer:
    """
    Handles training, calibration and persistence
    of sklearn pipelines.
    """

    def __init__(self, pipeline: Pipeline):

        self.pipeline = pipeline

    def fit(self, X_train, y_train):

        self.pipeline.fit(X_train, y_train)

        return self

    def calibrate(self, X_train, y_train):

        calibrated = CalibratedClassifierCV(
            self.pipeline,
            method=CALIBRATION_METHOD,
            cv=CALIBRATION_CV,
        )

        calibrated.fit(X_train, y_train)

        self.pipeline = calibrated

        return self


    @classmethod
    def load(cls, path):

        pipeline = joblib.load(path)

        return cls(pipeline)