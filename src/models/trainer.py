from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

from configs.config import (
    CALIBRATION_METHOD,
    CALIBRATION_CV,
    TRAINING_LOG
)


class ModelTrainer:
    """
    Handles training, calibration and persistence
    of sklearn pipelines.
    """

    def __init__(self, pipeline: Pipeline):

        self.pipeline = pipeline
        self.calibrated_pipeline = None


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

        self.calibrated_pipeline = calibrated

        return self

    @classmethod
    def load(cls, path):

        pipeline = joblib.load(path)

        trainer = cls(pipeline)
        trainer.calibrated_pipeline = pipeline

        return trainer