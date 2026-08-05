from __future__ import annotations

import json
import joblib
import pandas as pd

from configs.config import (
    MODEL_DIR,
    PIPELINE_FILENAME,
    THRESHOLD_FILENAME,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
)


class CreditPredictor:
    """
    Perform inference using the trained
    FairLoans pipeline.
    """

    def __init__(self):

        self.pipeline = joblib.load(
            MODEL_DIR / PIPELINE_FILENAME
        )

        with open(
            MODEL_DIR / THRESHOLD_FILENAME,
            "r",
        ) as file:

            self.threshold = json.load(
                file
            )["threshold"]

    def predict(
        self,
        applicant: dict,
    ) -> dict:
        """
        Predict the applicant's credit risk.
        """

        X = pd.DataFrame(
            [applicant]
        )

        probability_default = float(

            self.pipeline.predict_proba(
                X
            )[0, 1]

        )

        prediction = int(

            probability_default
            >= self.threshold

        )

        probability_good = (
            1.0 - probability_default
        )

        return {

            "prediction": prediction,

            "decision":

                "Bad Credit"

                if prediction

                else "Good Credit",

            "probability_default":
                probability_default,

            "probability_good":
                probability_good,

            "threshold":
                self.threshold,

            "risk_level":
                self.risk_level(
                    probability_default,
                ),

            "confidence":
                self.confidence(
                    probability_default,
                ),

        }

    @staticmethod
    def risk_level(
        probability: float,
    ) -> str:

        if probability < 0.20:
            return "Very Low"

        if probability < 0.40:
            return "Low"

        if probability < 0.60:
            return "Moderate"

        if probability < 0.80:
            return "High"

        return "Very High"

    @staticmethod
    def confidence(
        probability: float,
    ) -> str:

        if (
            probability >= HIGH_CONFIDENCE
            or probability <= LOW_CONFIDENCE
        ):
            return "High"

        return "Medium"