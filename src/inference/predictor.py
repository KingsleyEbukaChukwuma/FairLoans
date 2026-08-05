from __future__ import annotations

import joblib
import pandas as pd

from configs.config import (
    APPROVAL_THRESHOLD,
    MODEL_DIR,
    PIPELINE_FILENAME,
)


class CreditPredictor:
    """
    Perform inference using the trained FairLoans pipeline.
    """

    def __init__(self):

        self.pipeline = joblib.load(MODEL_DIR / PIPELINE_FILENAME)

    def predict(
        self,
        applicant: dict,
    ) -> dict:
        """
        Predict credit risk for a single applicant.
        """

        X = pd.DataFrame([applicant])

        probability = float(self.pipeline.predict_proba(X)[0, 1])

        prediction = int(probability >= APPROVAL_THRESHOLD)

        decision = "Bad Credit" if prediction == 1 else "Good Credit"

        return {
            "prediction": prediction,
            "decision": decision,
            "probability": probability,
            "risk_level": self.risk_level(probability),
        }

    @staticmethod
    def risk_level(
        probability: float,
    ) -> str:

        if probability < 0.20:

            return "Very Low"

        elif probability < 0.40:

            return "Low"

        elif probability < 0.60:

            return "Moderate"

        elif probability < 0.80:

            return "High"

        else:

            return "Very High"
