from pathlib import Path

import joblib
import pandas as pd


class CreditPredictor:

    def __init__(self, model_path):

        self.pipeline = joblib.load(model_path)

    def predict(self, applicant: dict):

        df = pd.DataFrame([applicant])

        prediction = self.pipeline.predict(df)[0]

        probability = self.pipeline.predict_proba(df)[0][1]

        return {

            "prediction": int(prediction),

            "probability_default": float(probability),

            "probability_good": float(1 - probability)

        }