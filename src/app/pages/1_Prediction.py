import streamlit as st

from src.app.components.prediction_form import (
    applicant_form,
)
from src.app.components.prediction_result import (
    show_prediction,
)
from src.app.utils import (
    get_predictor,
)

st.set_page_config(
    page_title="Prediction",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Credit Risk Prediction")

submitted, applicant = applicant_form()

if submitted:

    predictor = get_predictor()

    result = predictor.predict(applicant)

    show_prediction(
        result,
    )
