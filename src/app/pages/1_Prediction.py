import streamlit as st

from src.app.components.prediction_form import applicant_form
from src.app.components.prediction_result import show_prediction
from src.app.utils import get_predictor

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
)

st.title("💳 FairLoans Credit Risk Assessment")
st.caption("Complete the applicant's information below to evaluate the likelihood of credit default.")

submitted, applicant = applicant_form()

if submitted:

    predictor = get_predictor()

    with st.spinner("Assessing applicant..."):

        result = predictor.predict(applicant)

    show_prediction(result)
