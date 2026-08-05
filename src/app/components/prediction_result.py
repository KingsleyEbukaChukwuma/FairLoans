import streamlit as st


def show_prediction(result):

    st.subheader("Prediction")

    if result["prediction"] == 0:

        st.success("Good Credit")

    else:

        st.error("Bad Credit")

    st.metric(
        "Probability",
        f"{result['probability']:.1%}",
    )

    st.metric(
        "Risk Level",
        result["risk_level"],
    )
