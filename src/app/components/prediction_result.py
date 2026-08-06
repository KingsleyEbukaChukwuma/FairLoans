import streamlit as st


def show_prediction(result):

    st.divider()

    st.subheader("Assessment Result")

    #
    # Decision
    #

    if result["prediction"] == 0:

        st.success("✅ Credit Application Approved")

    else:

        st.error("❌ Credit Application Declined")

    #
    # Key metrics
    #

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Probability of Default",
            f"{result['probability_default']:.1%}",
        )

    with col2:

        st.metric(
            "Decision Threshold",
            f"{result['threshold']:.1%}",
        )

    with col3:

        st.metric(
            "Confidence",
            result["confidence"],
        )

    #
    # Explanation
    #

    if result["prediction"] == 0:

        st.info(f"""
The estimated probability of default is **{result['probability_default']:.1%}**,
which is **below** the decision threshold of **{result['threshold']:.1%}**.

The application is therefore recommended for approval.
""")

    else:

        st.warning(f"""
The estimated probability of default is **{result['probability_default']:.1%}**,
which is **above** the decision threshold of **{result['threshold']:.1%}**.

The application is therefore rejected or recommended for further review.
""")
