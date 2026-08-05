import streamlit as st


def dataset_summary(df):

    st.subheader("Summary Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        len(df),
    )

    c2.metric(
        "Columns",
        len(df.columns),
    )

    c3.metric(
        "Missing",
        int(df.isna().sum().sum()),
    )

    c4.metric(
        "Duplicates",
        int(df.duplicated().sum()),
    )

    st.divider()

    st.dataframe(
        df.describe(
            include="all",
        ),
        use_container_width=True,
    )
