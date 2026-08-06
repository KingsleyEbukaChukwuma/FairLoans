import streamlit as st

from src.app.components.dataset_filters import (
    apply_filters,
)
from src.app.components.dataset_missing import (
    dataset_missing,
)
from src.app.components.dataset_plots import (
    dataset_plots,
)
from src.app.components.dataset_summary import (
    dataset_summary,
)
from src.app.components.dataset_table import (
    dataset_table,
)
from src.app.utils import get_dataset

st.set_page_config(
    page_title="Dataset Explorer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Dataset Explorer")

st.caption("Explore the German Credit dataset, inspect data quality, and visualize feature distributions.")

#
# Load dataset
#

df = get_dataset()

#
# Tabs
#

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Data Preview",
        "📊 Summary Statistics",
        "📈 Visualizations",
        "🩹 Data Quality",
    ]
)

#
# Dataset Preview
#

with tab1:

    with st.expander(
        "🔍 Filter Dataset",
        expanded=True,
    ):

        filtered = apply_filters(df)

    dataset_table(filtered)

#
# Summary Statistics
#

with tab2:

    dataset_summary(filtered)

#
# Data Quality
#

with tab4:

    dataset_missing(filtered)

#
# Visualizations
#

with tab3:

    dataset_plots(filtered)
