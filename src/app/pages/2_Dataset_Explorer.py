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

#
# Load dataset
#

df = get_dataset()

#
# Sidebar filters
#

with st.sidebar:

    st.header("🔍 Filters")

    filtered = apply_filters(df)

#
# Create tabs
#

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Dataset",
        "📈 Summary",
        "🩹 Missing Values",
        "📊 Distributions",
    ]
)

#
# Dataset
#

with tab1:

    dataset_table(filtered)

#
# Summary
#

with tab2:

    dataset_summary(filtered)

#
# Missing Values
#

with tab3:

    dataset_missing(filtered)

#
# Distributions
#

with tab4:

    dataset_plots(filtered)
