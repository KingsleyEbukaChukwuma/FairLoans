from __future__ import annotations

import streamlit as st

from src.app.components.beeswarm import (
    beeswarm,
)
from src.app.components.explainability_overview import (
    explainability_overview,
)
from src.app.components.global_importance import (
    global_importance,
)
from src.app.components.local_explanation import (
    local_explanation,
)
from src.app.components.waterfall import (
    waterfall,
)
from src.app.utils import (
    get_explainability_dashboard,
)

st.set_page_config(
    page_title="Explainability",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Model Explainability")

st.caption("Understand how the model makes predictions through global and applicant-level explanations.")

#
# Load explainability artifacts
#

dashboard = get_explainability_dashboard()

#
# Overview
#

explainability_overview()

st.divider()

#
# Dashboard Tabs
#

tab1, tab2 = st.tabs(
    [
        "Model Insights",
        "Individual Prediction",
    ]
)

#
# Global Explanations
#

with tab1:

    global_importance(
        dashboard["feature_importance"],
    )

    st.divider()

    beeswarm(
        dashboard["shap_values"],
    )

#
# Local Prediction
#

with tab2:

    waterfall(
        dashboard["shap_values"],
    )

    st.divider()

    local_explanation(
        dashboard["local_explanation"],
    )
