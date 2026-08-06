from __future__ import annotations

import streamlit as st

from src.app.components.diagnostics import (
    diagnostics,
)
from src.app.components.footer import (
    footer,
)
from src.app.components.performance_banner import (
    performance_banner,
)
from src.app.components.performance_best_model import (
    performance_best_model,
)
from src.app.components.performance_metrics import (
    performance_metrics,
)
from src.app.components.performance_overview import (
    performance_overview,
)
from src.app.utils import (
    get_performance_dashboard,
)

#
# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
#

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Model Performance")

st.caption("Evaluate the predictive performance, diagnostic behaviour, and deployment readiness of the selected credit risk model.")


dashboard = get_performance_dashboard()


performance_overview(
    dashboard,
)

st.divider()


performance_banner(
    dashboard,
)

st.divider()


tab1, tab2, tab3 = st.tabs(
    [
        "📊 Performance Metrics",
        "📈 Model Diagnostics",
        "🏆 Best Model",
    ]
)


with tab1:

    performance_metrics(
        dashboard,
    )


with tab2:

    diagnostics(
        dashboard,
    )


with tab3:

    performance_best_model(
        dashboard,
    )

st.divider()


footer(
    dashboard,
)
