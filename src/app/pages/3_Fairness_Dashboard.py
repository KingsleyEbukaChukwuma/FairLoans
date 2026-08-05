from __future__ import annotations

import streamlit as st

from src.app.components.executive_summary import (
    executive_summary,
)
from src.app.components.fairness_comparison import (
    comparison,
)
from src.app.components.fairness_metricframe import (
    metricframe,
)
from src.app.components.fairness_overview import (
    fairness_overview,
)
from src.app.components.governance import (
    governance,
)
from src.app.utils import (
    get_fairness_dashboard,
)

st.set_page_config(
    page_title="Fairness Dashboard",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Responsible AI Dashboard")

st.caption("Evaluate fairness, bias mitigation " "and governance of the selected model.")

#
# -------------------------------------------------------
# Load all dashboard artifacts
# -------------------------------------------------------
#

dashboard = get_fairness_dashboard()

#
# -------------------------------------------------------
# KPI Cards
# -------------------------------------------------------
#

fairness_overview(
    dashboard["summary"],
)

st.divider()

#
# -------------------------------------------------------
# Dashboard Tabs
# -------------------------------------------------------
#

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Fairness by Group",
        "⚖️ Before vs After",
        "📋 Governance",
        "📌 Executive Summary",
    ]
)

#
# -------------------------------------------------------
# MetricFrame
# -------------------------------------------------------
#

with tab1:

    metricframe(
        baseline=dashboard["baseline"],
        mitigated=dashboard["mitigated"],
    )

#
# -------------------------------------------------------
# Comparison
# -------------------------------------------------------
#

with tab2:

    comparison(
        dashboard["comparison"],
    )

#
# -------------------------------------------------------
# Governance
# -------------------------------------------------------
#

with tab3:

    governance(
        dashboard["governance"],
    )

#
# -------------------------------------------------------
# Executive Summary
# -------------------------------------------------------
#

with tab4:

    executive_summary(
        dashboard["summary"],
    )
