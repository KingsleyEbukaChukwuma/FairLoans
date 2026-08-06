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
    page_title="Responsible AI Dashboard",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ Responsible AI Dashboard")

st.caption("Assess model fairness, evaluate the impact of bias mitigation, and review deployment governance.")


dashboard = get_fairness_dashboard()


fairness_overview(
    dashboard["summary"],
)


st.divider()

executive_summary(
    dashboard["summary"],
)

st.divider()


tab1, tab2, tab3 = st.tabs(
    [
        "📊 Fairness Analysis",
        "📈 Mitigation Impact",
        "🏛️ Governance",
    ]
)


with tab1:

    metricframe(
        baseline=dashboard["baseline"],
        mitigated=dashboard["mitigated"],
    )


with tab2:

    comparison(
        dashboard["comparison"],
    )

with tab3:

    governance(
        dashboard["governance"],
    )
