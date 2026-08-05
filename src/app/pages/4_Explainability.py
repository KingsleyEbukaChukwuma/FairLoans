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
from src.app.components.interaction import (
    interaction,
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

st.caption("Interpret how the model reaches its decisions.")

dashboard = get_explainability_dashboard()

explainability_overview()

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🌍 Global Importance",
        "🐝 Beeswarm",
        "💧 Waterfall",
        "🔗 Interactions",
        "📄 Local Explanation",
    ]
)

with tab1:

    global_importance(
        dashboard["global_importance"],
    )

with tab2:

    beeswarm(
        dashboard["beeswarm"],
    )

with tab3:

    waterfall(
        dashboard["waterfall"],
    )

with tab4:

    interaction(
        dashboard["interaction"],
    )

with tab5:

    local_explanation(
        dashboard["local"],
    )
