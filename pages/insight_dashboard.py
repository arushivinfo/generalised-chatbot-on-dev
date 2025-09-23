"""Streamlit dashboard page for visualising AI-generated insights."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from insight_dashboard_view import render_dashboard_view

st.set_page_config(page_title="Insight Dashboard", page_icon="📊", layout="wide")

insight_runs: List[Dict[str, Any]] = st.session_state.setdefault("insight_runs", [])
st.session_state["dashboard_runs"] = insight_runs
runs = insight_runs

user_id = st.session_state.get("user_name", "analyst")
team_id = st.session_state.get("team_id", "default_team")
session_id = st.session_state.get("dashboard_session_id", st.session_state.get("sid", ""))
st.session_state["dashboard_session_id"] = session_id
render_dashboard_view(
    runs,
    user_id=user_id,
    team_id=team_id,
    session_id=session_id,
    allow_generate=True,
    show_back_button=False,
)
