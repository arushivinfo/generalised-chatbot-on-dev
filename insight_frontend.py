"""Streamlit UI for the AI Insight dashboard workflow."""

from __future__ import annotations

import html
import time
from datetime import datetime
from typing import Any, Dict

import streamlit as st

from insight_pipeline import generate_dashboard_insight
from insight_visuals import ALT_AVAILABLE, build_charts, build_kpis

st.set_page_config(page_title="AI Insight Studio", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .insight-banner {
        background: linear-gradient(120deg, #4338ca, #6366f1);
        border-radius: 18px;
        padding: 1rem 1.4rem;
        color: #f9fafb;
        margin-bottom: 0.8rem;
        box-shadow: 0 18px 40px rgba(67, 56, 202, 0.25);
    }
    .insight-banner__badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        background: rgba(255, 255, 255, 0.18);
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 0.6rem;
    }
    .insight-banner__question {
        font-size: 1.05rem;
        font-weight: 600;
        line-height: 1.4;
    }
    .insight-callout {
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        background: linear-gradient(140deg, rgba(16,185,129,0.18), rgba(5,150,105,0.12));
        border: 1px solid rgba(16,185,129,0.35);
        color: #064e3b;
        margin-bottom: 1.1rem;
    }
    .insight-callout.error {
        background: linear-gradient(140deg, rgba(239,68,68,0.18), rgba(220,38,38,0.12));
        border-color: rgba(239,68,68,0.35);
        color: #7f1d1d;
    }
    .insight-meta {
        font-size: 0.85rem;
        opacity: 0.85;
    }
    div[data-testid="stExpander"] > div:first-child {
        background-color: rgba(79, 70, 229, 0.08);
        color: #312e81;
    }
    div[data-testid="stExpander"] > div:first-child:hover {
        background-color: rgba(79, 70, 229, 0.16);
    }
    div[data-testid="stExpander"] > div:first-child p {
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(79, 70, 229, 0.18);
        border-radius: 12px !important;
        margin-bottom: 0.8rem;
        overflow: hidden;
    }
    .insight-section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1e1b4b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "insight_runs" not in st.session_state:
    st.session_state.insight_runs = []
if "insight_session_id" not in st.session_state:
    st.session_state.insight_session_id = ""

with st.sidebar:
    st.header("Session")
    user_id = st.text_input("User ID", value="anonymous")
    team_id = st.text_input("Team ID", value="default_team")
    session_id = st.text_input(
        "Session ID (optional)",
        value=st.session_state.insight_session_id,
        help="Persist insights across runs by reusing the same session identifier.",
    )
    st.session_state.insight_session_id = session_id

    st.markdown("---")
    st.caption("Each insight request generates a tailored database query and narrative summary for dashboard ideation.")

st.title("AI Insight Studio")
st.caption("Describe your desired dashboard experience and let the assistant gather the supporting data + story.")

with st.form("insight_request"):
    request_text = st.text_area(
        "What dashboard or insight do you need?",
        placeholder="Example: Compare monthly revenue and churn for our SaaS plans over the last two quarters.",
        height=140,
    )
    submitted = st.form_submit_button("Generate Insight", use_container_width=True)

if submitted:
    cleaned_request = (request_text or "").strip()
    if not cleaned_request:
        st.warning("Please describe the dashboard you need before submitting.")
    else:
        st.session_state.insight_runs.insert(0, {
            "question": cleaned_request,
            "status": "pending",
            "result": None,
            "created_at": datetime.utcnow(),
        })
        idx = 0
        with st.spinner("Collecting data slices and crafting insight..."):
            result = generate_dashboard_insight(
                cleaned_request,
                user_id=user_id,
                team_id=team_id,
                session_id=session_id or None,
                request_label="dashboard",
            )
        st.session_state.insight_runs[idx]["status"] = "ready"
        st.session_state.insight_runs[idx]["result"] = result
        st.session_state.last_dashboard_prompt = cleaned_request
        st.session_state.last_dashboard_generated_at = time.time()

for run_idx, run in enumerate(st.session_state.insight_runs):
    display_idx = len(st.session_state.insight_runs) - run_idx
    question_label = html.escape(run.get("question", ""))
    result = run.get("result") or {}
    success = result.get("success", False)

    card = st.container()
    with card:
        st.markdown(
            f"""
            <div class="insight-banner">
                <div class="insight-banner__badge">Insight Request #{display_idx}</div>
                <div class="insight-banner__question">{question_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not success:
            callout = result.get("error", "Unable to generate insight.")
            st.markdown(
                f"""<div class=\"insight-callout error\">⚠️ {html.escape(callout)}</div>""",
                unsafe_allow_html=True,
            )
            if result.get("spec"):
                with st.expander("🔧 Generated Spec", expanded=False):
                    st.json(result["spec"])
            st.divider()
            continue

        meta = {
            "language": result.get("language"),
            "collections": result.get("debug", {}).get("chosen_collections", []),
            "restricted": result.get("debug", {}).get("restricted_collections", []),
        }

        st.markdown(
            """
            <div class="insight-callout">
                ✅ Insight generated successfully.<br>
                <span class="insight-meta">Use the drops below to inspect data, prompts, and specs.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p class='insight-section-title'>Narrative Insight</p>", unsafe_allow_html=True)
        st.markdown(result.get("answer", "(no answer)"))

        rows_data = result.get("rows_data", []) or []
        kpis = build_kpis(rows_data)
        if kpis:
            st.markdown("<p class='insight-section-title'>Insight KPIs</p>", unsafe_allow_html=True)
            col_count = min(3, len(kpis))
            for start in range(0, len(kpis), col_count):
                cols = st.columns(col_count)
                for col, item in zip(cols, kpis[start : start + col_count]):
                    col.metric(item["label"], item["value"], item.get("delta"))
                    if item.get("help"):
                        col.caption(item["help"])

        charts = build_charts(rows_data)
        if charts:
            st.markdown("<p class='insight-section-title'>Visual Explorations</p>", unsafe_allow_html=True)
            for pack in charts:
                st.markdown(f"**{pack['title']}**")
                st.altair_chart(pack["chart"], use_container_width=True)
        elif rows_data and not ALT_AVAILABLE:
            st.info("Install Altair to unlock automatic chart suggestions.")

        spec_obj: Dict[str, Any] = result.get("spec") or {}
        with st.expander("🔧 Query Specification", expanded=True):
            if spec_obj:
                st.json(spec_obj)
            else:
                st.code("Spec unavailable", language="text")

        with st.expander("📂 Retrieved Data", expanded=False):
            st.markdown(result.get("rows_text", "(no rows)"))

        with st.expander("🧠 Prompt Trace", expanded=False):
            st.code(result.get("prompt", ""), language="markdown")

        st.markdown("<p class='insight-section-title'>Search Metadata</p>", unsafe_allow_html=True)
        st.json(meta)

    st.divider()

if not st.session_state.insight_runs:
    st.info("Submit a dashboard request to see data-backed insights here.")

if st.session_state.get("insight_runs"):
    st.markdown(
        """
        <div style="margin:1rem 0 1.2rem 0;padding:0.9rem 1.1rem;border-radius:14px;background:rgba(79,70,229,0.12);border:1px solid rgba(99,102,241,0.25);">
            <strong>Need a full dashboard view?</strong> Jump to the dedicated page for KPI tiles and chart layouts.
        </div>
        """,
        unsafe_allow_html=True,
    )
    dashboard_page = "pages/insight_dashboard.py"
    try:
        st.page_link(dashboard_page, label="Open Dashboard Studio", icon="📊")
    except Exception:
        st.link_button("Open Dashboard Studio", "./insight_dashboard")
