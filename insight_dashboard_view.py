"""Shared dashboard rendering helpers for AI insight results."""

from __future__ import annotations

import html
import json
import time
import re
import textwrap
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from insight_pipeline import generate_dashboard_insight
from insight_visuals import ALT_AVAILABLE, build_charts, build_kpis

_DASHBOARD_STYLES = """
<style>
body {background-color: #f5f7fb;}
.card {
    background: #ffffff;
    border-radius: 22px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.1rem;
    box-shadow: 0 22px 46px rgba(15,23,42,0.08);
    border: 1px solid rgba(99,102,241,0.16);
}
.card__header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 1rem;
}
.card__meta {
    font-size: 0.78rem;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.14em;
}
.card__title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e1b4b;
    margin: 0.2rem 0 0 0;
}
.hero-banner {
    background: linear-gradient(135deg, #4c51bf, #6366f1);
    color: #eef2ff;
    padding: 1.6rem 1.9rem;
    border-radius: 20px;
    margin-bottom: 1.4rem;
    box-shadow: 0 22px 44px rgba(76,81,191,0.28);
}
.hero-banner h1 {margin: 0; font-size: 1.7rem;}
.hero-banner p {margin: 0.6rem 0 0 0; font-size: 1rem; color: #e0e7ff;}
.filter-card {
    background: rgba(79,70,229,0.08);
    border-radius: 18px;
    padding: 1.2rem 1rem;
    border: 1px solid rgba(79,70,229,0.25);
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.2);
}
.filter-card h3 {
    margin: 0 0 0.8rem 0;
    font-size: 1rem;
    color: #312e81;
}
.kpi-card {
    background: #f8fafc;
    border-radius: 18px;
    padding: 1rem 1.2rem;
    border: 1px solid rgba(226,232,240,0.9);
    box-shadow: 0 18px 32px rgba(148,163,184,0.18);
    height: 100%;
}
.kpi-card__label {
    font-size: 0.9rem;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.08em;
}
.kpi-card__value {
    display: block;
    font-size: 2.4rem;
    font-weight: 700;
    color: #111827;
    margin-top: 0.35rem;
}
.chart-card {
    background: #ffffff;
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    border: 1px solid rgba(226,232,240,0.9);
    box-shadow: 0 18px 30px rgba(148,163,184,0.16);
    height: 100%;
}
.chart-card__title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.6rem;
}
.history-title {
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin: 1.8rem 0 0.6rem 0;
}
.metric-help {font-size: 0.74rem; color: #6b7280; margin-top: 0.35rem;}
.session-chips {
    display: flex;
    gap: 0.9rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 1.2rem 0;
}
.session-chip {
    background: linear-gradient(135deg, rgba(79,70,229,0.85), rgba(59,130,246,0.85));
    color: #f8fafc;
    padding: 0.55rem 0.9rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    box-shadow: 0 12px 25px rgba(59,130,246,0.25);
}
.session-chip span {
    opacity: 0.8;
    font-weight: 500;
    margin-right: 0.35rem;
}
.cta-button > button {
    background: linear-gradient(135deg, #f97316, #fb7185) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 18px 32px rgba(251, 113, 133, 0.35);
    font-weight: 700;
}
.cta-button > button:hover {
    transform: translateY(-1px);
}
.flashcards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.2rem;
    margin: 1rem 0 1.6rem 0;
}
.flashcard {
    position: relative;
    background: linear-gradient(155deg, rgba(248,250,252,0.96), rgba(229,231,235,0.82));
    border-radius: 20px;
    padding: 1.2rem 1.35rem;
    border: 1px solid rgba(148,163,184,0.22);
    box-shadow: 0 28px 54px rgba(148,163,184,0.24);
    font-size: 0.94rem;
    line-height: 1.55;
    color: #0f172a;
    font-weight: 500;
    backdrop-filter: blur(8px);
    min-height: 160px;
    display: flex;
    align-items: flex-start;
}
.flashcard::after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 55%),
                radial-gradient(circle at bottom right, rgba(249,115,22,0.18), transparent 45%);
    opacity: 0.55;
    pointer-events: none;
}
.flashcard span {
    position: relative;
    z-index: 1;
}
.flashcard__badge {
    position: relative;
    display: inline-block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    background: rgba(15,23,42,0.12);
    color: #0f172a;
    margin-bottom: 0.65rem;
    font-weight: 700;
}
.flashcard__body {
    position: relative;
    display: block;
    font-size: 1.05rem;
    line-height: 1.6;
}
.flashcard--highlight .flashcard__badge {
    background: rgba(79,70,229,0.16);
    color: #312e81;
}
.flashcard--highlight {
    background: linear-gradient(155deg, rgba(129,140,248,0.18), rgba(79,70,229,0.28));
    color: #111827;
    font-weight: 600;
    box-shadow: 0 30px 60px rgba(79,70,229,0.25);
}
.flashcard--highlight::after {
    background: radial-gradient(circle at top right, rgba(255,255,255,0.35), transparent 60%),
                radial-gradient(circle at bottom left, rgba(129,140,248,0.25), transparent 40%);
    opacity: 0.75;
}
.flashcard__divider {
    position: relative;
    margin: 0.45rem 0 0.65rem 0;
    height: 1px;
    background: linear-gradient(90deg, rgba(148,163,184,0.4), rgba(148,163,184,0.08));
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.1rem;
    margin-bottom: 1.3rem;
}
.kpi-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(160deg, rgba(79,70,229,0.12), rgba(59,130,246,0.12));
    border-radius: 20px;
    padding: 1.4rem 1.45rem;
    border: 1px solid rgba(129,140,248,0.25);
    box-shadow: 0 28px 58px rgba(129,140,248,0.28);
    color: #111827;
}
.kpi-card::before {
    content: "";
    position: absolute;
    inset: -30% 40% 55% -35%;
    background: radial-gradient(circle at center, rgba(255,255,255,0.45), transparent 70%);
    opacity: 0.8;
}
.kpi-card__label {
    position: relative;
    font-size: 0.78rem;
    text-transform: uppercase;
    color: #475569;
    letter-spacing: 0.18em;
    font-weight: 700;
}
.kpi-card__value {
    position: relative;
    display: block;
    font-size: 2.7rem;
    font-weight: 700;
    margin-top: 0.65rem;
}
.kpi-card__delta {
    position: relative;
    margin-top: 0.55rem;
    font-size: 0.85rem;
    font-weight: 600;
}
.metric-help {
    font-size: 0.78rem;
    color: #475569;
    margin-top: 0.45rem;
}
</style>
"""

DASHBOARD_CSS = """
<style>
/* Enhanced KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.kpi-card {
    position: relative;
    overflow: hidden;
    background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 24px;
    padding: 2rem 1.8rem;
    border: 1px solid rgba(226,232,240,0.8);
    box-shadow: 
        0 20px 40px rgba(148,163,184,0.12),
        0 8px 24px rgba(148,163,184,0.08),
        inset 0 1px 0 rgba(255,255,255,0.9);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(12px);
}

.kpi-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
    border-radius: 24px 24px 0 0;
}

.kpi-card::after {
    content: "";
    position: absolute;
    top: -50%;
    right: -30%;
    width: 100px;
    height: 100px;
    background: radial-gradient(circle at center, rgba(99,102,241,0.1), transparent 70%);
    opacity: 0.6;
    transition: opacity 0.3s ease;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 
        0 32px 64px rgba(148,163,184,0.15),
        0 12px 32px rgba(148,163,184,0.1),
        inset 0 1px 0 rgba(255,255,255,0.9);
}

.kpi-card:hover::after {
    opacity: 0.8;
}

.kpi-card__label {
    position: relative;
    z-index: 2;
    font-size: 0.85rem;
    text-transform: uppercase;
    color: #64748b;
    letter-spacing: 0.1em;
    font-weight: 600;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.kpi-card__label::before {
    content: "📊";
    font-size: 1rem;
}

.kpi-card__value {
    position: relative;
    z-index: 2;
    display: block;
    font-size: 2.8rem;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 0.5rem;
    line-height: 1.1;
    background: linear-gradient(135deg, #1e293b 0%, #475569 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.kpi-card__delta {
    position: relative;
    z-index: 2;
    margin-top: 0.8rem;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 0.4rem 0.8rem;
    border-radius: 12px;
    background: rgba(34,197,94,0.1);
    color: #15803d;
    display: inline-block;
}

.metric-help {
    position: relative;
    z-index: 2;
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 1rem;
    line-height: 1.4;
    font-style: italic;
}

/* Enhanced Flashcards/Capsules */
.flashcards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 1.5rem;
    margin: 1.5rem 0 2rem 0;
}

.flashcard {
    position: relative;
    background: linear-gradient(145deg, rgba(255,255,255,0.9) 0%, rgba(248,250,252,0.9) 100%);
    border-radius: 20px;
    padding: 1.8rem 1.6rem;
    border: 1px solid rgba(226,232,240,0.6);
    box-shadow: 
        0 20px 40px rgba(148,163,184,0.1),
        0 8px 24px rgba(148,163,184,0.06);
    font-size: 1rem;
    line-height: 1.6;
    color: #1e293b;
    font-weight: 500;
    backdrop-filter: blur(16px);
    min-height: 180px;
    display: flex;
    flex-direction: column;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
}

.flashcard::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        radial-gradient(circle at 20% 80%, rgba(99,102,241,0.05), transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(168,85,247,0.05), transparent 50%);
    pointer-events: none;
}

.flashcard:hover {
    transform: translateY(-6px);
    box-shadow: 
        0 32px 64px rgba(148,163,184,0.15),
        0 12px 32px rgba(148,163,184,0.08);
}

.flashcard__badge {
    position: relative;
    z-index: 2;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.6rem 1rem;
    border-radius: 50px;
    background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(168,85,247,0.1));
    color: #4338ca;
    margin-bottom: 1rem;
    font-weight: 700;
    border: 1px solid rgba(99,102,241,0.2);
    width: fit-content;
}

.flashcard__divider {
    position: relative;
    z-index: 2;
    margin: 1rem 0;
    height: 2px;
    background: linear-gradient(90deg, rgba(99,102,241,0.3), rgba(168,85,247,0.3), transparent);
    border-radius: 2px;
}

.flashcard__body {
    position: relative;
    z-index: 2;
    flex-grow: 1;
    font-size: 1.1rem;
    line-height: 1.6;
    color: #374151;
}

/* Specialized Flashcard Variants */
.flashcard--highlight {
    background: linear-gradient(145deg, rgba(99,102,241,0.08) 0%, rgba(168,85,247,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.3);
    box-shadow: 
        0 24px 48px rgba(99,102,241,0.15),
        0 12px 24px rgba(99,102,241,0.08);
}

.flashcard--highlight .flashcard__badge {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
    color: #3730a3;
    border-color: rgba(99,102,241,0.4);
}

.flashcard--risk {
    background: linear-gradient(145deg, rgba(239,68,68,0.06) 0%, rgba(252,165,165,0.06) 100%);
    border-color: rgba(239,68,68,0.2);
}

.flashcard--risk .flashcard__badge {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(252,165,165,0.15));
    color: #dc2626;
    border-color: rgba(239,68,68,0.3);
}

.flashcard--trend {
    background: linear-gradient(145deg, rgba(34,197,94,0.06) 0%, rgba(134,239,172,0.06) 100%);
    border-color: rgba(34,197,94,0.2);
}

.flashcard--trend .flashcard__badge {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(134,239,172,0.15));
    color: #059669;
    border-color: rgba(34,197,94,0.3);
}

.flashcard--action {
    background: linear-gradient(145deg, rgba(249,115,22,0.06) 0%, rgba(253,186,116,0.06) 100%);
    border-color: rgba(249,115,22,0.2);
}

.flashcard--action .flashcard__badge {
    background: linear-gradient(135deg, rgba(249,115,22,0.15), rgba(253,186,116,0.15));
    color: #ea580c;
    border-color: rgba(249,115,22,0.3);
}

.flashcard--insight {
    background: linear-gradient(145deg, rgba(6,182,212,0.06) 0%, rgba(103,232,249,0.06) 100%);
    border-color: rgba(6,182,212,0.2);
}

.flashcard--insight .flashcard__badge {
    background: linear-gradient(135deg, rgba(6,182,212,0.15), rgba(103,232,249,0.15));
    color: #0891b2;
    border-color: rgba(6,182,212,0.3);
}

/* Responsive Design */
@media (max-width: 768px) {
    .kpi-grid {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .flashcards {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .kpi-card {
        padding: 1.5rem 1.2rem;
    }
    
    .kpi-card__value {
        font-size: 2.2rem;
    }
    
    .flashcard {
        padding: 1.5rem 1.2rem;
        min-height: 160px;
    }
}

/* Dark mode support */
@media (prefers-color-scheme: dark) {
    .kpi-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-color: rgba(71,85,105,0.8);
        color: #e2e8f0;
    }
    
    .kpi-card__value {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .flashcard {
        background: linear-gradient(145deg, rgba(30,41,59,0.9) 0%, rgba(15,23,42,0.9) 100%);
        border-color: rgba(71,85,105,0.6);
        color: #e2e8f0;
    }
}

/* Animation keyframes */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.kpi-card, .flashcard {
    animation: fadeInUp 0.6s ease-out;
}

.kpi-card:nth-child(2) { animation-delay: 0.1s; }
.kpi-card:nth-child(3) { animation-delay: 0.2s; }
.flashcard:nth-child(2) { animation-delay: 0.1s; }
.flashcard:nth-child(3) { animation-delay: 0.2s; }
.flashcard:nth-child(4) { animation-delay: 0.3s; }
.flashcard:nth-child(5) { animation-delay: 0.4s; }
</style>
"""

def _ensure_styles_injected() -> None:
    if st.session_state.get("_dashboard_styles_injected"):
        return
    st.markdown(_DASHBOARD_STYLES, unsafe_allow_html=True)
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)  # Add this line
    st.session_state["_dashboard_styles_injected"] = True


def render_dashboard_view(
    runs: List[Dict[str, Any]],
    *,
    user_id: str,
    team_id: str,
    session_id: Optional[str],
    allow_generate: bool = True,
    show_back_button: bool = False,
    back_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Render the dashboard experience for previously generated runs."""

    _ensure_styles_injected()
    runs = runs or []

    if show_back_button:
        back_col, _ = st.columns([1, 5])
        if back_col.button("← Back to Insight Studio", use_container_width=True, key="back_to_studio_sidebar"):
            if back_callback:
                back_callback()
            else:
                st.query_params["view"] = "studio"
            st.rerun()

    if allow_generate and not show_back_button:
        cta_col, _, _ = st.columns([1.2, 2, 2])
        with cta_col:
            st.markdown("<div class='cta-button'>", unsafe_allow_html=True)
            if st.button("← Back to Insight Studio", use_container_width=True, key="back_to_studio_main"):
                st.query_params["view"] = "studio"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    if not runs:
        st.info("No insights yet. Return to the studio to create your first dashboard.")
        return

    def _render_dashboard_card(
        run: Dict[str, Any],
        display_idx: int,
        *,
        show_query_expander: bool = True,
    ) -> None:
        payload = run.get("result") or {}
        title_text = run.get("question", "")
        created_at = run.get("created_at")
        stamp = created_at.strftime("%Y-%m-%d %H:%M UTC") if isinstance(created_at, datetime) else ""

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="card__header">
                <div>
                    <p class="card__meta">Insight #{display_idx}</p>
                    <h2 class="card__title">{html.escape(title_text)}</h2>
                </div>
                <span class="card__meta">{stamp}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not payload.get("success", False):
            st.error(payload.get("error", "Unable to generate insight."))
            if payload.get("spec"):
                with st.expander("Generated Spec", expanded=False):
                    st.json(payload["spec"])
            st.markdown("</div>", unsafe_allow_html=True)
            return
        rows_data = payload.get("rows_data", []) or []
        planner_kpis = payload.get("kpi_cards") or []
        kpi_cards = planner_kpis if planner_kpis else build_kpis(rows_data)[:3]

        capsules_plan = payload.get("capsules_data")
        answer = payload.get("answer")
        if capsules_plan or answer:
            st.markdown("#### Insight Capsules")
            _render_flashcards(answer, rows_data, kpi_cards, capsules_plan)

        if kpi_cards:
            st.markdown("#### KPI Highlights")
            cards_html = ["<div class='kpi-grid'>"]
            for metric in kpi_cards:
                label = html.escape(metric.get("label", ""))
                value = html.escape(str(metric.get("value", "")))
                detail = metric.get("help") or metric.get("detail")
                detail_html = (
                    f"<div class='kpi-card__delta'>{html.escape(detail)}</div>"
                    if detail
                    else ""
                )
                cards_html.append(
                    "<div class='kpi-card'>"
                    + f"<span class='kpi-card__label'>{label}</span>"
                    + f"<span class='kpi-card__value'>{value}</span>"
                    + detail_html
                    + "</div>"
                )
            cards_html.append("</div>")
            st.markdown("".join(cards_html), unsafe_allow_html=True)

        charts = payload.get("chart_data")
        if not charts:
            charts = build_charts(rows_data)
        if charts:
            st.markdown("#### Visuals")
            for start in range(0, len(charts), 2):
                row = charts[start:start + 2]
                cols = st.columns(len(row))
                for col, pack in zip(cols, row):
                    col.markdown(
                        f"""
                        <div class="chart-card">
                            <div class="chart-card__title">{html.escape(pack['title'])}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    col.altair_chart(pack["chart"], use_container_width=True)
                    col.markdown("</div>", unsafe_allow_html=True)
        elif rows_data and not ALT_AVAILABLE:
            st.info("Install Altair + pandas to unlock chart previews.")

        st.markdown("#### Data Preview")
        if rows_data:
            preview = rows_data[: min(len(rows_data), 50)]
            st.dataframe(preview)
        else:
            st.info("No documents returned for this insight.")

        meta = payload.get("debug", {})
        meta_cols = st.columns(3)
        meta_cols[0].metric("Language", payload.get("language", "-"))
        meta_cols[1].metric("Collections", ", ".join(meta.get("chosen_collections", []) or ["-"]))
        restricted = ", ".join(meta.get("restricted_collections", []) or ["None"])
        meta_cols[2].metric("Restricted", restricted)

        if show_query_expander:
            with st.expander("Query & Prompt", expanded=False):
                st.markdown("**Query Spec**")
                spec_obj = payload.get("spec") or {}
                if spec_obj:
                    st.json(spec_obj)
                else:
                    st.info("Spec unavailable")
                st.markdown("**Prompt**")
                st.code(payload.get("prompt", ""), language="markdown")
        else:
            st.markdown("**Query Spec**")
            spec_obj = payload.get("spec") or {}
            if spec_obj:
                st.json(spec_obj)
            else:
                st.info("Spec unavailable")
            st.markdown("**Prompt**")
            st.code(payload.get("prompt", ""), language="markdown")

        st.markdown("</div>", unsafe_allow_html=True)

    latest_display_number = len(runs)
    _render_dashboard_card(runs[0], latest_display_number)

    if len(runs) > 1:
        st.markdown("<div class='history-title'>Previous Dashboards</div>", unsafe_allow_html=True)
        for offset, historical in enumerate(runs[1:], start=1):
            label = html.escape(historical.get("question", ""))
            with st.expander(f"Insight #{latest_display_number - offset}: {label[:80]}"):
                _render_dashboard_card(
                    historical,
                    latest_display_number - offset,
                    show_query_expander=False,
                )


def _render_flashcards(
    answer: Optional[str],
    rows_data: List[dict],
    kpis: List[dict],
    capsules_data: Optional[List[Dict[str, Any]]] = None,
) -> None:
    icon_map = {
        "Risk": "⚠️",
        "Trend": "📈",
        "Action": "✅",
        "Insight": "💡",
    }

    def _class_for(badge: str) -> str:
        return {
            "Risk": "flashcard--risk",
            "Trend": "flashcard--trend",
            "Action": "flashcard--action",
            "Visual": "flashcard--visual",
            "Insight": "flashcard--insight",
        }.get(badge, "flashcard--insight")

    def _short_text(text: str, limit: int = 11) -> str:
        words = text.split()
        if len(words) > limit:
            return " ".join(words[:limit]) + " …"
        return text

    if not answer:
        return

    def _has_digit(text: str) -> bool:
        return bool(re.search(r"\d", text))

    def _fallback_capsules() -> list[dict]:
        capsules: list[dict] = []
        if rows_data:
            count = len(rows_data)
            capsules.append({
                "badge": "Insight",
                "text": f"Rows analysed: {count}",
                "detail": f"Processed {count} records in current insight run."
            })
        for metric in kpis:
            label = metric.get("label", "Metric")
            value = metric.get("value", "-")
            helper = metric.get("help") or ""
            badge = "Trend"
            label_lower = label.lower()
            if any(token in label_lower for token in ("risk", "pending", "failed")):
                badge = "Risk"
            elif any(token in label_lower for token in ("action", "focus", "follow")):
                badge = "Action"
            capsules.append({
                "badge": badge,
                "text": f"{label}: {value}",
                "detail": helper or f"{label} equals {value} across returned rows."
            })
            if len(capsules) >= 5:
                break
        return capsules[:5]

    json_capsules = None
    match = re.search(r"INSIGHT_CAPSULES:\s*```json\s*(\{.*?\})\s*```", answer, re.DOTALL)
    if not match:
        match = re.search(r"INSIGHT_CAPSULES:\s*(\{.*?\})", answer, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            raw_capsules = data.get("capsules") or []
            json_capsules = [
                {
                    "badge": cap.get("badge", "Insight"),
                    "text": cap.get("text", ""),
                    "detail": cap.get("detail", cap.get("text", "")),
                }
                for cap in raw_capsules
                if cap.get("text")
            ][:5]
        except Exception:
            json_capsules = None

    if json_capsules:
        capsules = []
        for cap in json_capsules:
            text = cap.get("text", "")
            detail = cap.get("detail", text)
            if not _has_digit(text) and detail:
                detail_words = detail
                if _has_digit(detail_words):
                    text = _short_text(detail_words)
            capsules.append({
                "badge": cap.get("badge", "Insight"),
                "text": _short_text(text),
                "detail": detail,
            })

        if not _has_digit(capsules[0]["text"]) and _has_digit(capsules[0]["detail"]):
            capsules[0]["text"] = _short_text(capsules[0]["detail"])

        highlight_cap = capsules[0]
        rest_caps = capsules[1:]

        cards_html = [
            "<div class='flashcard flashcard--highlight "
            + _class_for(highlight_cap["badge"])
            + "' title='"
            + html.escape(highlight_cap.get("detail", highlight_cap["text"]))
            + "'>"
            + f"<span class='flashcard__badge'>{html.escape(icon_map.get(highlight_cap['badge'], '💡'))} {html.escape(highlight_cap['badge'])}</span>"
            + "<div class='flashcard__divider'></div>"
            + f"<span class='flashcard__body'><strong>{html.escape(highlight_cap['text'])}</strong></span>"
            + "</div>"
        ]

        for cap in rest_caps:
            cards_html.append(
                "<div class='flashcard "
                + _class_for(cap["badge"])
                + "' title='"
                + html.escape(cap.get("detail", cap["text"]))
                + "'>"
                + f"<span class='flashcard__badge'>{html.escape(icon_map.get(cap['badge'], '💡'))} {html.escape(cap['badge'])}</span>"
                + "<div class='flashcard__divider'></div>"
                + f"<span class='flashcard__body'><strong>{html.escape(cap['text'])}</strong></span>"
                + "</div>"
            )

        st.markdown("<div class='flashcards'>" + "".join(cards_html) + "</div>", unsafe_allow_html=True)
        return

    cleaned_answer = re.sub(r"INSIGHT_CAPSULES:.*?```", "", answer, flags=re.DOTALL)
    cleaned_text = cleaned_answer.replace("**", " ").replace("__", " ")
    cleaned_text = re.sub(r"`{1,3}", " ", cleaned_text)

    sentences: list[str] = []
    for block in re.split(r"\n+", cleaned_text):
        block = block.strip()
        if not block:
            continue
        block = re.sub(r"^[\-•#>]+\s*", "", block)
        for sentence in re.split(r"(?<=[.!?])\s+", block):
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if not sentence or sentence in {"-", "--"}:
                continue
            sentences.append(sentence)
            if len(sentences) >= 8:
                break
        if len(sentences) >= 8:
            break

    if not sentences:
        return

    fallback_caps = _fallback_capsules()
    if fallback_caps:
        highlight_cap = fallback_caps[0]
        rest_caps = fallback_caps[1:]
        cards_html = [
            "<div class='flashcard flashcard--highlight "
            + _class_for(highlight_cap["badge"])
            + "' title='"
            + html.escape(highlight_cap.get("detail", highlight_cap["text"]))
            + "'>"
            + f"<span class='flashcard__badge'>{html.escape(icon_map.get(highlight_cap['badge'], '💡'))} {html.escape(highlight_cap['badge'])}</span>"
            + "<div class='flashcard__divider'></div>"
            + f"<span class='flashcard__body'><strong>{html.escape(_short_text(highlight_cap['text']))}</strong></span>"
            + "</div>"
        ]
        for cap in rest_caps:
            cards_html.append(
                "<div class='flashcard "
                + _class_for(cap["badge"])
                + "' title='"
                + html.escape(cap.get("detail", cap["text"]))
                + "'>"
                + f"<span class='flashcard__badge'>{html.escape(icon_map.get(cap['badge'], '💡'))} {html.escape(cap['badge'])}</span>"
                + "<div class='flashcard__divider'></div>"
                + f"<span class='flashcard__body'><strong>{html.escape(_short_text(cap['text']))}</strong></span>"
                + "</div>"
            )
        st.markdown("<div class='flashcards'>" + "".join(cards_html) + "</div>", unsafe_allow_html=True)
        return

    # Fallback to sentence-based extraction if no KPIs available
    highlight_sentence = sentences[0]
    highlight = _short_text(highlight_sentence)
    highlight_badge = "Insight"
    rest_sentences = sentences[1:5]

    cards_html = [
        "<div class='flashcard flashcard--highlight flashcard--insight' title='"
        + html.escape(highlight_sentence)
        + "'>"
        + "<span class='flashcard__badge'>💡 Insight</span>"
        + "<div class='flashcard__divider'></div>"
        + f"<span class='flashcard__body'><strong>{html.escape(highlight)}</strong></span>"
        + "</div>"
    ]
    for sentence in rest_sentences:
        cards_html.append(
            "<div class='flashcard flashcard--insight' title='"
            + html.escape(sentence)
            + "'>"
            + "<span class='flashcard__badge'>💡 Insight</span>"
            + "<div class='flashcard__divider'></div>"
            + f"<span class='flashcard__body'><strong>{html.escape(_short_text(sentence))}</strong></span>"
            + "</div>"
        )
    st.markdown("<div class='flashcards'>" + "".join(cards_html) + "</div>", unsafe_allow_html=True)
