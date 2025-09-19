"""Pipeline helpers for the AI Insight dashboard app."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from cache_memory import get_memory_prompt, save_to_cache
from lang_detect import LangDetectAgent
from llm_services import call_narrator_model
from response_gen import compose_prompt
from search_agent_new import run_search_agent

from insight_prompts import get_insight_prompt_sections, use_insight_search_prompt

_lang_agent = LangDetectAgent()


def _build_rows_for_prompt(debug_blob: Dict[str, Any]) -> str:
    """Render docs from search agent debug info into JSON blocks."""
    collections = debug_blob.get("chosen_collections", []) or []
    results = debug_blob.get("results", []) or []
    blocks = []

    for idx, coll in enumerate(collections):
        res = results[idx] if idx < len(results) else {}
        docs = res.get("docs", []) or []
        if not docs:
            blocks.append(f"# {coll}\n(no rows)")
            continue
        doc_blobs = [json.dumps(doc, indent=2, default=str) for doc in docs]
        blocks.append(f"# {coll}\n" + "\n\n".join(f"```json\n{blob}\n```" for blob in doc_blobs))

    return "\n\n".join(blocks) or "(no rows)"


def _extract_primary_rows(debug_blob: Dict[str, Any]) -> list[dict[str, Any]]:
    """Return the first result set as JSON-safe dicts for downstream visuals."""
    results = (debug_blob or {}).get("results", []) or []
    if not results:
        return []

    first = results[0] or {}
    docs = first.get("docs", []) or []
    safe_docs: list[dict[str, Any]] = []

    for doc in docs:
        try:
            serialised = json.loads(json.dumps(doc, default=str))
        except TypeError:
            if hasattr(doc, "items"):
                serialised = json.loads(json.dumps(dict(doc), default=str))
            else:
                serialised = {"value": str(doc)}
        if isinstance(serialised, dict):
            safe_docs.append(serialised)

    return safe_docs


def generate_dashboard_insight(
    question: str,
    user_id: str,
    team_id: str,
    *,
    session_id: Optional[str] = None,
    request_label: str = "dashboard"
) -> Dict[str, Any]:
    """Run search + response flow tailored for dashboard ideation."""
    if not question or not question.strip():
        return {"success": False, "error": "Question is empty."}

    language = _lang_agent.detect_language(question)

    with use_insight_search_prompt():
        spec, raw_answer, debug_blob = run_search_agent(
            user_id=user_id or "anonymous",
            team_id=team_id or "default_team",
            query=question,
            session_id=session_id,
        )

    if isinstance(raw_answer, str) and raw_answer.strip().startswith("⚠️"):
        return {
            "success": False,
            "error": raw_answer,
            "spec": spec,
            "debug": debug_blob,
            "language": language,
            "rows_data": [],
        }

    rows_text = _build_rows_for_prompt(debug_blob)
    rows_data = _extract_primary_rows(debug_blob)
    if rows_text.strip() == "(no rows)" and (not raw_answer or raw_answer.strip() == ""):
        return {
            "success": False,
            "error": "No data returned for the request.",
            "spec": spec,
            "debug": debug_blob,
            "language": language,
            "rows_data": rows_data,
        }

    sections = get_insight_prompt_sections()
    memory_context = get_memory_prompt(3, user_id, team_id, session_id)
    prompt = compose_prompt(
        sections,
        question=f"[{request_label.upper()}] {question}",
        rows=rows_text,
        language=language,
        memory_context=memory_context,
    )

    messages = [{"role": "user", "content": prompt}]
    answer = call_narrator_model(messages, stream=False, temperature=0.2)

    if answer:
        save_to_cache(question, answer, user_id, team_id, session_id=session_id)

    return {
        "success": True,
        "answer": answer,
        "spec": spec,
        "debug": debug_blob,
        "rows_text": rows_text,
        "prompt": prompt,
        "language": language,
        "rows_data": rows_data,
    }
