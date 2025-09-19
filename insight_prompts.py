"""Prompt assets for the AI Insight dashboard experience."""

from __future__ import annotations

import json
import textwrap
from contextlib import contextmanager

import search_agent_new
from langchain_core.prompts import ChatPromptTemplate
from pydantic import model_validator

INSIGHT_SEARCH_AGENT_PROMPT = textwrap.dedent(
    """
    You are a business intelligence query planner preparing dataset slices for interactive dashboards.
    Build precise MongoDB JSON specs that highlight the measures, dimensions, and time ranges designers can plug into charts.

    Available collections and field operations:
    {schema_section}

    Guidance:
    • Prefer aggregations, sorts, and limits that surface the most decision-ready records.
    • Never guess field values. Only use explicit values or enumerations from the schema description.
    • When the user mentions trends, ensure date filters or sorts reflect that intent.
    • Resolve pronouns with the latest memory context before building filters.

    Output format:
    • Return a single JSON object (no code fences) describing collection, filters, joins, sort, and limit.
    • Include joins only when information truly spans collections; keep them minimal and justified by the request.
    • For superlative requests ("top", "highest", "lowest"), add a matching sort and appropriate limit.

    Memory context (use to disambiguate entities and follow-up intents):
    <CONVERSATION_HISTORY>

    Reminder:
    • Acceptable operations: regex, keyword (array only), range, sort.
    • Dates must use the provided ISO format.
    • Do not invent collections or fields.
    • Respect row-level guardrails already enforced downstream; just describe the query faithfully.
    Today is {today_date}. Use it when the user references relative dates.
    """
)

INSIGHT_MAIN_PROMPT = textwrap.dedent(
    """
    You are an AI Insight Architect helping data teams design dashboards.
    Turn the retrieved `rows` into narrative insights that explain what should appear on the dashboard.

    Key duties:
    1. Identify the primary business question hidden in the user request.
    2. Summarize the most dashboard-worthy metrics and categorizations from `rows`.
    3. Suggest how to visualize the insight (e.g., KPI tile, time-series, bar comparison) and why it matters.
    4. If data is missing, clearly state the gap and suggest additional queries.
    5. Always match the user's language when responding; hinge to Hinglish when detected.

    Style:
    • Start with a one-sentence insight that answers the request directly.
    • Follow with a bold heading and 2–3 crisp sentences clarifying the story.
    • Provide a short bullet list (max 4) of supporting facts or segment cuts ready for dashboards.
    • Close with a "Next visual move" recommendation describing the chart type or layout.
    • Sprinkle relevant emojis (📊📈🧭) only when they aid clarity.
    • Never fabricate metrics; everything must be traceable to `rows`.
    """
)

INSIGHT_FORMATTING_PROMPT = textwrap.dedent(
    """
    Formatting expectations:
    • Keep paragraphs short (≤3 sentences).
    • Use bold labels for key numbers or segment names.
    • Prefer inline statistics over tables unless the question explicitly asks for tabular output.
    • When suggesting visuals, name both the chart type and the axes or slice (e.g., "Stacked bar by region vs. revenue").
    • If no data is returned, respond with a gentle prompt to refine the request for dashboard design.
    """
)

INSIGHT_ANALYSIS_PROMPT = textwrap.dedent(
    """
    Analysis checklist:
    • Compare categories or time periods when possible to reveal trends.
    • Highlight anomalies, outliers, or noteworthy thresholds the dashboard should call out.
    • Translate raw numbers into business impact (growth, declines, share of total, coverage).
    • If multiple collections were joined, clarify how each contributes to the finding.
    """
)

INSIGHT_CONTEXT_PROMPT = textwrap.dedent(
    """
    Context reminders:
    • Respect any access warnings or empty results — never imply data you cannot see.
    • Leverage MEMORY_CONTEXT for follow-up questions and to maintain continuity.
    • Encourage the user to iterate on the dashboard spec if insights feel incomplete.
    """
)

INSIGHT_PROMPT_SECTIONS = {
    "main": INSIGHT_MAIN_PROMPT,
    "formatting": INSIGHT_FORMATTING_PROMPT,
    "analysis": INSIGHT_ANALYSIS_PROMPT,
    "context": INSIGHT_CONTEXT_PROMPT,
}


def get_insight_prompt_sections() -> dict[str, str]:
    """Return a copy of the insight prompt sections."""
    return dict(INSIGHT_PROMPT_SECTIONS)


@contextmanager
def use_insight_search_prompt():
    """Temporarily swap in the insight search prompt for run_search_agent."""
    original_system_prompt = search_agent_new.SYSTEM_PROMPT
    original_search_prompt = getattr(search_agent_new, "SEARCH_AGENT_SYSTEM_PROMPT", None)
    original_template = getattr(search_agent_new, "PROMPT", None)
    original_collection_query = search_agent_new.CollectionQuery
    original_multi_query = search_agent_new.MultiEntityQuery

    def _normalise_filters(raw_filters):
        normalised = []
        if isinstance(raw_filters, dict):
            items = raw_filters.items()
        elif isinstance(raw_filters, list):
            return raw_filters
        else:
            return []

        for field, condition in items:
            if condition is None:
                continue
            owner = None
            if isinstance(condition, dict):
                condition_map = dict(condition)

                # Owner-aware filters: {"field": {"owner": "alias", "$gte": ...}}
                if "owner" in condition_map:
                    owner = condition_map.pop("owner")

                if any(str(key).startswith("$") for key in condition_map.keys()):
                    normalised.append({
                        "field": field,
                        "operation": "range",
                        "value": condition_map,
                        **({"owner": owner} if owner else {})
                    })
                    continue

                if len(condition_map) == 1 and not any(str(key).startswith("$") for key in condition_map.keys()):
                    inner_key, inner_value = next(iter(condition_map.items()))
                    if isinstance(inner_key, str) and inner_key.startswith("$"):
                        normalised.append({
                            "field": field,
                            "operation": "range",
                            "value": condition_map,
                            **({"owner": owner} if owner else {})
                        })
                        continue
                    condition = inner_value
                else:
                    condition = condition_map

            if isinstance(condition, dict):
                cond_value = json.dumps(condition)
            elif isinstance(condition, (list, tuple)):
                cond_value = ", ".join(str(v) for v in condition if v is not None)
            else:
                cond_value = condition

            normalised.append({
                "field": field,
                "operation": "regex",
                "value": cond_value,
                **({"owner": owner} if owner else {})
            })

        return normalised

    def _normalise_joins(raw_joins):
        if not raw_joins:
            return []
        if isinstance(raw_joins, dict):
            raw_list = [raw_joins]
        else:
            raw_list = list(raw_joins)

        cleaned = []
        for idx, join in enumerate(raw_list):
            if not isinstance(join, dict):
                continue

            alias = join.get("alias") or join.get("as")
            collection = join.get("collection") or join.get("from") or join.get("target")
            local_field = join.get("local_field") or join.get("localField")
            foreign_field = join.get("foreign_field") or join.get("foreignField")
            from_alias = join.get("from_") or join.get("fromAlias") or join.get("source")

            # Derive alias if missing
            if not alias and collection:
                alias = f"join_{idx}_{collection}"

            if not from_alias:
                from_alias = join.get("from") if join.get("collection") else "base"

            if isinstance(local_field, str) and "." in local_field:
                owner_hint, remainder = local_field.split(".", 1)
                local_field = remainder
                if from_alias in (None, "base") and owner_hint:
                    from_alias = owner_hint

            if not collection or not local_field or not foreign_field:
                continue

            cleaned.append({
                "alias": alias,
                "collection": collection,
                "from_": from_alias or "base",
                "local_field": local_field,
                "foreign_field": foreign_field,
                "cardinality": join.get("cardinality") or "one_to_many",
                "join_type": join.get("join_type") or join.get("joinType") or "left",
            })

        return cleaned

    class InsightCollectionQuery(original_collection_query):
        @model_validator(mode="before")
        def _coerce_fields(cls, data):
            if not isinstance(data, dict):
                return data

            data["filters"] = _normalise_filters(data.get("filters"))

            sort_block = data.get("sort") or {}
            if isinstance(sort_block, dict):
                fixed_sort = {}
                for key, value in sort_block.items():
                    if isinstance(value, int):
                        fixed_sort[key] = "asc" if value >= 0 else "desc"
                    elif isinstance(value, str):
                        cleaned = value.strip().lower()
                        if cleaned in {"1", "+1", "asc"}:
                            fixed_sort[key] = "asc"
                        elif cleaned in {"-1", "desc"}:
                            fixed_sort[key] = "desc"
                        else:
                            fixed_sort[key] = cleaned or "asc"
                    else:
                        fixed_sort[key] = "asc"
                data["sort"] = fixed_sort
            else:
                data["sort"] = {}

            data["joins"] = _normalise_joins(data.get("joins"))

            limit_val = data.get("limit")
            if isinstance(limit_val, str):
                try:
                    data["limit"] = int(limit_val)
                except ValueError:
                    pass

            return data

    class InsightMultiEntityQuery(original_multi_query):
        @model_validator(mode="before")
        def _coerce_collection_queries(cls, data):
            if not isinstance(data, dict):
                return data
            queries = data.get("queries")
            if isinstance(queries, dict):
                data["queries"] = [queries]
            return data

    search_agent_new.SYSTEM_PROMPT = INSIGHT_SEARCH_AGENT_PROMPT
    if original_search_prompt is not None:
        search_agent_new.SEARCH_AGENT_SYSTEM_PROMPT = INSIGHT_SEARCH_AGENT_PROMPT
    if original_template is not None:
        search_agent_new.PROMPT = ChatPromptTemplate.from_messages([
            ("system", INSIGHT_SEARCH_AGENT_PROMPT),
            ("placeholder", "{messages}")
        ])
    search_agent_new.CollectionQuery = InsightCollectionQuery
    search_agent_new.MultiEntityQuery = InsightMultiEntityQuery

    try:
        yield
    finally:
        search_agent_new.SYSTEM_PROMPT = original_system_prompt
        if original_search_prompt is not None:
            search_agent_new.SEARCH_AGENT_SYSTEM_PROMPT = original_search_prompt
        if original_template is not None:
            search_agent_new.PROMPT = original_template
        search_agent_new.CollectionQuery = original_collection_query
        search_agent_new.MultiEntityQuery = original_multi_query
