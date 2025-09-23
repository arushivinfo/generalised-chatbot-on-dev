"""LLM-driven dashboard planner utilities."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

# from llm_services import call_dashboard_planner_model
from insight_visuals import (
    ALT_AVAILABLE,
    alt,
    pd,
    _as_number,
    _collect_field_examples,
    _detect_category_field,
    _detect_date_field,
    _detect_numeric_fields,
    _to_chart_data,
)

# ── Summaries ---------------------------------------------------------------

IDENTIFIER_TOKENS = {"id", "phone", "contact", "mobile", "number", "code", "postal", "zip", "ssn"}


def _looks_identifier(field: str, values: List[Any]) -> bool:
    name = (field or "").lower()
    if any(token in name for token in IDENTIFIER_TOKENS):
        return True

    numeric_values = []
    for raw in values:
        num = _as_number(raw)
        if num is None:
            return False
        numeric_values.append(num)

    if not numeric_values:
        return False

    ints = [value for value in numeric_values if float(value).is_integer()]
    if len(ints) != len(numeric_values):
        return False

    unique_ratio = len(set(ints)) / max(len(ints), 1)
    if unique_ratio <= 0.9:
        return False

    avg_length = mean(len(str(int(abs(x)))) for x in ints if x is not None)
    return avg_length >= 8


def _summarise_numeric(field: str, values: List[Any]) -> str:
    nums = [n for v in values if (n := _as_number(v)) is not None]
    if not nums:
        return ""
    return (
        f"min={min(nums):,.2f}, max={max(nums):,.2f}, avg={mean(nums):,.2f}, "
        f"count={len(nums)}"
    )


def _summarise_categorical(field: str, values: List[Any], max_items: int = 5) -> str:
    counter: Counter[str] = Counter(str(v) for v in values if v not in (None, ""))
    common = counter.most_common(max_items)
    return ", ".join(f"{val} ({count})" for val, count in common)


def summarise_rows(rows: List[Dict[str, Any]], max_fields: int = 6) -> str:
    if not rows:
        return "No rows returned."

    samples = _collect_field_examples(rows)
    field_names = list(samples.keys())[:max_fields]
    summary_parts: List[str] = [f"Total rows analysed: {len(rows)}", ""]

    for field in field_names:
        values = samples[field]
        if not values:
            continue
        snippet = []
        if _looks_identifier(field, values):
            summary_parts.append(f"- {field}: identifier-like (excluded from metrics)")
            continue

        if field in _detect_numeric_fields({field: values}):
            stats = _summarise_numeric(field, values)
            summary_parts.append(f"- {field} [numeric]: {stats}")
        elif _detect_date_field({field: values}):
            summary_parts.append(f"- {field} [date/time]")
        else:
            cats = _summarise_categorical(field, values)
            summary_parts.append(f"- {field} [categorical]: {cats}")
    return "\n".join(summary_parts)


# ── Planner ----------------------------------------------------------------

PLANNER_SYSTEM_PROMPT = """You are a senior analytics planner creating dashboard summaries.
Given a user question, query metadata, and field summaries, choose which metrics and charts are most
insightful. Only use fields that exist in the data. Prefer numeric measures for KPIs and charts.
Avoid identifier fields (IDs, contact numbers, codes) for metrics or visuals.
Always return valid JSON.
"""

PLANNER_USER_TEMPLATE = """User question: {question}

Query details:
{spec_summary}

Field summaries:
{field_summary}

Output JSON with keys:
- "capsules": list of up to 5 objects with fields {"badge", "text", "detail"}. `text` must be ≤12 words and contain at least one numeric fact when possible. Badge must be one of ["Trend","Risk","Action","Insight"].
- "kpis": list of up to 3 objects with fields {"label","field","stat","help","format"}. Use numeric stats (sum, avg, count, max, min, count_distinct). Avoid identifiers.
- "charts": list of up to 2 objects with fields {"type","x","y","group","agg","title"}. Allowed chart types: line, bar, area, pie. Use numeric measures on axes that make sense.

Return only JSON, no extra text.
"""


def plan_dashboard(
    question: str,
    rows: List[Dict[str, Any]],
    spec: Dict[str, Any],
    debug_blob: Dict[str, Any],
) -> Dict[str, Any]:
    if not rows:
        return {}

    field_summary = summarise_rows(rows)
    spec_summary = json.dumps(spec or {}, indent=2)[:2000]

    user_prompt = PLANNER_USER_TEMPLATE.format(
        question=question,
        spec_summary=spec_summary,
        field_summary=field_summary,
    )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # try:
    #     # response = call_dashboard_planner_model(messages, temperature=0.1)
    #     # if isinstance(response, str):
    #     #     plan = json.loads(response)
    #     # else:
    #     #     plan = json.loads(response.get("content"))
    #     if not isinstance(plan, dict):
    #         raise ValueError("Planner response is not a dict")
    #     return plan
    # except Exception as exc:
    #     print(f"[PLANNER] Failed to generate plan: {exc}")
    #     return {}


# ── KPI computation --------------------------------------------------------

AGGREGATION_MAP = {
    "sum": sum,
    "avg": mean,
    "average": mean,
    "mean": mean,
    "max": max,
    "min": min,
}


def _format_value(value: float, fmt: Optional[str]) -> str:
    if fmt == "percent":
        return f"{value:.1%}"
    if fmt == "integer":
        return f"{value:,.0f}"
    if fmt == "currency":
        return f"${value:,.2f}"
    return f"{value:,.2f}"


def _numeric_series(rows: List[Dict[str, Any]], field: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        num = _as_number(row.get(field))
        if num is not None:
            values.append(num)
    return values


def compute_kpis_from_plan(kpi_specs: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if not rows:
        return results

    cache: Dict[str, List[float]] = {}
    for spec in kpi_specs:
        field = spec.get("field")
        stat = (spec.get("stat") or "sum").lower()
        label = spec.get("label") or (f"{stat.title()} of {field}" if field else stat.title())
        fmt = spec.get("format")
        detail = spec.get("help") or spec.get("detail") or ""

        if stat == "count":
            value = len(rows)
            formatted = f"{value:,}"
        elif stat == "count_distinct" and field:
            values = cache.setdefault(field, [row.get(field) for row in rows if row.get(field) is not None])
            value = len(set(values))
            formatted = f"{value:,}"
        else:
            if not field:
                continue
            series = cache.setdefault(field, _numeric_series(rows, field))
            if not series:
                continue
            if stat in AGGREGATION_MAP:
                try:
                    value = AGGREGATION_MAP[stat](series)
                except Exception:
                    continue
            elif stat == "count_distinct":
                value = len(set(series))
            else:
                continue
            formatted = _format_value(value, fmt)
        results.append(
            {
                "label": label,
                "value": formatted,
                "delta": None,
                "help": detail,
            }
        )
        if len(results) >= 4:
            break
    return results


# ── Chart building ---------------------------------------------------------

CHART_VALUE_COLUMN = "value"


def _build_chart_dataframe(
    rows: List[Dict[str, Any]],
    x_field: str,
    y_field: Optional[str],
    group_field: Optional[str],
    agg: str,
) -> Optional[List[Dict[str, Any]]]:
    if not rows:
        return None

    if pd is not None:
        try:
            df = pd.DataFrame(rows)
        except Exception:
            df = None
    else:
        df = None

    if df is not None:
        cols = [col for col in [x_field, y_field, group_field] if col]
        if not all(col in df.columns for col in cols):
            return None
        work_df = df[cols].copy()
        if y_field:
            work_df[y_field] = pd.to_numeric(work_df[y_field], errors="coerce")
        work_df = work_df.dropna()
        group_cols = [x_field]
        if group_field:
            group_cols.append(group_field)
        agg_lower = (agg or "sum").lower()
        if agg_lower in ("sum", "avg", "average", "mean", "max", "min") and y_field:
            func = {
                "sum": "sum",
                "avg": "mean",
                "average": "mean",
                "mean": "mean",
                "max": "max",
                "min": "min",
            }[agg_lower]
            agg_df = work_df.groupby(group_cols)[y_field].agg(func).reset_index()
            agg_df = agg_df.rename(columns={y_field: CHART_VALUE_COLUMN})
        elif agg_lower == "count":
            agg_df = work_df.groupby(group_cols).size().reset_index(name=CHART_VALUE_COLUMN)
        elif agg_lower == "count_distinct" and y_field:
            agg_df = work_df.groupby(group_cols)[y_field].nunique().reset_index(name=CHART_VALUE_COLUMN)
        else:
            return None
        return agg_df.to_dict(orient="records")

    # Manual fallback without pandas
    grouping: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
    for row in rows:
        x_val = row.get(x_field)
        if x_val is None:
            continue
        g_val = row.get(group_field) if group_field else None
        key = (x_val, g_val)
        if y_field:
            num = _as_number(row.get(y_field))
            if num is None:
                continue
            grouping[key].append(num)
        else:
            grouping[key].append(1.0)

    data: List[Dict[str, Any]] = []
    agg_lower = (agg or "sum").lower()
    for key, series in grouping.items():
        x_val, g_val = key
        if agg_lower in ("sum", "avg", "average", "mean", "max", "min"):
            if not series:
                continue
            if agg_lower == "sum":
                value = sum(series)
            elif agg_lower in ("avg", "average", "mean"):
                value = mean(series)
            elif agg_lower == "max":
                value = max(series)
            else:
                value = min(series)
        elif agg_lower == "count":
            value = len(series)
        elif agg_lower == "count_distinct":
            value = len(set(series))
        else:
            continue
        record = {x_field: x_val, CHART_VALUE_COLUMN: value}
        if group_field:
            record[group_field] = g_val
        data.append(record)
    return data or None


def build_charts_from_plan(chart_specs: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not chart_specs or not rows or not ALT_AVAILABLE:
        return []

    charts: List[Dict[str, Any]] = []
    for spec in chart_specs:
        ctype = (spec.get("type") or "bar").lower()
        x = spec.get("x")
        y = spec.get("y")
        group = spec.get("group")
        agg = spec.get("agg") or spec.get("stat") or "sum"
        title = spec.get("title") or f"{ctype.title()} chart"

        if ctype == "pie":
            category = spec.get("category") or x or group
            value_field = spec.get("value") or y
            if not category or not value_field:
                continue
            data = _build_chart_dataframe(rows, category, value_field, None, agg)
            if not data:
                continue
            chart = (
                alt.Chart(_to_chart_data(data))
                .mark_arc()
                .encode(
                    theta=alt.Theta(f"{CHART_VALUE_COLUMN}:Q", title=value_field.replace("_", " ").title()),
                    color=alt.Color(f"{category}:N", title=category.replace("_", " ").title()),
                    tooltip=[category, CHART_VALUE_COLUMN],
                )
                .properties(height=280)
            )
            charts.append({"title": title, "chart": chart})
            continue

        if not x:
            continue

        value_field = y or CHART_VALUE_COLUMN
        data = _build_chart_dataframe(rows, x, y, group, agg)
        if not data:
            continue

        x_encoding = alt.X(f"{x}", title=x.replace("_", " ").title())
        y_encoding = alt.Y(f"{CHART_VALUE_COLUMN}:Q", title=(y or CHART_VALUE_COLUMN).replace("_", " ").title())
        color_encoding = (
            alt.Color(f"{group}:N", title=group.replace("_", " ").title()) if group else None
        )

        if ctype == "line":
            chart = alt.Chart(_to_chart_data(data)).mark_line(point=True)
        elif ctype == "area":
            chart = alt.Chart(_to_chart_data(data)).mark_area()
        else:  # bar by default
            chart = alt.Chart(_to_chart_data(data)).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)

        encodings = {"x": x_encoding, "y": y_encoding, "tooltip": [x, CHART_VALUE_COLUMN]}
        if color_encoding:
            encodings["color"] = color_encoding
        chart = chart.encode(**encodings).properties(height=320)
        charts.append({"title": title, "chart": chart})

    return charts


# ── Fallback capsules -----------------------------------------------------

def build_fallback_capsules(rows: List[Dict[str, Any]], kpis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    capsules: List[Dict[str, Any]] = []
    if rows:
        capsules.append(
            {
                "badge": "Insight",
                "text": f"Rows analysed: {len(rows):,}",
                "detail": f"Processed {len(rows):,} records in current insight run.",
            }
        )
    for kpi in kpis:
        capsules.append(
            {
                "badge": "Trend",
                "text": kpi.get("label", "Metric"),
                "detail": f"{kpi.get('label', 'Metric')} = {kpi.get('value', '')}",
            }
        )
        if len(capsules) >= 5:
            break
    return capsules
