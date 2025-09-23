"""Utility helpers to surface KPIs and charts for insight results."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional

try:  # Altair drives the visuals; degrade gracefully if unavailable.
    import altair as alt
    ALT_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime when altair missing.
    alt = None  # type: ignore
    ALT_AVAILABLE = False

try:  # Optional pandas support improves type handling for charts.
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        stripped = value.replace(",", "").strip()
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _is_date_like(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return False
        if text.endswith("Z"):
            text = text[:-1]
        try:
            datetime.fromisoformat(text)
            return True
        except ValueError:
            return False
    return False


def _collect_field_examples(rows: List[Dict[str, Any]], limit: int = 50) -> Dict[str, List[Any]]:
    samples: Dict[str, List[Any]] = defaultdict(list)
    for row in rows[:limit]:
        for key, value in row.items():
            samples[key].append(value)
    return samples


def _detect_numeric_fields(samples: Dict[str, List[Any]], min_hits: int = 2) -> List[str]:
    numeric_fields = []
    for field, values in samples.items():
        hits = [v for v in values if _as_number(v) is not None]
        if len(hits) >= min_hits:
            numeric_fields.append(field)
    return numeric_fields


def _detect_date_field(samples: Dict[str, List[Any]]) -> Optional[str]:
    for field, values in samples.items():
        if any(_is_date_like(v) for v in values):
            return field
    return None


def _detect_category_field(samples: Dict[str, List[Any]], exclude: Iterable[str] = ()) -> Optional[str]:
    exclude_set = {e.lower() for e in exclude}
    for field, values in samples.items():
        if field.lower() in exclude_set:
            continue
        str_values = [str(v) for v in values if isinstance(v, (str, int, float))]
        unique = {s for s in str_values if s}
        if 1 < len(unique) <= 12:
            return field
    return None


def build_kpis(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    samples = _collect_field_examples(rows)
    numeric_fields = _detect_numeric_fields(samples)

    kpis: List[Dict[str, Any]] = [
        {
            "label": "Rows Returned",
            "value": f"{len(rows):,}",
            "delta": None,
            "help": "Documents surfaced by the insight query",
        }
    ]

    if numeric_fields:
        field = numeric_fields[0]
        numbers = [n for v in samples[field] if (n := _as_number(v)) is not None]
        if numbers:
            total = sum(numbers)
            avg_value = mean(numbers)
            max_value = max(numbers)
            kpis.append(
                {
                    "label": f"Sum of {field}",
                    "value": f"{total:,.2f}",
                    "delta": None,
                    "help": "Aggregate across returned rows",
                }
            )
            kpis.append(
                {
                    "label": f"Avg {field}",
                    "value": f"{avg_value:,.2f}",
                    "delta": None,
                    "help": "Mean value within rows",
                }
            )
            kpis.append(
                {
                    "label": f"Max {field}",
                    "value": f"{max_value:,.2f}",
                    "delta": None,
                    "help": "Peak value in current dataset",
                }
            )

    return kpis[:4]


def _aggregated_numeric(rows: List[Dict[str, Any]], field_key: str, by_key: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, float] = defaultdict(float)
    for row in rows:
        group_raw = row.get(by_key)
        metric = _as_number(row.get(field_key))
        if group_raw is None or metric is None:
            continue
        group = str(group_raw)
        buckets[group] += metric
    return [
        {by_key: bucket, field_key: value}
        for bucket, value in sorted(buckets.items(), key=lambda item: item[1], reverse=True)
    ]


def _aggregated_counts(rows: List[Dict[str, Any]], by_key: str) -> List[Dict[str, Any]]:
    counter = Counter(str(row.get(by_key)) for row in rows if row.get(by_key) is not None)
    return [
        {by_key: bucket, "count": count}
        for bucket, count in counter.most_common(12)
    ]


def _to_chart_data(values: List[Dict[str, Any]]):
    if pd is not None:
        try:
            return pd.DataFrame(values)
        except Exception:
            pass
    return alt.Data(values=values)


def build_charts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ALT_AVAILABLE or not rows:
        return []

    samples = _collect_field_examples(rows)
    numeric_fields = _detect_numeric_fields(samples)

    charts: List[Dict[str, Any]] = []

    if numeric_fields:
        value_field = numeric_fields[0]
        date_field = _detect_date_field(samples)
        if date_field:
            data = _aggregated_numeric(rows, value_field, date_field)
            if data:
                charts.append(
                    {
                        "title": f"{value_field} over {date_field}",
                        "chart": alt.Chart(_to_chart_data(data))
                        .mark_line(point=True)
                        .encode(
                            x=alt.X(f"{date_field}:T", title=date_field.replace("_", " ").title()),
                            y=alt.Y(f"{value_field}:Q", title=value_field.replace("_", " ").title()),
                            tooltip=[
                                alt.Tooltip(f"{date_field}:T", title=date_field.replace("_", " ").title()),
                                alt.Tooltip(f"{value_field}:Q", title=value_field.replace("_", " ").title()),
                            ],
                        )
                        .properties(height=280),
                    }
                )
        category_field = _detect_category_field(samples, exclude=[value_field])
        if category_field:
            data = _aggregated_numeric(rows, value_field, category_field)[:12]
            if data:
                charts.append(
                    {
                        "title": f"{value_field} by {category_field}",
                        "chart": alt.Chart(_to_chart_data(data))
                        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                        .encode(
                            x=alt.X(f"{value_field}:Q", title=value_field.replace("_", " ").title()),
                            y=alt.Y(
                                f"{category_field}:N",
                                sort="-x",
                                title=category_field.replace("_", " ").title(),
                            ),
                            tooltip=[
                                alt.Tooltip(f"{category_field}:N", title=category_field.replace("_", " ").title()),
                                alt.Tooltip(f"{value_field}:Q", title=value_field.replace("_", " ").title()),
                            ],
                        )
                        .properties(height=320),
                    }
                )
    else:
        category_field = _detect_category_field(samples)
        if category_field:
            data = _aggregated_counts(rows, category_field)
            charts.append(
                {
                    "title": f"Row counts by {category_field}",
                    "chart": alt.Chart(_to_chart_data(data))
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X("count:Q", title="Count"),
                        y=alt.Y(
                            f"{category_field}:N",
                            sort="-x",
                            title=category_field.replace("_", " ").title(),
                        ),
                        tooltip=[
                            alt.Tooltip(f"{category_field}:N", title=category_field.replace("_", " ").title()),
                            alt.Tooltip("count:Q", title="Count"),
                        ],
                    )
                    .properties(height=320),
                }
            )

    return charts


__all__ = ["build_kpis", "build_charts", "ALT_AVAILABLE"]
