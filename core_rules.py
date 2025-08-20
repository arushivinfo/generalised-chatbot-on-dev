# core_rules.py
from typing import Dict, List, Optional

# ---------- Core rules text ----------

# ---------- Core rules text (role-less) ----------
def render_core_rules(collections: List[str] | None = None) -> str:
    """
    Role-less guidance. Optionally lists a few collection names to anchor the model.
    """
    examples = ""
    if collections:
        names = ", ".join(collections[:3])
        examples = f"\n- Available collections include: {names} (and more configured in Admin)."
    return (
        "**Guidance for answering**\n"
        "- Use the smallest set of collections needed; prefer structured filters (field:value, ranges).\n"
        "- Always set sort and limit explicitly when the intent implies ordering or top-K.\n"
        "- If a question mixes “current status/forecast” with history, query both a ‘latest’ style collection and at least one historical/context collection."
        + examples
    )


def render_match_context(user_match_context):
    # Treat as generic admin-provided “Extra Context”
    return (user_match_context or "").strip()


# ---------- Schema bullets ----------

def _format_fields_with_options(fields: List[dict], options_max: int) -> List[str]:
    """
    Render: name (type) [ops] (options: a, b, c…)
    IMPORTANT: avoid curly braces {…} because ChatPromptTemplate treats them as variables.
    """
    out: List[str] = []
    for f in fields:
        name = f.get("name", "")
        ftype = f.get("type", "string")
        ops = ", ".join(f.get("operations", []))
        base = f"{name} ({ftype}) [{ops}]"
        opts = f.get("options") or []
        if opts:
            preview = ", ".join(map(str, opts[:options_max]))
            suffix = "…" if len(opts) > options_max else ""
            base += f" (options: {preview}{suffix})"
        out.append(base)
    return out


def render_schema_section_all(
    all_fields: Dict[str, List[dict]],
    descriptions: Dict[str, str],
    options_max: int = 20,
) -> str:
    """
    Full schema section for **all collections** (admin may add many).
    - Keys are actual collection names (not roles).
    - Sorted alpha for stability.
    """
    lines: List[str] = []
    for coll in sorted(all_fields.keys()):
        desc = descriptions.get(coll, "")
        field_lines = _format_fields_with_options(all_fields[coll], options_max)
        lines.append(f"• {coll} [{desc}] → " + ("; ".join(field_lines) if field_lines else "(no fields)"))
    return "\n".join(lines)


# --- Backward-compat shim so older imports keep working ---
def render_schema_section(coll_map, searchable_fields, coll_desc, options_max: int = 20):
    """
    Legacy wrapper: same signature as before.
    - coll_map: {'matches': '<coll>', 'players': '<coll>', 'venues': '<coll>', 'upcoming_match': '<coll>'}
    - searchable_fields: { '<coll>': [field dicts...] }
    - coll_desc: {'matches': '...', 'players': '...', 'venues': '...', 'upcoming_match': '...' }
    """
    # We delegate to the new roles-only renderer:
    return render_schema_section_roles(
        core_map=coll_map,
        searchable_fields=searchable_fields,
        role_descriptions=coll_desc,
        options_max=options_max,
    )