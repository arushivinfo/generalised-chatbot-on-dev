# core_rules.py
from typing import Dict, List, Optional

# ---------- Core rules text ----------

def render_core_rules(core_map: Dict[str, str]) -> str:
    """
    core_map: {'matches': <coll>, 'players': <coll>, 'venues': <coll>, 'upcoming_match': <coll>}
    Safe if some roles are missing.
    """
    V = core_map.get("venues", "<venues_collection>")
    M = core_map.get("matches", "<matches_collection>")
    P = core_map.get("players", "<players_collection>")
    U = core_map.get("upcoming_match", "<upcoming_match_collection>")

    return f"""
**Core rule for venue based questions:**  
    Always include below collections when the user asks about a venue, along with the `{V}` collection:
    - `{M}` (to provide match history at that venue)
    - `{P}` (to provide player performance at that venue)

**Core rule for prediction questions**  
Prediction data alone often lacks full context (e.g. past performance, venue history, player roles). Therefore, whenever the user 
requests any prediction‐related information, you **must** include `{U}` **plus** at least one of the other 
three collections—`{M}`, `{P}`, or `{V}`. This ensures you combine both the 
latest predictive insights and the relevant historical or contextual data to craft a comprehensive answer.
""".strip()


def render_match_context(user_match_context: Optional[str]) -> str:
    """Optional user-provided 'extra add-up'. Return empty string if not provided."""
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


def render_schema_section_roles(
    core_map: Dict[str, str],
    searchable_fields: Dict[str, List[dict]],
    role_descriptions: Dict[str, str],
    options_max: int = 20,
) -> str:
    """
    Small schema section for the four core roles only.
    - Bullets show the **actual collection name**.
    - 'upcoming_match' gets the single-row note.
    """
    lines: List[str] = []
    for role in ["matches", "players", "venues", "upcoming_match"]:
        coll = core_map.get(role)
        if not coll:
            continue
        desc = role_descriptions.get(role, "")
        field_lines = _format_fields_with_options(searchable_fields.get(coll, []), options_max)
        if role == "upcoming_match":
            lines.append(
                f"• {coll} [{desc}] →\n"
                f"  - **single-row** collection for the configured fixture.\n"
                f"  - You may leave \"filters\", \"sort\", and \"limit\" empty to retrieve the full object."
            )
        else:
            lines.append(f"• {coll} [{desc}] → " + ("; ".join(field_lines) if field_lines else "(no fields)"))
    return "\n".join(lines)


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