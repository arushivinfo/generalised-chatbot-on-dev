# search_agent.py
import os, json
from pathlib import Path
from datetime import datetime as _dt
from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime, date

from dotenv import load_dotenv
from pymongo import MongoClient
from bson import json_util

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from cache_memory import get_last_memories

from typing import List, Optional
from langchain_core.callbacks import BaseCallbackHandler

from difflib import get_close_matches
# search_agent_new.py  (only showing relevant edits)
from core_rules import render_core_rules, render_match_context, render_schema_section_all
from schema_registry import load_registry, get_all_fields, get_core_coll_map, get_descriptions

def _load_schema_live():
    reg = load_registry()
    all_fields   = get_all_fields(reg)
    core_map     = get_core_coll_map(reg)
    descriptions = get_descriptions(reg)
    options_max  = reg.get("options_max", 20)
    return all_fields, core_map, descriptions, options_max


class PrintIntermediateStepsHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("\n[CHAIN START]")
        print(f"Inputs: {inputs}")

    def on_chain_end(self, outputs, **kwargs):
        print("\n[CHAIN END]")
        print(f"Outputs: {outputs}")

    def on_tool_start(self, tool, input, **kwargs):
        print(f"\n[TOOL START] Tool: {tool}, Input: {input}")

    def on_tool_end(self, output, **kwargs):
        print(f"[TOOL END] Output: {output}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n[LLM START]")
        for prompt in prompts:
            print(f"Prompt: {prompt}")

    def on_llm_end(self, response, **kwargs):
        print("\n[LLM END]")
        print(f"Response: {response}")

    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)


class QueryResult(TypedDict, total=False):
    ok:     bool
    docs:   List[Dict[str, Any]]
    error:  str
    filter: Dict[str, Any]

def _apply_sort(cursor_or_pipeline, sort_clause):
    """Attach .sort() / $sort only when the list is non-empty."""
    if not sort_clause:
        return cursor_or_pipeline
    if isinstance(cursor_or_pipeline, list):                 # aggregation pipe
        cursor_or_pipeline.append({"$sort": dict(sort_clause)})
        return cursor_or_pipeline
    return cursor_or_pipeline.sort(sort_clause)

# ────────────────────────────────
# 0. Environment & DB connection
# ────────────────────────────────

MATCH_CONTEXT = """
### MATCH CONTEXT – KEEP AS SEPARATE SYSTEM MESSAGE ###
This assistant covers **one fixture only**:

• Fixture  : Australia(Home Team) vs South Africa(Away Team)
• League   : South Africa tour of Australia
• Ground   : Marrara Cricket Ground (MCG 2), Darwin, Australia
• Team UIDs: 5↔ 19 (either side can be home/away)

Full squad (25):
**Australia (AUS):**
Mitchell Owen, Adam Zampa, Travis Head, Ben Dwarshuis, Matthew Short, Josh Inglis, Matthew Kuhnemann, 
Sean Abbott, Glenn Maxwell, Mitchell Marsh, Josh Hazlewood, Cameron Green, Tim David, Aaron Hardie,Nathan Ellis

**South Africa (SA):**
Dewald Brevis, Kwena Maphaka, Lhuan dre Pretorius, Kagiso Rabada, Nqabayomzi Peter, Aiden Markram, Lungisani Ngidi,
Rassie van der Dussen, George Linde, Senuran Muthusamy, Prenelan Subrayen, Nandre Burger, Corbin Bosch, Ryan Rickelton, Tristan Stubbs


🛈 If the user says “this match / venue / league / team / these players”, resolve the reference to **this fixture** unless they clearly mention something else.
"""

LOG_PATH = Path("last_search_run.json")
load_dotenv()

MONGO_URI = "mongodb://ec2-35-154-176-120.ap-south-1.compute.amazonaws.com:27017/"
DB_NAME    = "sports_feed_stg"
client     = MongoClient(MONGO_URI)
db         = client[DB_NAME]

# Fetch all unique league names from the DB for canonicalization
LEAGUE_NAMES = db.matches_filtered_90696.distinct("league_name")

# ────────────────────────────────
# 1. Schema metadata
# ────────────────────────────────


# SEARCHABLE_FIELDS: Dict[str, List[Dict[str, Any]]] = {
#     "matches_filtered_90696": [
#         {"name": "league_name",           "type": "string", "operations": ["regex"]},
#         {"name": "league_name_abbr",      "type": "string", "operations": ["regex"]},
#         # {"name": "match_format",          "type": "string", "operations": ["regex"]},       
#         # {"name": "full_match_title",      "type": "string", "operations": ["regex"]},

#         # {"name": "title",                 "type": "string", "operations": ["regex"]},
#         {"name": "venue",                 "type": "string", "operations": ["regex", "sort"]},
#         {"name": "city",                  "type": "string", "operations": ["regex"]},
#         {"name": "scheduled_date",        "type": "date",   "operations": ["range", "sort"]},
#         {"name": "bat_first_team_score",  "type": "int",    "operations": ["range", "sort"]},
#         {"name": "bat_second_team_score", "type": "string", "operations": ["regex"]},
#         {"name": "players",               "type": "array",  "operations": ["keyword"]},

#         {"name": "away_display_team_name",    "type": "string", "operations": ["regex"]},
#         {"name": "home_display_team_name",    "type": "string", "operations": ["regex"]},
#         # {"name": "bat_first_team_name",       "type": "string", "operations": ["regex"]},
#         # {"name": "bat_second_team_name",      "type": "string", "operations": ["regex"]},
#         {"name": "winning_team_name",         "type": "string", "operations": ["regex"]},
#     ],

#     # "matches_filtered_90696": [
#     #     {"name": "league_name",          "type": "string", "operations": ["regex"]},
#     #     {"name": "title",                "type": "string", "operations": ["regex"]},
#     #     {"name": "venue",                "type": "string", "operations": ["regex", "sort"]},
#     #     {"name": "city",                 "type": "string", "operations": ["regex"]},
#     #     {"name": "scheduled_date",       "type": "date",   "operations": ["range", "sort"]},
#     #     {"name": "bat_first_team_score", "type": "int",    "operations": ["range", "sort"]},
#     #     {"name": "players",              "type": "array",  "operations": ["keyword"]},
#     # ],
#     "players_filtered_90696": [
#         {"name": "player_name",          "type": "string", "operations": ["regex"]},
#         {"name": "team_name",            "type": "string", "operations": ["regex"]},
#         {"name": "position",             "type": "string", "operations": ["regex", "keyword"], "options": ["Batsman", "Bowler", "All-rounder", "Wicketkeeper"]},
#         # {"name": "league_name",          "type": "string", "operations": ["regex"]},
#         # {"name": "match_title",          "type": "string", "operations": ["regex"]},
#         {"name": "venue",                "type": "string", "operations": ["regex", "sort"]},
#         {"name": "fantasy_points",       "type": "float",  "operations": ["range", "sort"]},
#         {"name": "runs_scored",          "type": "int",    "operations": ["range", "sort"]},
#         {"name": "wickets",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "overs_bowled",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "Strike Rate",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "batting_dots",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "batting_fours",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "batting_sixes",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "bowling_wides",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "bowling_noballs",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "bowling_economy",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "bowling_order",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "bowling_dots",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "catch",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "run_out",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "run_out_throw",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "run_out_catch",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "power_wickets",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "power_overs",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "death_wickets",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "death_overs",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "selected_percentage",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "captain_selection_percentage",         "type": "float",  "operations": ["range", "sort"]},
#         {"name": "vice_captain_selection_percentage",         "type": "float",  "operations": ["range", "sort"]},

#         {"name": "scheduled_date",       "type": "date",   "operations": ["range", "sort"]},
#     ],
#     "venues_filtered_90696": [
#         {"name": "venue",          "type": "string", "operations": ["regex","sort"]},
#         {"name": "city",           "type": "string", "operations": ["regex"]},
#         {"name": "country",        "type": "string", "operations": ["regex"]},
#         {"name": "capacity",       "type": "int",    "operations": ["range", "sort"]},
#         {"name": "run_per_over",   "type": "float",  "operations": ["range", "sort"]},
#         {"name": "run_per_wicket", "type": "float",  "operations": ["range", "sort"]},
#     ]
# }

# COLL_MAP = {
#     "matches": "matches_filtered_90696",
#     "players": "players_filtered_90696",
#     "venues":  "venues_filtered_90696",
#     "upcoming_match": "upcoming_match_90696_summary",
# }

# COLL_DESCRIPTIONS = {
#     "matches":        "contains data of historical matches",
#     "players":        "contains squad & historical player data",
#     "venues":         "contains venue and ground stats",
#     "upcoming_match": "contains upcoming match and prediction data",
# }

_reg = load_registry()
ALL_FIELDS      = get_all_fields(_reg)         # {coll_name: fields[]}
CORE_COLL_MAP   = get_core_coll_map(_reg)      # {'matches': 'matches_filtered_90696', ...}
DESCRIPTIONS    = get_descriptions(_reg)
OPTIONS_MAX     = _reg.get("options_max", 20)


SCHEMA_SECTION  = render_schema_section_all(ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX)

CORE_RULES_TEXT = render_core_rules(CORE_COLL_MAP)
USER_MATCH_CONTEXT = render_match_context(MATCH_CONTEXT)  # Make MATCH_CONTEXT user-editable; can be empty

# 🔧 Back-compat so older code keeps working:
COLL_MAP = CORE_COLL_MAP           # {'matches': 'matches_filtered_…', ...}
SEARCHABLE_FIELDS = ALL_FIELDS  

# ────────────────────────────────
# 2. Pydantic models (no collection)
# ────────────────────────────────
class EntityFilter(BaseModel):
    field: str
    operation: str
    value: Any

class EntityQuery(BaseModel):
    filters: List[EntityFilter]
    sort:   Dict[str, str] = {}
    limit:  int = 10


class CollectionQuery(EntityQuery):
    """EntityQuery with an explicit collection name."""
    collection: str


class MultiEntityQuery(BaseModel):
    """Container for multiple collection-specific queries."""
    queries: List[CollectionQuery]


# ────────────────────────────────
# 4.5. Collection picker
# ────────────────────────────────

def pick_collection(spec: dict) -> str | None:
    """
    Return one of "matches", "players", "venues" if *all*
    spec['filters'] fields and the sort field live in that schema.
    """
    # 1) gather all filter fields
    fields = [filt["field"] for filt in spec.get("filters", [])]
    # 2) include sort field too
    sf = next(iter(spec.get("sort", {})), None)
    if sf:
        fields.append(sf)

    # 3) test each collection
    for coll_key, schema_name in COLL_MAP.items():
        schema_fields = {f["name"] for f in SEARCHABLE_FIELDS[schema_name]}
        if set(fields).issubset(schema_fields):
            return coll_key
    return None


# ────────────────────────────────
# 3. Shared query executor
# ────────────────────────────────


# 3.a  Post-process each filter coming from the LLM
#      • unwrap ["All-rounder"] → "All-rounder"
#      • if op = keyword but the field is scalar string (and schema
#        doesn’t list keyword) → flip to regex
# def _normalise_filter(filt: dict) -> dict:
def _normalise_filter(filt: dict, collection: str = "players_filtered", invalid_fields: Optional[List[tuple]] = None) -> dict | None:
    out = filt.copy()

    # 1) unwrap single-element lists so $regex always gets a string
    # unwrap ["value"] -> "value"
    if isinstance(out.get("value"), list) and len(out["value"]) == 1:
        out["value"] = out["value"][0]

    # position is a scalar string in the DB → always use regex
    if out["field"] == "position" and out["operation"] == "keyword":
        out["operation"] = "regex"

    # Validate against SEARCHABLE_FIELDS options
    # for coll_fields in SEARCHABLE_FIELDS.values():
    #     for field_meta in coll_fields:
    #         if field_meta["name"] == out["field"] and "options" in field_meta:
    #             if out["value"] not in field_meta["options"]:
    #                 if invalid_fields is not None:
    #                     invalid_fields.append((out["field"], out["value"]))
    #                 else:
    #                     print(f"⚠️ Invalid value for {out['field']}: {out['value']} — ignoring this filter.")
    #                 return None


    # 3) fuzzy match for allowed options
    for field_meta in SEARCHABLE_FIELDS.get(collection, []):
        if field_meta["name"] == out["field"] and "options" in field_meta:
            allowed = field_meta["options"]
            raw_val = out["value"]

            # try case-insensitive exact match
            if raw_val in allowed:
                return out
            
            # try fuzzy match
            match = get_close_matches(raw_val, allowed, n=1, cutoff=0.6)
            if match:
                out["value"] = match[0]  # auto-correct
                return out

            # fallback: not matched
            if invalid_fields is not None:
                invalid_fields.append((out["field"], raw_val))
            else:
                print(f"⚠️ Invalid value for {out['field']}: {raw_val} — ignoring this filter.")
            return None

    return out

def _safe(obj):
    """Convert non-JSON types to strings for dumps()."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return json_util.default(obj)


def _run_query(collection_key: str, spec: Dict[str, Any]) -> QueryResult:
    try:
        parsed      = EntityQuery(**spec)
        coll_name   = COLL_MAP[collection_key]
        coll        = db[coll_name]
        mongo_filter = {}
        regular_filters = []
        team_name_map = {}  # group team name values

        # # ---------- filters ----------
        # for f in parsed.filters:
        #     fld_meta = next((m for m in SEARCHABLE_FIELDS[coll_name]
        #                      if m["name"] == f.field), None)
        #     if not fld_meta or f.operation not in fld_meta["operations"]:
        #         return {"ok": False,
        #                 "error": f"Invalid field/operation: {f.field},{f.operation}",
        #                 "filter": mongo_filter}

        #     if f.operation == "regex":
        #         mongo_filter[f.field] = {"$regex": f.value, "$options": "i"}
        #     elif f.operation == "keyword":
        #         mongo_filter[f.field] = {"$elemMatch": {"$regex": f.value,
        #                                                 "$options": "i"}}
        #     elif f.operation == "range":
        #         rng = ( {op: datetime.fromisoformat(v) if isinstance(v, str) else v
        #                  for op, v in f.value.items()}
        #                 if fld_meta["type"] == "date" else f.value )
        #         mongo_filter[f.field] = rng

        # ---------- filters ----------
        for f in parsed.filters:
            fld_meta = next((m for m in SEARCHABLE_FIELDS[coll_name]
                            if m["name"] == f.field), None)
            if not fld_meta or f.operation not in fld_meta["operations"]:
                return {"ok": False,
                        "error": f"Invalid field/operation: {f.field},{f.operation}",
                        "filter": mongo_filter}

            # GROUP team filters
            if f.field in ["home_display_team_name", "away_display_team_name"]:
                val = f.value.lower()
                team_name_map.setdefault(val, set()).add(f.field)
                continue

            # Regular filters
            if f.operation == "regex":
                regular_filters.append({f.field: {"$regex": f.value, "$options": "i"}})
            elif f.operation == "keyword":
                regular_filters.append({f.field: {"$elemMatch": {"$regex": f.value, "$options": "i"}}})
            elif f.operation == "range":
                rng = (
                    {op: datetime.fromisoformat(v) if isinstance(v, str) else v
                    for op, v in f.value.items()}
                    if fld_meta["type"] == "date" else f.value
                )
                regular_filters.append({f.field: rng})

        # 🔁 Now build grouped $or filters for each unique team name
        for team_val, fields in team_name_map.items():
            or_group = []
            if "home_display_team_name" in fields:
                or_group.append({ "home_display_team_name": { "$regex": team_val, "$options": "i" } })
            if "away_display_team_name" in fields:
                or_group.append({ "away_display_team_name": { "$regex": team_val, "$options": "i" } })
            if or_group:
                regular_filters.append({ "$or": or_group })

        # Final mongo_filter
        if len(regular_filters) == 1:
            mongo_filter = regular_filters[0]
        elif len(regular_filters) > 1:
            mongo_filter = { "$and": regular_filters }


        # ---------- sort ----------
        sort_clause = [(fld, 1 if d.lower() == "asc" else -1)
                       for fld, d in parsed.sort.items()]
        for fld, _ in sort_clause:
            if fld not in {m["name"] for m in SEARCHABLE_FIELDS[coll_name]
                           if "sort" in m["operations"]}:
                return {"ok": False,
                        "error": f"Cannot sort on field: {fld}",
                        "filter": mongo_filter}

        # cursor = _apply_sort(
        #     coll.find(mongo_filter, {"summary": 1, "_id": 0}),
        #     sort_clause
        # ).limit(parsed.limit)

        # For normal collections we still project just "summary".
        # For the one‑row upcoming_match_91776 we return the full doc.

        # projection = (
        #     {"_id": 0}                               # ← all keys
        #     if coll_name == "upcoming_match_90696_summary"  # ← upcoming_match_91776
        #     else {"summary": 1, "_id": 0}            # legacy behaviour
        # )

        projection = {"_id": 0} if collection_key == "upcoming_match" else {"summary": 1, "_id": 0}

        cursor = _apply_sort(
            coll.find(mongo_filter, projection),
            sort_clause
        ).limit(parsed.limit)

        docs = list(cursor)
        if not docs:
            return {"ok": False, "error": "No results found",
                    "filter": mongo_filter}

        safe_json = json.loads(json.dumps(docs, default=_safe))
        return {"ok": True, "docs": safe_json, "filter": mongo_filter}

    except Exception as exc:
        return {"ok": False, "error": f"Query failed: {exc}",
                "filter": mongo_filter}
    

# ────────────────────────────────
# 4. Tools (collection-fixed)
# ────────────────────────────────
@tool("search_matches")
def search_matches(**q) -> List[Dict[str, Any]]:
    """Search the *matches* collection."""
    return _run_query("matches", q)

@tool("search_players")
def search_players(**q) -> List[Dict[str, Any]]:
    """Search the *players* collection."""
    return _run_query("players", q)

@tool("search_venues")
def search_venues(**q) -> List[Dict[str, Any]]:
    """Search the *venues* collection."""
    return _run_query("venues", q)

@tool("search_current_match")
def search_current_match() -> List[Dict[str, Any]]:
    """Search the current match."""
    return _run_query("matches", q)

TOOLS = [search_matches, search_players, search_venues]

# ────────────────────────────────
# 5. Prompt
# ────────────────────────────────


# List of strings with "name (type) [operations]" format for matches
matches_fields = [
    f"{f['name']} ({f['type']}) [{', '.join(f['operations'])}]"
    for f in SEARCHABLE_FIELDS[COLL_MAP["matches"]]
]

players_fields = [
    f"{f['name']} ({f['type']}) [{', '.join(f['operations'])}]"
    for f in SEARCHABLE_FIELDS[COLL_MAP["players"]]
]

venues_fields = [
    f"{f['name']} ({f['type']}) [{', '.join(f['operations'])}]"
    for f in SEARCHABLE_FIELDS[COLL_MAP["venues"]]
]

def format_fields_with_options(fields: List[Dict[str, Any]]) -> List[str]:
    formatted = []
    for f in fields:
        base = f"{f['name']} ({f['type']}) [{', '.join(f['operations'])}]"
        if "options" in f:
            opts = ", ".join(f["options"])
            base += f" (options: {opts})"
        formatted.append(base)
    return formatted

matches_fields = format_fields_with_options(SEARCHABLE_FIELDS[COLL_MAP["matches"]])
players_fields = format_fields_with_options(SEARCHABLE_FIELDS[COLL_MAP["players"]])
venues_fields = format_fields_with_options(SEARCHABLE_FIELDS[COLL_MAP["venues"]])

print(matches_fields)

MATCH_CONTEXT = """
### MATCH CONTEXT – KEEP AS SEPARATE SYSTEM MESSAGE ###
This assistant covers **one fixture only**:

• Fixture  : Australia(Home Team) vs South Africa(Away Team)
• League   : South Africa tour of Australia
• Ground   : Marrara Cricket Ground (MCG 2), Darwin, Australia
• Team UIDs: 5↔ 19 (either side can be home/away)

Full squad (25):
**Australia (AUS):**
Mitchell Owen, Adam Zampa, Travis Head, Ben Dwarshuis, Matthew Short, Josh Inglis, Matthew Kuhnemann, 
Sean Abbott, Glenn Maxwell, Mitchell Marsh, Josh Hazlewood, Cameron Green, Tim David, Aaron Hardie,Nathan Ellis

**South Africa (SA):**
Dewald Brevis, Kwena Maphaka, Lhuan dre Pretorius, Kagiso Rabada, Nqabayomzi Peter, Aiden Markram, Lungisani Ngidi,
Rassie van der Dussen, George Linde, Senuran Muthusamy, Prenelan Subrayen, Nandre Burger, Corbin Bosch, Ryan Rickelton, Tristan Stubbs


🛈 If the user says “this match / venue / league / team / these players”, resolve the reference to **this fixture** unless they clearly mention something else.
"""

# Initialize MemoryAgent for language detection only
# mem = MemoryAgent(k=5)

U = COLL_MAP.get("upcoming_match", "upcoming_match")
M = COLL_MAP.get("matches", "matches")
P = COLL_MAP.get("players", "players")
V = COLL_MAP.get("venues",  "venues")

SYSTEM_PROMPT = """
You are an expert MongoDB query planner for three collections:

{schema_section}

Only use the operations listed for each field above.
If a field has a list of allowed options (shown after →), you must use one of those exact values for that field. Do not invent or assume values not in the list.


**Current Match Context Priority**:
   If the user's question refers to:
   - “this match”
   - “current match”
   - “the match”
   - “these players”
   - “this venue”
   - “our fixture”
   ...or any similar implicit references, assume they are referring to the fixture below.
    Use the current match context to answer the question.

#Also use the memory context if available, if any of the last 3 answers say "no data available" or similar, ignore that answer for reasoning.
"If the user query contains pronouns (e.g., 'he', 'him', 'his'), 
always resolve them to the correct entity using the most recent relevant memory context. 
Never use a pronoun as a value in any query field."



Your job:

1. Read the user’s natural-language request.

• If it mentions **“this match”, “current match”, “our fixture”, “the match in context”** (or similar) and the intent is to get details of that single fixture, use the upcoming match collection.
  If additional collections are also needed, include this object as one element of a top-level "queries" list.

2. For other requests, output either:
   {{{{
     "collection": "collection1" | "collection2" | "colection3",
     "filters": [{{{{"field": "...", "operation": "...", "value": ...}}}}, ...],
     "sort":    {{{{"field_name": "asc|desc"}}}},
     "limit":   <int - default 10>
   }}}}
   when only one collection is needed, **or**
   {{{{
     "queries": [
       {{{{
         "collection": "...",
         "filters": [{{{{"field": "...", "operation": "...", "value": ...}}}}, ...],
         "sort": {{{{"field_name": "asc|desc"}}}},
         "limit": <int - default 10>
       }}}},
       {{{{
         "collection": "...",
         "filters": [{{{{"field": "...", "operation": "...", "value": ...}}}}, ...],
         "sort": {{{{"field_name": "asc|desc"}}}},
         "limit": <int - default 10>
       }}}}
     ]
   }}}}
   when the request requires multiple collections. In that case return one query object per collection.
   – always include the "collection" key in each query object.
3. Decide which tool to call (search_matches / search_players / search_venues / search_upcoming_match)
   based on which collection those fields belong to.
4. After the tool returns, write a concise answer for the user.
5. For queries asking for the "highest", "most", "top", or "best", use a sort on the relevant field
   (descending) and set limit to the required number.
6. If the user explicitly specifies a date or date range, include it in the query filters.
   Otherwise, do not add any date filters.

   
For Filtering, REMEMBER:
When extracting player or venue names, correct spelling mistakes and use the official name as per your knowledge.

Memory context:
<CONVERSATION_HISTORY>

#Also use the memory context if available, if any of the last few answers say "no data available" or similar, ignore that answer for reasoning.
    "If the user query contains pronouns (e.g., 'he', 'him', 'his'), 
    always resolve them to the correct entity using the most recent relevant memory context. 
    Never use a pronoun as a value in any query field."
    "When the user query contains pronouns like 'he', 'him', 'his''इसको','इसके'(any languadge), always resolve them to the correct player name using the most recent relevant memory. For example, if the last answer was about 'X', and the user now asks 'his last 5 matches', use 'X' as the value for 'player_name'.\n"
    "Never use a pronoun as a value in any query field. For example, if the last answer was about 'Virat Kohli', and the user now asks 'How many runs did he make?', use 'Virat Kohli' as the value for 'player_name'.\n"
    "The most recent memory (highest weight) is listed first.\n"


Allowed operations
• regex   – case-insensitive substring match (strings)
• keyword – substring match inside *array* fields only (e.g., players); **do not use on scalar strings like position, venue, city, etc.**
• range   – {{{{"$gte": ..}}}}, {{{{"$lte": ..}}}} on numbers or dates (YYYY-MM-DD)
• sort    – asc / desc on sortable numeric/date fields


Output Instructions:
Return exactly one JSON object and nothing else.
• For single-collection requests, return the query object directly.
• For multi-collection requests, return {{{{"queries": [{{{{...}}}}, ...]}}}} with one object per collection.
Do not add any text outside the JSON.

IMPORTANT:

• For questions asking for a player's total, cumulative, or overall statistic for a season or tournament (e.g., “How many wickets did X take in Y 2025?”,
  “What is the average score at Wankhede Stadium in IPL 2024?”, “What is the win rate for Mumbai Indians in IPL 2023?”), you must fetch all relevant 
  rows matching the filters (e.g., all matches for that player and season), and do not use limit: 1.
• For such questions, set limit to a high value (such as 20-30) to ensure all relevant data is returned for aggregation or pattern observation.
• Only use limit: 1 for queries that explicitly ask for the single top value (e.g., “Who scored the most runs in IPL 2025?”).
• For questions about patterns, trends, or averages, always fetch enough data to allow for meaningful observation (e.g., all matches in a season or all matches for a player/team).
• If the user asks for a “summary”, “trend”, “pattern”, “average”, “total”, or “how many”, do not sort or limit unless specifically requested.
• For any prediction‐related questions (captain/vice‐captain trends, venue tags, player split recommendations):
  - Do not add filters for prediction_data fields.
  - Fetch and use the full "prediction_data" object from upcoming_match.
  - Base your response suggestions directly on its values.


""".format(schema_section=SCHEMA_SECTION)

PROMPT_without_memory = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("system", MATCH_CONTEXT), ("placeholder", "{messages}")]
)

# ────────────────────────────────
# 6. LLM & agent
# ────────────────────────────────
llm = ChatOpenAI(model="gpt-4.1-mini")

def get_memory_prompt():
    memories = get_last_memories(1)
    mem_text = "\n".join(
        [f"Previous Q: {m['query']}\nPrevious A: {m['answer']}" for m in memories if "no data" not in m['answer'].lower()]
    )
    return (
        "### RECENT MEMORY CONTEXT\n"
        "If any of the last 3 answers below say 'no data available' or similar, ignore that answer for reasoning.\n"
        f"{mem_text}\n"
        "When the user query contains pronouns like 'he', 'him', 'his''इसको','इसके'(any language), always resolve them to the correct player name using the most recent relevant memory. For example, if the last answer was about 'X', and the user now asks 'his last 5 matches', use 'X' as the value for 'player_name'.\n"
        "Never use a pronoun as a value in any query field. For example, if the last answer was about 'Virat Kohli', and the user now asks 'How many runs did he make?', use 'Virat Kohli' as the value for 'player_name'.\n"
        "The most recent memory (highest weight) is listed first.\n"
    )
MEMORY_PROMPT = get_memory_prompt()
print("Memory context for prompt(Search agent):", MEMORY_PROMPT) 

# If you currently build PROMPT via ChatPromptTemplate, keep that; just swap in variables:
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", CORE_RULES_TEXT),        # core rules injected here (generalised)
    ("system", USER_MATCH_CONTEXT),     # optional, can be "" (user add-on)
    ("system", MEMORY_PROMPT),          # you already compute this:contentReference[oaicite:4]{index=4}
    ("placeholder", "{messages}")
])


# PROMPT = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_PROMPT),
#     ("system", MATCH_CONTEXT),
#     ("system", MEMORY_PROMPT),
#     ("placeholder", "{messages}")
# ])

# agent = create_react_agent(
#     model   = llm,
#     tools   = TOOLS,
#     prompt  = PROMPT,   
# )

# ────────────────────────────────
# 7. User-facing wrapper
# ────────────────────────────────
def run_search_agent(
    query: str,
    history: None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    debug: bool = False,
) -> tuple[dict, str, Dict[str, Any]]:        # spec, answer, dbg

    """
    1) Ask the LLM for a JSON spec
    2) Parse the JSON
    3) Figure out which tool to call
    4) Call it and pretty-print the results
    """

    global ALL_FIELDS, CORE_COLL_MAP, DESCRIPTIONS, OPTIONS_MAX
    global COLL_MAP, SEARCHABLE_FIELDS, CORE_RULES_TEXT, SCHEMA_SECTION

    ALL_FIELDS, CORE_COLL_MAP, DESCRIPTIONS, OPTIONS_MAX = _load_schema_live()
    COLL_MAP          = CORE_COLL_MAP            # back-compat alias (roles → names)
    SEARCHABLE_FIELDS = ALL_FIELDS               # back-compat alias ({name: fields})

    # ---------- 1) build dynamic prompt blocks ----------
    SCHEMA_SECTION  = render_schema_section_all(ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX)
    CORE_RULES_TEXT = render_core_rules(CORE_COLL_MAP)
    

    # -------- always-defined placeholders --------
    debug_blob: Dict[str, Any] = {}
    coll_key:   str | None     = None
    mongo_filter: Dict[str, Any] = {}
    results:    list           = []
    invalid_fields: List[tuple] = []  # <-- collect invalid filter attempts


    # response = agent.invoke(
    #     {"messages":[{"role":"user","content":query}]},
    #     config={"recursion_limit": 10}
    # )
   

    # STEP 2: Create prompt with history and query
    prompt = SYSTEM_PROMPT.replace("<CONVERSATION_HISTORY>", history)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("system", MATCH_CONTEXT),
        ("system", CORE_RULES_TEXT),
        ("system", MEMORY_PROMPT),
        ("human", query)
    ])

    response = create_react_agent(
        model=llm,
        tools=TOOLS,
        prompt=prompt_template
    ).invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 30, "callbacks": callbacks or [PrintIntermediateStepsHandler()]}
    )

    invoke_cfg = {"recursion_limit": 30}
    if callbacks:
        invoke_cfg["callbacks"] = callbacks
    else:
        # Use our handler by default if not provided
        invoke_cfg["callbacks"] = [PrintIntermediateStepsHandler()]

    # response = agent.invoke(
    #     {"messages": [{"role": "user", "content": query}]},
    #     config=invoke_cfg,
    # )

    ai_msg   = response["messages"][-1]
    spec_str = ai_msg.content.strip()

    # 1) Parse the JSON

    print("\n[LLM-RAW]\n", spec_str)

    try:
        decoder = json.JSONDecoder()
        raw_spec, idx = decoder.raw_decode(spec_str)
    except json.JSONDecodeError as e:
        answer = f"⚠️ JSON parse error:\n{e}\n```json\n{spec_str}\n```"
        return {}, answer, debug_blob

    # Determine whether we have a single query or multiple queries
    if isinstance(raw_spec, dict) and "queries" in raw_spec:
        multi = MultiEntityQuery(**raw_spec)
        query_specs = [q.dict() for q in multi.queries]
        spec = {"queries": query_specs}
    elif isinstance(raw_spec, list):
        multi = MultiEntityQuery(queries=[CollectionQuery(**q) for q in raw_spec])
        query_specs = [q.dict() for q in multi.queries]
        spec = {"queries": query_specs}
    else:
        single = CollectionQuery(**raw_spec)
        spec = single.dict()
        query_specs = [spec]

    answer_parts = []
    chosen_collections: List[str] = []
    filters_debug: List[Dict[str, Any]] = []
    results_debug: List[Dict[str, Any]] = []

    for qspec in query_specs:
        # 1) Normalize dynamic names → base COLL_MAP keys
        raw_name = qspec.get("collection", "")
        if raw_name.startswith("upcoming_match"):
            coll_key = "upcoming_match"
        elif raw_name.startswith("players"):
            coll_key = "players"
        elif raw_name.startswith("venues"):
            coll_key = "venues"
        elif raw_name.startswith("matches"):
            coll_key = "matches"
        else:
            coll_key = next((k for k, v in COLL_MAP.items() if v == raw_name), None)
        if not coll_key:
            answer = f"⚠️ Unknown collection: {raw_name}"
            return spec, answer, debug_blob

        # Normalize filters with collection-aware validation and team-name expansion
        raw_filters = []
        for f in qspec.get("filters", []):
            norm = _normalise_filter(f, COLL_MAP[coll_key], invalid_fields)
            if norm:
                if norm["field"] in ["home_display_team_name", "away_display_team_name"]:
                    for field in ["home_display_team_name", "away_display_team_name"]:
                        raw_filters.append({
                            "field": field,
                            "operation": norm["operation"],
                            "value": norm["value"]
                        })
                else:
                    raw_filters.append(norm)
        qspec["filters"] = raw_filters

        # Normalize legacy sort format, add scheduled_date when needed...
        if isinstance(qspec.get("sort", {}), dict) and "field_name" in qspec["sort"]:
            field = qspec["sort"].pop("field_name")
            order = qspec["sort"].pop("order", "asc")
            qspec["sort"] = {field: order}
        # after
        if coll_key in ("matches", "players"):
            qspec.setdefault("sort", {})
            qspec["sort"].setdefault("scheduled_date", "desc")

        print("\n[NORMALIZED QUERY SPEC]\n", json.dumps(qspec, indent=2))

        # 2) ALWAYS run each query, regardless of filters
        res = _run_query(coll_key, qspec)
        print(f"\n[DEBUG] Result for {coll_key}:", json.dumps(res, indent=2, default=str))

        # 3) Handle errors without breaking out
        if not res.get("ok"):
            debug_blob.setdefault("errors", []).append({
                "collection": coll_key,
                "error":      res.get("error"),
                "filter":     res.get("filter", {})
            })
            continue

        # 4) Accumulate successful results
        chosen_collections.append(coll_key)
        filters_debug.append(res.get("filter", {}))
        results_debug.append(res)

        if coll_key == "upcoming_match":
            answer_parts.append(json.dumps(res["docs"][0], indent=2))
        else:
            answer_parts.extend(f"• {d['summary']}" for d in res["docs"])




    # for qspec in query_specs:
    #     coll_name = qspec.get("collection")
    #     coll_key = next((k for k, v in COLL_MAP.items() if v == coll_name), None)
    #     if not coll_key:
    #         answer = f"⚠️ Unknown collection: {coll_name}"
    #         return spec, answer, debug_blob

    #     # Normalize filters with collection-aware validation and team-name expansion
    #     raw_filters = []
    #     for f in qspec.get("filters", []):
    #         norm = _normalise_filter(f, coll_name, invalid_fields)
    #         if norm:
    #             if norm["field"] in ["home_display_team_name", "away_display_team_name"]:
    #                 new_value = norm["value"]
    #                 raw_filters.append({
    #                     "field": "home_display_team_name",
    #                     "operation": norm["operation"],
    #                     "value": new_value
    #                 })
    #                 raw_filters.append({
    #                     "field": "away_display_team_name",
    #                     "operation": norm["operation"],
    #                     "value": new_value
    #                 })
    #             else:
    #                 raw_filters.append(norm)

    #     qspec["filters"] = [f for f in raw_filters if f is not None]

    #     # Normalize legacy sort format
    #     if isinstance(qspec.get("sort", {}), dict) and "field_name" in qspec["sort"]:
    #         field = qspec["sort"].pop("field_name")
    #         order = qspec["sort"].pop("order", "asc")
    #         qspec["sort"] = {field: order}

    #     # Add scheduled_date sort for matches/players if not present
    #     if coll_name in ["matches_filtered_90696", "players_filtered_90696"]:
    #         sort_clause = qspec.get("sort", {})
    #         if "scheduled_date" not in sort_clause:
    #             sort_clause["scheduled_date"] = "desc"
    #         qspec["sort"] = sort_clause

    #     print("\n[NORMALIZED QUERY SPEC]\n", json.dumps(qspec, indent=2))

    #     res = _run_query(coll_key, qspec)

    #     print(f"\n[DEBUG] Result for {coll_key}:", json.dumps(res, indent=2, default=str))

    #     print("\n[FINAL QUERY SPEC BEFORE RUNNING QUERY]\n", json.dumps(qspec, indent=2))
    #     print("\n[FINAL MONGODB FILTER USED]\n", json.dumps(res.get('filter', {}), indent=2, default=str))

    #     # if not res["ok"]:
    #     #     return spec, f"⚠️ {res['error']}", {**res, "chosen_collections": chosen_collections}

    #     if not res["ok"]:
    #         # Log the failure for this specific collection, but do NOT stop the loop
    #         debug_blob.setdefault("errors", []).append({
    #             "collection": coll_key,
    #             "error":      res["error"],
    #             "filter":     res.get("filter", {})
    #         })
    #         continue

    #     chosen_collections.append(coll_key)
    #     filters_debug.append(res.get("filter", {}))
    #     results_debug.append(res)

    #     if coll_key == "upcoming_match":
    #         answer_parts.append(json.dumps(res["docs"][0], indent=2))
    #     else:
    #         answer_parts.extend(f"• {d['summary']}" for d in res["docs"])

    if invalid_fields:
        debug_blob["invalid_fields"] = invalid_fields

    debug_blob.update({
        "filters": filters_debug,
        "chosen_collections": chosen_collections,
        "results": results_debug,
    })

    # answer = "\n".join(answer_parts)

    # return spec, answer, debug_blob
    
    # If nothing succeeded, surface all errors
    if not results_debug and debug_blob.get("errors"):
        msgs = [f"{e['collection']}: {e['error']}" for e in debug_blob["errors"]]
        return spec, f"⚠️ All queries failed: {'; '.join(msgs)}", debug_blob

    # Otherwise return whatever docs we did retrieve
    answer = "\n".join(answer_parts)
    return spec, answer, debug_blob



# ────────────────────────────────
# 8. Index hints (optional, safe to rerun)
# ────────────────────────────────
def ensure_indexes():
    try:
        db.matches_filtered_90696.create_index("league_name")
        db.matches_filtered_90696.create_index("players")
        db.matches_filtered_90696.create_index("scheduled_date")
        db.matches_filtered_90696.create_index("bat_first_team_score")

        db.players_filtered_90696.create_index("player_name")
        db.players_filtered_90696.create_index("fantasy_points")
        db.players_filtered_90696.create_index("position")

        db.venues_filtered_90696.create_index("city")
        db.venues_filtered_90696.create_index("capacity")
        db.venues_filtered_90696.create_index("run_per_over")
    except Exception as e:
        print(f"Index creation warning: {e}")

ensure_indexes()

# ────────────────────────────────
# 9. CLI test
# ────────────────────────────────
if __name__ == "__main__":
    TEST_QUERIES = [
        "squad of this match",
        # "Find matches in Caribbean Premier League with player Mohammad Nabi after 2020, sort by bat_first_team_score descending",
        # "Find players named Nabi with fantasy points over 5 and position All-rounder, sort by fantasy points descending",
        # "provide match with highest runs of Indian T20 league in 2025",
        # "highest runs of virat kohli",
        # "Show all matches in Indian Premier League 2024.",
        # "Find venues in Hyderabad with capacity over 50000, sort by run_per_over ascending",
    ]
    for q in TEST_QUERIES:
        print("\n🠚  ", q)
        spec, answer, dbg = run_search_agent(q,history=MATCH_CONTEXT)

        # overwrite the log file with this run’s data
        log_entry = {
            "timestamp":    _dt.utcnow().isoformat() + "Z",
            "user_query":   q,
            "parsed_spec":  spec,
            "final_answer": answer
        }
        LOG_PATH.write_text(json.dumps(log_entry, indent=2))

        # still print to console
        print(answer)
