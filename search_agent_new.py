# search_agent.py
import os, json
import re
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
from schema_registry import load_registry, get_all_fields, get_collection_names, get_descriptions

def _extract_json_spec(text: str) -> dict:
    """
    Pull a JSON object out of LLM output that may include
    code fences, nested fences, or extra prose.
    """
    if not isinstance(text, str):
        raise ValueError("Spec text is not a string")

    # 1) Prefer the last ```json ... ``` fenced block
    blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for blk in reversed(blocks):
        try:
            return json.loads(blk)
        except Exception:
            pass

    # 2) Fallback: take the largest {...} span
    start = text.find("{")
    end   = text.rfind("}")
    if 0 <= start < end:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3) Last attempt: strip leading/trailing junk lines and retry
    cleaned = text.strip().strip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"could not parse JSON spec: {e}")
    
def _load_schema_live():
    reg = load_registry()
    all_fields   = get_all_fields(reg)
    descriptions = get_descriptions(reg)
    options_max  = reg.get("options_max", 20)
    return all_fields, descriptions, options_max

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

MATCH_CONTEXT = ""

LOG_PATH = Path("last_search_run.json")

from schema_registry import load_registry, get_connection_config
load_dotenv()
# Prefer Admin UI config first; fall back to env; finally to hardcoded defaults
_cfg = get_connection_config(load_registry()) or {}
MONGO_URI = _cfg.get("mongo_uri")
DB_NAME   = _cfg.get("mongo_db") 

client = MongoClient(MONGO_URI); db = client[DB_NAME]

db         = client[DB_NAME]

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o3-mini")

_reg = load_registry()
ALL_FIELDS        = get_all_fields(_reg)        # {coll_name: [fields]}
DESCRIPTIONS      = get_descriptions(_reg)      # {coll_name: description}
COLLECTION_NAMES  = get_collection_names(_reg)  # ["orders", "customers", ...]
OPTIONS_MAX       = _reg.get("options_max", 20)


SCHEMA_SECTION  = render_schema_section_all(ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX)

CORE_RULES_TEXT = render_core_rules(COLLECTION_NAMES)
USER_MATCH_CONTEXT = render_match_context("")  # Make MATCH_CONTEXT user-editable; can be empty

# 🔧 Back-compat so older code keeps working:        # {'matches': 'matches_filtered_…', ...}
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
    Pick the collection whose schema contains ALL referenced fields (filters + sort).
    Returns the actual collection name (or None).
    """
    fields = [filt["field"] for filt in spec.get("filters", [])]
    sf = next(iter(spec.get("sort", {})), None)
    if sf:
        fields.append(sf)

    wanted = set(fields)
    for coll_name, metas in SEARCHABLE_FIELDS.items():  # SEARCHABLE_FIELDS == ALL_FIELDS
        schema_fields = {m["name"] for m in metas}
        if wanted.issubset(schema_fields):
            return coll_name
    return None


# ────────────────────────────────
# 3. Shared query executor
# ────────────────────────────────


# 3.a  Post-process each filter coming from the LLM
#      • unwrap ["All-rounder"] → "All-rounder"
#      • if op = keyword but the field is scalar string (and schema
#        doesn’t list keyword) → flip to regex
# def _normalise_filter(filt: dict) -> dict:
def _normalise_filter(filt: dict, collection: str, invalid_fields: Optional[List[tuple]] = None) -> dict | None:
    out = filt.copy()

    # 1) unwrap single-element lists so $regex always gets a string
    # unwrap ["value"] -> "value"
    if isinstance(out.get("value"), list) and len(out["value"]) == 1:
        out["value"] = out["value"][0]

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

@tool("search_collection")
def search_collection(collection: str, **q) -> List[Dict[str, Any]]:
    """Search any configured collection by name."""
    return _run_query(collection, q)

TOOLS = [search_collection]

def _run_query(collection: str, spec: Dict[str, Any]) -> QueryResult:
    try:
        parsed    = EntityQuery(**spec)
        # Accept either a role key or a real collection name:
        coll_name   = collection
        coll        = db[coll_name]
        mongo_filter = {}
        regular_filters = []

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
        
        # Always return full objects (no summary-only projection)
        projection = {"_id": 0}

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
# 5. Prompt
# ────────────────────────────────


# List of strings with "name (type) [operations]" format for matches
# matches_fields = [
#     f"{f['name']} ({f['type']}) [{', '.join(f['operations'])}]"
#     for f in SEARCHABLE_FIELDS[COLL_MAP["matches"]]
# ]

# players_fields = [
#     f"{f['name']} ({f['type']}) [{', '.join(f['operations'])}]"
#     for f in SEARCHABLE_FIELDS[COLL_MAP["players"]]
# ]

# venues_fields = [
#     f"{f['name']} ({f['type']}) [{', '.join(f['operations'])}]"
#     for f in SEARCHABLE_FIELDS[COLL_MAP["venues"]]
# ]

# def format_fields_with_options(fields: List[Dict[str, Any]]) -> List[str]:
#     formatted = []
#     for f in fields:
#         base = f"{f['name']} ({f['type']}) [{', '.join(f['operations'])}]"
#         if "options" in f:
#             opts = ", ".join(f["options"])
#             base += f" (options: {opts})"
#         formatted.append(base)
#     return formatted

# matches_fields = format_fields_with_options(SEARCHABLE_FIELDS[COLL_MAP["matches"]])
# players_fields = format_fields_with_options(SEARCHABLE_FIELDS[COLL_MAP["players"]])
# venues_fields = format_fields_with_options(SEARCHABLE_FIELDS[COLL_MAP["venues"]])

# print(matches_fields)

MATCH_CONTEXT = ""

# Initialize MemoryAgent for language detection only
# mem = MemoryAgent(k=5)

# U = COLL_MAP.get("upcoming_match", "upcoming_match")
# M = COLL_MAP.get("matches", "matches")
# P = COLL_MAP.get("players", "players")
# V = COLL_MAP.get("venues",  "venues")

SYSTEM_PROMPT = """
You are an expert MongoDB query planner for three collections:

{schema_section}

Only use the operations listed for each field above.
If a field has a list of allowed options (shown after →), you must use one of those exact values for that field. Do not invent or assume values not in the list.

#Also use the memory context if available, if any of the last 3 answers say "no data available" or similar, ignore that answer for reasoning.
"If the user query contains pronouns (e.g., 'he', 'him', 'his'), 
always resolve them to the correct entity using the most recent relevant memory context. 
Never use a pronoun as a value in any query field."

Your job:

1. Read the user’s natural-language request.

2. Produce **only** a JSON object with output either:
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
3. ALWAYS use the **search_collection** tool and pass the exact `collection` name shown above.
   - For multi-collection queries, call the tool once per collection.
   - If data needs to be combined across collections, create multiple queries.
4. After the tool returns, write a concise answer for the user.
5. For queries asking for the "highest", "most", "top", or "best", use a sort on the relevant field
   (descending) and set limit to the required number.
6. If the user explicitly specifies a date or date range, include it in the query filters.
   Otherwise, do not add any date filters.
7. For questions that clearly require data from multiple collections, 
   ALWAYS use the multi-collection format with "queries" array.

   
For Filtering, REMEMBER:
When extracting entity names, correct spelling mistakes and use the official name as per your knowledge.

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
• keyword – substring match inside *array* fields only; **do not use on scalar strings**
• range   – {{{{"$gte": ..}}}}, {{{{"$lte": ..}}}} on numbers or dates (YYYY-MM-DD)
• sort    – asc / desc on sortable numeric/date fields


Output Instructions:
Return ONLY a single JSON object (no backticks, no code fences, no extra text).
• For single-collection requests, return the query object directly.
• For multi-collection requests, return {{{{"queries": [{{{{...}}}}, ...]}}}} with one object per collection.
Do not add any text outside the JSON.

""".format(schema_section=SCHEMA_SECTION)
SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")
# PROMPT_without_memory = ChatPromptTemplate.from_messages(
#     [("system", SYSTEM_PROMPT), ("system", MATCH_CONTEXT), ("placeholder", "{messages}")]
# )

# ────────────────────────────────
# 6. LLM & agent
# ────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

def get_llm():
    """
    Create the LLM lazily and only if an API key is available.
    Return None if missing so the caller can fail gracefully.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    # Newer langchain-openai uses `api_key` (not openai_api_key)
    return ChatOpenAI(model=OPENAI_MODEL, api_key=api_key)

def get_memory_prompt(n):
    memories = get_last_memories(n)
    mem_text = "\n".join(
        [f"Previous Q: {m['query']}\nPrevious A: {m['answer']}" for m in memories if "no data" not in m['answer'].lower()]
    )
    return (
        "### RECENT MEMORY CONTEXT\n"
        "If any of the last answers below say 'no data available' or similar, ignore that answer for reasoning.\n"
        f"{mem_text}\n"
        "When the user query contains pronouns like 'he', 'him', 'his''इसको','इसके'(any language), always resolve them to the correct player name using the most recent relevant memory. For example, if the last answer was about 'X', and the user now asks 'his last 5 matches', use 'X' as the value for 'player_name'.\n"
        "Never use a pronoun as a value in any query field. For example, if the last answer was about 'Virat Kohli', and the user now asks 'How many runs did he make?', use 'Virat Kohli' as the value for 'player_name'.\n"
        "The most recent memory (highest weight) is listed first.\n"
    )


# If you currently build PROMPT via ChatPromptTemplate, keep that; just swap in variables:
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", CORE_RULES_TEXT),        # core rules injected here (generalised)
    ("system", USER_MATCH_CONTEXT),     # optional, can be "" (user add-on)      # you already compute this:contentReference[oaicite:4]{index=4}
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

    global ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX
    global SEARCHABLE_FIELDS, CORE_RULES_TEXT, SCHEMA_SECTION

    ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX = _load_schema_live()
    SEARCHABLE_FIELDS = ALL_FIELDS
    COLLECTION_NAMES = list(ALL_FIELDS.keys())        # NEW
    SCHEMA_SECTION  = render_schema_section_all(ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX)
    CORE_RULES_TEXT = render_core_rules(COLLECTION_NAMES)  # REPLACE old CORE_COLL_MAP call
    

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
        ("system", CORE_RULES_TEXT),
        ("system", USER_MATCH_CONTEXT),  # optional; may be ""
        ("human", query)
    ])

    llm = get_llm()
    
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
        raw_spec = _extract_json_spec(spec_str)
    except Exception as e:
        # show the entire returned text so you can see what the model sent
        answer = f"⚠️ JSON parse error:\n{e}\n```json\n{spec_str}\n```"
        debug_blob["spec"] = {}
        return {}, answer, debug_blob
    
    debug_blob["spec"] = raw_spec  # keep it for the UI
    qspec = raw_spec

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
        # 1) Determine the actual collection name requested
        coll_name = qspec.get("collection", "")
        if not coll_name:
            answer = "⚠️ Missing 'collection' in spec."
            return spec, answer, debug_blob
        # Validate against known schema (admin-registered)
        if coll_name not in SEARCHABLE_FIELDS:
            answer = f"⚠️ Unknown collection: {coll_name}"
            return spec, answer, debug_blob


        # # Normalize filters with collection-aware validation and team-name expansion
        # raw_filters = []
        # for f in qspec.get("filters", []):
        #     norm = _normalise_filter(f, coll_name, invalid_fields)
        #     if norm:
        #         raw_filters.append(norm)
        # qspec["filters"] = raw_filters

        # # Normalize legacy sort format, add scheduled_date when needed...
        # if isinstance(qspec.get("sort", {}), dict) and "field_name" in qspec["sort"]:
        #     field = qspec["sort"].pop("field_name")
        #     order = qspec["sort"].pop("order", "asc")
        #     qspec["sort"] = {field: order}

        # print("\n[NORMALIZED QUERY SPEC]\n", json.dumps(qspec, indent=2))

        # 2) ALWAYS run each query, regardless of filters
        res = _run_query(coll_name, qspec)
        print(f"\n[DEBUG] Result for {coll_name}:", json.dumps(res, indent=2, default=str))

        # 3) Handle errors without breaking out
        if not res.get("ok"):
            debug_blob.setdefault("errors", []).append({
                "collection": coll_name,
                "error":      res.get("error"),
                "filter":     res.get("filter", {})
            })
            continue

        # 4) Accumulate successful results
        chosen_collections.append(coll_name)
        filters_debug.append(res.get("filter", {}))
        results_debug.append(res)

        # Always output full object(s)
        for d in res["docs"]:
            answer_parts.append(json.dumps(d, indent=2, default=str))




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

    # Otherwise return whatever docs we did retrieve (full objects)
    answer = "\n\n".join(answer_parts)
    return spec, answer, debug_blob



# ────────────────────────────────
# 8. Index hints (optional, safe to rerun)
# ────────────────────────────────
def ensure_indexes():
    """Create indexes on fields marked sortable in the admin schema."""
    try:
        for coll_name, fields in SEARCHABLE_FIELDS.items():
            for f in fields:
                if "operations" in f and "sort" in f["operations"]:
                    try:
                        db[coll_name].create_index(f["name"])
                    except Exception as ie:
                        print(f"Index warn {coll_name}.{f['name']}: {ie}")
    except Exception as e:
        print(f"Index creation warning: {e}")

ensure_indexes()

# ────────────────────────────────
# 9. CLI test
# ────────────────────────────────
if __name__ == "__main__":
    TEST_QUERIES = [
        "Show top 3 orders by total amount in the last 30 days",
        "List customers from Bangalore sorted by signup_date desc limit 5"
    ]
    for q in TEST_QUERIES:
        print("\n🠚  ", q)
        spec, answer, dbg = run_search_agent(q, history="")

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