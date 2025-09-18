# search_agent.py
import os, json
import re
import traceback
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
from cache_memory import get_last_memories, get_memory_prompt
from llm_services import call_search_agent_model
from typing import List, Optional
from langchain_core.callbacks import BaseCallbackHandler

from difflib import get_close_matches
# search_agent_new.py  (only showing relevant edits)
from core_rules import render_core_rules, render_match_context, render_schema_section_with_relations
from schema_registry import (
    load_registry, get_all_fields, get_collection_names, 
    get_descriptions, get_user_collections, get_user_access_config,
    get_user_accessible_collections, validate_join_relationship
)

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


SCHEMA_SECTION  = render_schema_section_with_relations(ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX)

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
    owner: Optional[str] = None  # For joined fields, specify which join alias

class Join(BaseModel):
    alias: str
    collection: str
    from_: str = "base"  # "base" or alias name
    local_field: str
    foreign_field: str
    cardinality: str = "one_to_many"  # "one_to_many", "many_to_one", "one_to_one"
    join_type: str = "left"  # "left" or "inner"

class EntityQuery(BaseModel):
    filters: List[EntityFilter]
    joins: List[Join] = []  # Optional joins array
    sort:   Dict[str, str] = {}
    limit:  int = 20



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
    if isinstance(out.get("value"), list) and len(out["value"]) == 1:
        out["value"] = out["value"][0]

    # 2) Handle date normalization for range queries
    if out.get("operation") == "range" and isinstance(out.get("value"), dict):
        # Get field metadata to check if it's a date field
        for field_meta in SEARCHABLE_FIELDS.get(collection, []):
            if field_meta["name"] == out["field"] and field_meta["type"] == "date":
                # Normalize date ranges to include full day
                for op in ["$gte", "$gt"]:
                    if op in out["value"] and isinstance(out["value"][op], str):
                        # If the date doesn't have time component, set to beginning of day
                        if len(out["value"][op]) <= 10:  # YYYY-MM-DD format
                            out["value"][op] = f"{out['value'][op].split()[0]}T00:00:00"
                
                for op in ["$lte", "$lt"]:
                    if op in out["value"] and isinstance(out["value"][op], str):
                        # If the date doesn't have time component, set to end of day
                        if len(out["value"][op]) <= 10:  # YYYY-MM-DD format
                            out["value"][op] = f"{out['value'][op].split()[0]}T23:59:59"
                break

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
        # Parse the EntityQuery (now with joins support)
        parsed = EntityQuery(**spec)
        
        # Check if we have joins - if so, use aggregation pipeline
        if parsed.joins:
            return _run_aggregation_query(collection, parsed)
        else:
            return _run_simple_query(collection, parsed)
            
    except Exception as exc:
        import traceback
        print(f"Query error: {exc}")
        print(traceback.format_exc())
        return {"ok": False, "error": f"Query failed: {exc}", "filter": {}}

def _run_simple_query(collection: str, parsed: EntityQuery) -> QueryResult:
    """Execute a simple query without joins."""
    try:
        # Get RLS values and role from session state (set in frontend)
        rls_values = {}
        user_role = "user"  # default role
        
        try:
            import streamlit as st
            # Get all RLS values and role from session state (set in frontend) 
            rls_values = st.session_state.get("rls_values", {})
            user_role = st.session_state.get("user_role", "user")
            
            # Filter out empty values - be more careful with empty string checks
            rls_values = {k: v for k, v in rls_values.items() if v is not None and str(v).strip() != ""}
        except:
            # Not running in Streamlit context
            pass
        
        # Use RLS engine if any RLS values are provided
        if rls_values:
            from rls_engine import create_rls_interceptor
            
            # Create RLS interceptor with user context (use user_id if available, otherwise use default)
            primary_user_id = rls_values.get('user_id') or "anonymous"
            rls = create_rls_interceptor(primary_user_id, user_role)
            
            # Check if RLS should be bypassed
            if rls.should_bypass_rls():
                print(f"\n[RLS] RLS bypassed for user {primary_user_id} with role {user_role}")
                # Process without RLS (admin or bypass role)
                coll = db[collection]
                mongo_filter = {}
                regular_filters = []

                # Build filters without RLS
                for f in parsed.filters:
                    fld_meta = next((m for m in SEARCHABLE_FIELDS[collection]
                                    if m["name"] == f.field), None)
                    if not fld_meta or f.operation not in fld_meta["operations"]:
                        return {"ok": False,
                                "error": f"Invalid field/operation: {f.field},{f.operation}",
                                "filter": mongo_filter}

                    # Add filters without RLS restrictions
                    if f.operation == "regex":
                        regular_filters.append({f.field: {"$regex": f.value, "$options": "i"}})
                    elif f.operation == "keyword":
                        regular_filters.append({f.field: {"$elemMatch": {"$regex": f.value, "$options": "i"}}})
                    elif f.operation == "range":
                        if fld_meta["type"] == "date":
                            rng = {}
                            for op, v in f.value.items():
                                if isinstance(v, str):
                                    try:
                                        if len(v) <= 10:  # YYYY-MM-DD
                                            if op in ["$gte", "$gt"]:
                                                rng[op] = f"{v} 00:00:00"
                                            else:
                                                rng[op] = f"{v} 23:59:59"
                                        else:
                                            rng[op] = v
                                    except ValueError:
                                        rng[op] = v
                                else:
                                    rng[op] = v
                            regular_filters.append({f.field: rng})
                        else:
                            regular_filters.append({f.field: f.value})

                # Combine filters without RLS
                if len(regular_filters) == 1:
                    mongo_filter = regular_filters[0]
                elif len(regular_filters) > 1:
                    mongo_filter = {"$and": regular_filters}

            else:
                print(f"\n[RLS] Applying multi-layer RLS for user {primary_user_id} with values: {rls_values}")
                # Apply RLS filtering
                coll = db[collection]
                
                # Build base query
                base_query = {}
                regular_filters = []

                # Build filters from user query
                for f in parsed.filters:
                    fld_meta = next((m for m in SEARCHABLE_FIELDS[collection]
                                    if m["name"] == f.field), None)
                    if not fld_meta or f.operation not in fld_meta["operations"]:
                        return {"ok": False,
                                "error": f"Invalid field/operation: {f.field},{f.operation}",
                                "filter": base_query}

                    # Add user filters
                    if f.operation == "regex":
                        regular_filters.append({f.field: {"$regex": f.value, "$options": "i"}})
                    elif f.operation == "keyword":
                        regular_filters.append({f.field: {"$elemMatch": {"$regex": f.value, "$options": "i"}}})
                    elif f.operation == "range":
                        if fld_meta["type"] == "date":
                            rng = {}
                            for op, v in f.value.items():
                                if isinstance(v, str):
                                    try:
                                        if len(v) <= 10:  # YYYY-MM-DD
                                            if op in ["$gte", "$gt"]:
                                                rng[op] = f"{v} 00:00:00"
                                            else:
                                                rng[op] = f"{v} 23:59:59"
                                        else:
                                            rng[op] = v
                                    except ValueError:
                                        rng[op] = v
                                else:
                                    rng[op] = v
                            regular_filters.append({f.field: rng})
                        else:
                            regular_filters.append({f.field: f.value})

                # Combine user filters
                if len(regular_filters) == 1:
                    base_query = regular_filters[0]
                elif len(regular_filters) > 1:
                    base_query = {"$and": regular_filters}
                else:
                    base_query = {}

                # Apply RLS enhancement to add multi-layer filtering
                mongo_filter = rls.enhance_find_query_with_multilayer(collection, base_query, rls_values)
                # Log the type information for debugging
                print(f"[RLS] Enhanced query type debug:")
                for key, value in mongo_filter.items():
                    if key != "$and":  # Skip complex nested structures
                        print(f"   {key}: {value} ({type(value).__name__})")

        else:
            print(f"\n[RLS] No RLS user selected - no filtering applied")
            # No RLS user selected - process normally
            coll = db[collection]
            mongo_filter = {}
            regular_filters = []

            # Build filters without RLS
            for f in parsed.filters:
                fld_meta = next((m for m in SEARCHABLE_FIELDS[collection]
                                if m["name"] == f.field), None)
                if not fld_meta or f.operation not in fld_meta["operations"]:
                    return {"ok": False,
                            "error": f"Invalid field/operation: {f.field},{f.operation}",
                            "filter": mongo_filter}

                if f.operation == "regex":
                    regular_filters.append({f.field: {"$regex": f.value, "$options": "i"}})
                elif f.operation == "keyword":
                    regular_filters.append({f.field: {"$elemMatch": {"$regex": f.value, "$options": "i"}}})
                elif f.operation == "range":
                    if fld_meta["type"] == "date":
                        rng = {}
                        for op, v in f.value.items():
                            if isinstance(v, str):
                                try:
                                    if len(v) <= 10:
                                        if op in ["$gte", "$gt"]:
                                            rng[op] = f"{v} 00:00:00"
                                        else:
                                            rng[op] = f"{v} 23:59:59"
                                    else:
                                        rng[op] = v
                                except ValueError:
                                    rng[op] = v
                            else:
                                rng[op] = v
                        regular_filters.append({f.field: rng})
                    else:
                        regular_filters.append({f.field: f.value})

            # Combine filters without RLS
            if len(regular_filters) == 1:
                mongo_filter = regular_filters[0]
            elif len(regular_filters) > 1:
                mongo_filter = {"$and": regular_filters}

        # Execute query
        projection = {"_id": 0}
        sort_clause = [(fld, 1 if d.lower() == "asc" else -1)
                       for fld, d in parsed.sort.items()]
        
        print(f"\n[QUERY] Running query on {collection} with filter:", json.dumps(mongo_filter, default=str))
        check_count = coll.count_documents(mongo_filter)
        print(f"[QUERY] Documents found: {check_count}")

        cursor = _apply_sort(
            coll.find(mongo_filter, projection),
            sort_clause
        ).limit(parsed.limit)

        docs = list(cursor)
        if not docs:
            return {"ok": False, "error": f"No results found (confirmed count: {check_count})",
                    "filter": mongo_filter}

        safe_json = json.loads(json.dumps(docs, default=_safe))
        return {"ok": True, "docs": safe_json, "filter": mongo_filter}

    except Exception as exc:
        import traceback
        print(f"Query error: {exc}")
        print(traceback.format_exc())
        return {"ok": False, "error": f"Query failed: {exc}",
                "filter": {}}

def _run_aggregation_query(collection: str, parsed: EntityQuery) -> QueryResult:
    """Execute a query with joins using MongoDB aggregation pipeline."""
    try:
        from schema_registry import validate_join_relationship
        
        # Get RLS values and role from session state (set in frontend)
        rls_values = {}
        user_role = "user"  # default role
        
        try:
            import streamlit as st
            rls_values = st.session_state.get("rls_values", {})
            user_role = st.session_state.get("user_role", "user")
            rls_values = {k: v for k, v in rls_values.items() if v is not None and str(v).strip() != ""}
        except:
            pass
        
        coll = db[collection]
        pipeline = []
        
        # Validate joins
        for join in parsed.joins:
            # Validate join relationship exists in schema
            join_dict = join.dict() if hasattr(join, 'dict') else join.__dict__
            if not validate_join_relationship(collection, join_dict):
                print(f"Warning: Join relationship not defined in schema for {collection} -> {join.collection}")
        
        # Build initial $match stage for base collection filters and RLS
        base_match = {}
        base_filters = []
        
        # Process base collection filters (no owner specified)
        for f in parsed.filters:
            if not f.owner:  # Base collection filter
                fld_meta = next((m for m in SEARCHABLE_FIELDS[collection]
                               if m["name"] == f.field), None)
                if not fld_meta or f.operation not in fld_meta["operations"]:
                    return {"ok": False,
                           "error": f"Invalid field/operation: {f.field},{f.operation}",
                           "filter": {}}
                
                if f.operation == "regex":
                    base_filters.append({f.field: {"$regex": f.value, "$options": "i"}})
                elif f.operation == "keyword":
                    base_filters.append({f.field: {"$elemMatch": {"$regex": f.value, "$options": "i"}}})
                elif f.operation == "range":
                    if fld_meta["type"] == "date":
                        rng = {}
                        for op, v in f.value.items():
                            if isinstance(v, str) and len(v) <= 10:
                                if op in ["$gte", "$gt"]:
                                    rng[op] = f"{v} 00:00:00"
                                else:
                                    rng[op] = f"{v} 23:59:59"
                            else:
                                rng[op] = v
                        base_filters.append({f.field: rng})
                    else:
                        base_filters.append({f.field: f.value})
        
        # Combine base filters
        if base_filters:
            if len(base_filters) == 1:
                base_match = base_filters[0]
            else:
                base_match = {"$and": base_filters}
        
        # Apply RLS to base query
        if rls_values:
            from rls_engine import create_rls_interceptor
            primary_user_id = rls_values.get('user_id') or "anonymous"
            rls = create_rls_interceptor(primary_user_id, user_role)
            
            if not rls.should_bypass_rls():
                base_match = rls.enhance_find_query_with_multilayer(collection, base_match, rls_values)

        
        # Add initial $match stage if we have base filters or RLS
        if base_match:
            pipeline.append({"$match": base_match})
        
        # Add $lookup stages for each join
        for join in parsed.joins:
            lookup_stage = {
                "$lookup": {
                    "from": join.collection,
                    "localField": join.local_field,
                    "foreignField": join.foreign_field,
                    "as": join.alias
                }
            }
            pipeline.append(lookup_stage)
            
            # Add $unwind if cardinality is one_to_many and we need to flatten
            if join.cardinality == "one_to_many":
                # Check if any filters apply to this joined collection
                has_joined_filters = any(f.owner == join.alias for f in parsed.filters)
                # Check if any sort applies to this joined collection  
                has_joined_sort = any(key.startswith(f"{join.alias}.") for key in parsed.sort.keys())
                
                if has_joined_filters or has_joined_sort:
                    if join.join_type == "left":
                        unwind_stage = {
                            "$unwind": {
                                "path": f"${join.alias}",
                                "preserveNullAndEmptyArrays": True
                            }
                        }
                    else:
                        unwind_stage = {"$unwind": f"${join.alias}"}
                    pipeline.append(unwind_stage)
        
        # Add $match stages for joined collection filters
        joined_matches = {}
        for f in parsed.filters:
            if f.owner:  # Joined collection filter
                # Find the join alias
                join_alias = f.owner
                joined_field = f"{join_alias}.{f.field}"
                
                # Get target collection for field validation
                target_collection = None
                for join in parsed.joins:
                    if join.alias == join_alias:
                        target_collection = join.collection
                        break
                
                if target_collection:
                    fld_meta = next((m for m in SEARCHABLE_FIELDS.get(target_collection, [])
                                   if m["name"] == f.field), None)
                    
                    if fld_meta and f.operation in fld_meta["operations"]:
                        if f.operation == "regex":
                            joined_matches[joined_field] = {"$regex": f.value, "$options": "i"}
                        elif f.operation == "keyword":
                            joined_matches[joined_field] = {"$elemMatch": {"$regex": f.value, "$options": "i"}}
                        elif f.operation == "range":
                            if fld_meta["type"] == "date":
                                rng = {}
                                for op, v in f.value.items():
                                    if isinstance(v, str) and len(v) <= 10:
                                        if op in ["$gte", "$gt"]:
                                            rng[op] = f"{v} 00:00:00"
                                        else:
                                            rng[op] = f"{v} 23:59:59"
                                    else:
                                        rng[op] = v
                                joined_matches[joined_field] = rng
                            else:
                                joined_matches[joined_field] = f.value
        
        # Add joined filters as $match stage
        if joined_matches:
            pipeline.append({"$match": joined_matches})
        
        # Add $sort stage
        if parsed.sort:
            sort_spec = {}
            for field, direction in parsed.sort.items():
                sort_direction = 1 if direction.lower() == "asc" else -1
                sort_spec[field] = sort_direction
            pipeline.append({"$sort": sort_spec})
        
        # Add $limit stage
        if parsed.limit:
            pipeline.append({"$limit": parsed.limit})
        
        # Remove _id field
        pipeline.append({"$project": {"_id": 0}})
        
        print(f"\n[AGGREGATION] Running pipeline on {collection}:", json.dumps(pipeline, default=str, indent=2))
        
        # Execute aggregation pipeline
        cursor = coll.aggregate(pipeline)
        docs = list(cursor)
        
        if not docs:
            return {"ok": False, "error": "No results found", "filter": pipeline}
        
        safe_json = json.loads(json.dumps(docs, default=_safe))
        return {"ok": True, "docs": safe_json, "filter": pipeline}
        
    except Exception as exc:
        import traceback
        print(f"Aggregation query error: {exc}")
        print(traceback.format_exc())
        return {"ok": False, "error": f"Aggregation query failed: {exc}", "filter": {}}

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
from datetime import datetime
 
# Get today's date
today_date = datetime.today().date()
SYSTEM_PROMPT = """
You are an expert MongoDB query planner for the following collections:

{schema_section}

Only use the operations listed for each field above.
If a field has a list of allowed options (shown after →), you must use one of those exact values for that field. Do not invent or assume values not in the list.

Also use the memory context if available (Always if there is any follow up question(detect yourself)).
"If the user query contains pronouns (e.g., 'he', 'him', 'his'), 
always resolve them to the correct entity using the most recent relevant memory context. 
Never use a pronoun as a value in any query field."

**CRITICAL JOIN DECISION RULES - READ CAREFULLY:**

❌ **NEVER DO THESE WRONG PATTERNS:**

1. **DON'T search for names in ID fields:**
   ```json
   {{{{"collection": "appointments", "filters": [{{{{"field": "patient_id", "operation": "regex", "value": "david"}}}}]}}}}
   ```
   **Problem**: patient_id contains "P001", "P002", not names like "david"
   
2. **DON'T try direct patient↔doctor joins:**
   ```json
   {{{{"collection": "patients", "joins": [{{{{"alias": "doc", "collection": "doctors", "local_field": "doctor_id", "foreign_field": "doctor_id"}}}}]}}}}
   ```
   **Problem**: patients don't have doctor_id field

3. **DON'T ignore the data model relationships:**
   - patients ↔ appointments ↔ doctors (appointments is the bridge)

✅ **ALWAYS USE THESE CORRECT PATTERNS:**

**Pattern 1: Find appointments for a patient by name**
```json
{{{{
  "collection": "patients",
  "joins": [{{{{
    "alias": "appt",
    "collection": "appointments", 
    "from": "base",
    "local_field": "patient_id",
    "foreign_field": "patient_id",
    "cardinality": "one_to_many",
    "join_type": "left"
  }}}}],
  "filters": [
    {{{{"field": "first_name", "operation": "regex", "value": "david"}}}}
  ],
  "sort": {{{{"appt.appointment_date": "desc"}}}}
}}}}
```

**Pattern 2: Find appointments for a doctor by name**
```json
{{{{
  "collection": "doctors",
  "joins": [{{{{
    "alias": "appt",
    "collection": "appointments",
    "from": "base", 
    "local_field": "doctor_id",
    "foreign_field": "doctor_id",
    "cardinality": "one_to_many",
    "join_type": "left"
  }}}}],
  "filters": [
    {{{{"field": "first_name", "operation": "regex", "value": "linda"}}}},
    {{{{"field": "last_name", "operation": "regex", "value": "brown"}}}}
  ],
  "sort": {{{{"appt.appointment_date": "desc"}}}}
}}}}
```

**Pattern 3: Complex query needing all three collections**
```json
{{{{
  "collection": "appointments",
  "joins": [
    {{{{
      "alias": "patient",
      "collection": "patients",
      "from": "base",
      "local_field": "patient_id",
      "foreign_field": "patient_id",
      "cardinality": "many_to_one",
      "join_type": "inner"
    }}}},
    {{{{
      "alias": "doctor", 
      "collection": "doctors",
      "from": "base",
      "local_field": "doctor_id",
      "foreign_field": "doctor_id",
      "cardinality": "many_to_one",
      "join_type": "inner"
    }}}}
  ],
  "filters": [
    {{{{"owner": "patient", "field": "first_name", "operation": "regex", "value": "david"}}}},
    {{{{"owner": "doctor", "field": "last_name", "operation": "regex", "value": "brown"}}}}
  ]
}}}}
```

**DECISION TREE FOR QUERIES:**

1. **Question about patient name + appointments?** 
   → Start with "patients", join "appointments"

2. **Question about doctor name + appointments?**
   → Start with "doctors", join "appointments"

3. **Need patient + doctor + appointment details?**
   → Start with "appointments", join both "patients" and "doctors"

4. **Only appointment details (by ID, date, status)?**
   → Use "appointments" alone, no joins needed

5. **Only patient/doctor demographics?**
   → Use "patients"/"doctors" alone, no joins needed

Your job:

1. Read the user's natural-language request.

2. **Apply the decision tree above** to determine correct approach.

3. Produce **only** a JSON object with one of these formats:

   **Single Collection (No Joins Needed):**
   {{{{
     "collection": "collection_name",
     "filters": [{{{{"field": "...", "operation": "...", "value": ...}}}}, ...],
     "sort":    {{{{"field_name": "asc|desc"}}}},
     "limit":   <int>
   }}}}

   **Single Collection with Joins (WHEN DATA SPANS MULTIPLE COLLECTIONS):**
   {{{{
     "collection": "base_collection",
     "joins": [
       {{{{
         "alias": "short_name",
         "collection": "target_collection",
         "from": "base",
         "local_field": "field_in_base",
         "foreign_field": "field_in_target",
         "cardinality": "one_to_many|many_to_one|one_to_one",
         "join_type": "left|inner"
       }}}}
     ],
     "filters": [
       {{{{"field": "base_field", "operation": "...", "value": ...}}}},
       {{{{"owner": "alias_name", "field": "joined_field", "operation": "...", "value": ...}}}}
     ],
     "sort": {{{{"field_name": "asc|desc"}}}},
     "limit": <int>
   }}}}

4. **filters[] field usage:**
   - For base collection fields: {{{{"field": "field_name", "operation": "...", "value": ...}}}}
   - For joined fields: {{{{"field": "field_name", "operation": "...", "value": ..., "owner": "join_alias"}}}}

5. **sort[] field usage:**
   - Base collection fields: {{{{"field_name": "asc|desc"}}}}
   - Joined fields: {{{{"alias.field_name": "asc|desc"}}}} (use dot notation)

6. For queries asking for the "highest", "most", "top", or "best", use a sort on the relevant field
   (descending) and set limit to the required number.

7. If the user explicitly specifies a date or date range, include it in the query filters.
   Otherwise, do not add any date filters.

8. If user said today use {today_date}, use the actual current todays date in YYYY-MM-DD format.

Memory context:
<CONVERSATION_HISTORY>

#Also use the memory context if available, if any of the last few answers say "no data available" or similar, ignore that answer for reasoning.
    "If the user query contains pronouns (e.g., 'he', 'him', 'his'), 
    always resolve them to the correct entity using the most recent relevant memory context. 
    Never use a pronoun as a value in any query field."
    "When the user query contains pronouns like 'he', 'him', 'his''इसको','इसके'(any languadge), always resolve them to the correct player name using the most recent relevant memory. For example, if the last answer was about 'X', and the user now asks 'his last 5 matches', use 'X' as the value for 'player_name'.\n"
    "Never use a pronoun as a value in any query field. For example, if the last answer was about 'Virat Kohli', and the user now asks 'How many runs did he make?', use 'Virat Kohli' as the value for 'player_name'.\n"
    "The most recent memory (highest weight) is listed first.\n"

Allowed operations:
• regex   – case-insensitive substring match (strings)
• keyword – substring match inside *array* fields only; **do not use on scalar strings**
• range   – {{{{"$gte": ..}}}}, {{{{"$lte": ..}}}} on numbers or dates (YYYY-MM-DD)
• sort    – asc / desc on sortable numeric/date fields

Output Instructions:
Return ONLY a single JSON object (no backticks, no code fences, no extra text).
Include joins[] field when data spans multiple collections.
Do not add any text outside the JSON.

"""
#.format(schema_section=SCHEMA_SECTION, today_date=today_date)
# SYSTEM_PROMPT = SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")
# PROMPT_without_memory = ChatPromptTemplate.from_messages(
#     [("system", SYSTEM_PROMPT), ("system", MATCH_CONTEXT), ("placeholder", "{messages}")]
# )

# ────────────────────────────────
# 6. LLM & agent
# ────────────────────────────────
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

# def get_llm():
#     """
#     Create the LLM lazily and only if an API key is available.
#     Return None if missing so the caller can fail gracefully.
#     """
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         return None
#     # Newer langchain-openai uses `api_key` (not openai_api_key)
#     return ChatOpenAI(model=OPENAI_MODEL, api_key=api_key)

# Using get_memory_prompt from cache_memory.py instead of local implementation


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
def build_search_prompt(query: str, SYSTEM_PROMPT) -> list:
    """Build messages for LiteLLM completion"""
    
    # Build the system message
    system_content = f"""{SYSTEM_PROMPT}

    {CORE_RULES_TEXT}

    {USER_MATCH_CONTEXT}

"""
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query}
    ]
def run_search_agent(user_id: str, team_id: str,
    query: str,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    debug: bool = False,
    session_id: Optional[str] = None,
) -> tuple[dict, str, Dict[str, Any]]:        # spec, answer, dbg

    """
    1) Ask the LLM for a JSON spec
    2) Parse the JSON
    3) Figure out which tool to call
    4) Call it and pretty-print the results
    """

    global ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX
    global SEARCHABLE_FIELDS, CORE_RULES_TEXT, SCHEMA_SECTION

    # Get all fields and descriptions from the registry
    ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX = _load_schema_live()
    
    # Store original collection data to check for attempted access to restricted collections
    original_collections = set(ALL_FIELDS.keys())
    
    # Filter collections based on user access if a user_id is provided
    if user_id and user_id != "anonymous":
        # Get collections the user has access to
        user_collections = get_user_accessible_collections(user_id)
        
        # Filter ALL_FIELDS to only include collections the user has access to
        if user_collections:  # Only filter if user has specific permissions
            ALL_FIELDS = {coll: fields for coll, fields in ALL_FIELDS.items() 
                         if coll in user_collections}
            DESCRIPTIONS = {coll: desc for coll, desc in DESCRIPTIONS.items() 
                           if coll in user_collections}
            
            # Log filtered collections for debugging
            print(f"User {user_id} has access to collections: {user_collections}")
        else:
            print(f"User {user_id} has no collection permissions defined")
            # If user has no permissions but permissions are being enforced, they have no access
            if get_user_access_config():  # Only restrict if access control is active
                ALL_FIELDS = {}  # No collections available
                DESCRIPTIONS = {}
    else:
        print("No user_id provided or anonymous user - using all collections")
        
    # Store information about restricted collections
    restricted_collections = original_collections - set(ALL_FIELDS.keys())
    
    # Update searchable fields and collection names after filtering
    SEARCHABLE_FIELDS = ALL_FIELDS
    COLLECTION_NAMES = list(ALL_FIELDS.keys())
    
    # Generate schema section and core rules based on filtered collections
    user_schema_section = render_schema_section_with_relations(ALL_FIELDS, DESCRIPTIONS, OPTIONS_MAX)
    print(f"Updated SCHEMA_SECTION:\n{user_schema_section}\n")
    print(f"Restricted collections for user {user_id}: {restricted_collections}\n")
    CORE_RULES_TEXT = render_core_rules(COLLECTION_NAMES)

    if not ALL_FIELDS:
        answer = "⚠️ You do not have access to any collections or no collections are configured."
        return {}, answer, {"restricted_collections": list(restricted_collections)}
    

    # -------- always-defined placeholders --------
    debug_blob: Dict[str, Any] = {}
    coll_key:   str | None     = None
    mongo_filter: Dict[str, Any] = {}
    results:    list           = []
    invalid_fields: List[tuple] = []  # <-- collect invalid filter attempts


    #response = agent.invoke(
    #     {"messages":[{"role":"user","content":query}]},
    #     config={"recursion_limit": 10}
    # )
   

    # STEP 2: Create prompt with history and query
    memory_prompt = get_memory_prompt(3, user_id, team_id, session_id)
    user_system_prompt = SYSTEM_PROMPT.format(
        schema_section=user_schema_section, 
        today_date=today_date
    ).replace("<CONVERSATION_HISTORY>", memory_prompt)
    messages = build_search_prompt(query, user_system_prompt)
   
    response = call_search_agent_model(messages)
    print(f"[SEARCH_AGENT] Raw response: {response}")
    if isinstance(response, str):
        spec_str = response
    elif hasattr(response, 'choices') and len(response.choices) > 0:
        spec_str = response.choices[0].message.content.strip()
    else:
        raise ValueError(f"Unexpected response format: {type(response)}")

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

    # ai_msg   = response["messages"][-1]
    # spec_str = ai_msg.content.strip()
    # spec_str = response.content.strip()

    # 1) Parse the JSON

    print("\n[LLM-RAW]\n", spec_str)

    try:
        print(f"🔍 Raw LLM response content: {spec_str[:200]}...")
        raw_spec = _extract_json_spec(spec_str)
        print(f"✅ Successfully extracted JSON spec: {type(raw_spec)}")
        if isinstance(raw_spec, dict):
            print(f"   Available keys: {list(raw_spec.keys())}")
        print(f"   Full spec: {json.dumps(raw_spec, indent=2)}")
    except Exception as e:
        # show the entire returned text so you can see what the model sent
        answer = f"⚠️ JSON parse error:\n{e}\n```json\n{spec_str}\n```"
        debug_blob["spec"] = {}
        debug_blob["json_parse_error"] = str(e)
        debug_blob["raw_response"] = spec_str
        return {}, answer, debug_blob
    
    # Validate the query for common join mistakes
    try:
        from join_teacher import validate_query_joins
        is_valid, warnings, analysis = validate_query_joins(raw_spec)
        
        if not is_valid:
            debug_blob["join_validation"] = {
                "valid": is_valid,
                "warnings": warnings,
                "analysis": analysis
            }
            
            # Add warnings to debug but continue execution with warnings
            print(f"\n[JOIN VALIDATION] Query has potential issues:")
            for warning in warnings:
                print(f"  {warning}")
        
        if warnings:
            debug_blob["join_warnings"] = warnings
    
    except Exception as e:
        print(f"[JOIN VALIDATION] Warning: Could not validate joins: {e}")
        return {}, answer, debug_blob

    debug_blob["spec"] = raw_spec  # keep it for the UI
    qspec = raw_spec

    # Determine whether we have a single query or multiple queries
    if isinstance(raw_spec, dict) and "error" in raw_spec:
        answer = f"⚠️ LLM error: {raw_spec['error']}"
        return {}, answer, debug_blob
        
    # Determine whether we have a single query or multiple queries
    if isinstance(raw_spec, dict) and "queries" in raw_spec:
        try:
            multi = MultiEntityQuery(**raw_spec)
            query_specs = [q.dict() for q in multi.queries]
            spec = {"queries": query_specs}
        except Exception as e:
            answer = f"⚠️ Invalid multi-query format: {str(e)}\n```json\n{json.dumps(raw_spec, indent=2)}\n```"
            return {}, answer, debug_blob
    elif isinstance(raw_spec, list):
        try:
            multi = MultiEntityQuery(queries=[CollectionQuery(**q) for q in raw_spec])
            query_specs = [q.dict() for q in multi.queries]
            spec = {"queries": query_specs}
        except Exception as e:
            answer = f"⚠️ Invalid query list format: {str(e)}\n```json\n{json.dumps(raw_spec, indent=2)}\n```"
            return {}, answer, debug_blob
    else:
        # Check if required fields are present before creating CollectionQuery
        if not isinstance(raw_spec, dict):
            answer = f"⚠️ Expected dict but got {type(raw_spec)}: {raw_spec}"
            debug_blob["invalid_spec_type"] = type(raw_spec).__name__
            return {}, answer, debug_blob
            
        # Log the parsed spec for debugging
        print(f"🔍 Parsed raw_spec type: {type(raw_spec)}")
        print(f"🔍 Parsed raw_spec content: {json.dumps(raw_spec, indent=2, default=str)}")
            
        if "collection" not in raw_spec:
            available_keys = list(raw_spec.keys()) if isinstance(raw_spec, dict) else "Not a dict"
            answer = f"⚠️ Missing required field 'collection' in spec. Available keys: {available_keys}\n```json\n{json.dumps(raw_spec, indent=2, default=str)}\n```"
            debug_blob["missing_collection_field"] = True
            debug_blob["available_keys"] = available_keys
            return {}, answer, debug_blob
                
        try:
            print(f"🔍 Creating CollectionQuery from: {json.dumps(raw_spec, indent=2, default=str)}")
            
            # Ensure filters field exists (even if empty)
            if "filters" not in raw_spec:
                print("⚠️ No 'filters' field found, adding empty filters array")
                raw_spec["filters"] = []
            
            single = CollectionQuery(**raw_spec)
            spec = single.dict()
            query_specs = [spec]
            
            print(f"✅ Successfully created CollectionQuery: {json.dumps(spec, indent=2, default=str)}")
            
        except Exception as e:
            print(f"❌ Failed to create CollectionQuery: {e}")
            print(f"   Raw spec: {json.dumps(raw_spec, indent=2, default=str)}")
            answer = f"⚠️ Invalid single-query format: {str(e)}\n```json\n{json.dumps(raw_spec, indent=2, default=str)}\n```"
            debug_blob["collection_query_error"] = str(e)
            debug_blob["raw_spec_at_error"] = raw_spec
            return {}, answer, debug_blob

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
            # Check if the collection exists but the user doesn't have access
            if coll_name in restricted_collections:
                answer = f"⚠️ Access denied: You do not have permission to access the '{coll_name}' collection."
            else:
                answer = f"⚠️ Unknown collection: {coll_name}"
            return spec, answer, debug_blob

        res = _run_query(coll_name, qspec)
        print(f"\n[DEBUG] Result for {coll_name}:", json.dumps(res, indent=2, default=str))
        filters_debug.append(res.get("filter", {}))
        # 3) Handle errors without breaking out
        if not res.get("ok"):
            # Still add the collection to chosen_collections even if no results
            chosen_collections.append(coll_name)
            
            debug_blob.setdefault("errors", []).append({
                "collection": coll_name,
                "error":      res.get("error"),
                "filter":     res.get("filter", {})
            })
            continue

        # 5) Accumulate successful results
        chosen_collections.append(coll_name)
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

    # Add information about restricted collections to debug_blob
    debug_blob.update({
        "filters": filters_debug,
        "chosen_collections": chosen_collections,
        "results": results_debug,
        "restricted_collections": list(restricted_collections) if restricted_collections else [],
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
        spec, answer, dbg = run_search_agent("test_user", "test_team", q)

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