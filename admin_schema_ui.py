# admin_schema_ui.py
import os, json, pandas as pd, streamlit as st
from datetime import date, datetime, time
from pymongo import MongoClient
from schema_registry import (
    load_registry, save_registry, list_collections, upsert_collection, delete_collection,
    set_options_max,
    get_connection_config, set_connection_config,
)
from core_rules import render_schema_section_all

import re
import pandas as pd
import numpy as np
import json
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

def _to_safe_text(x):
    """Coerce any value to a displayable string without crashing on bytes/NaN."""
    # Treat missing values early
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass

    # Bytes → try utf-8, then cp1252, then latin-1, then replacement
    if isinstance(x, (bytes, bytearray)):
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return x.decode(enc)
            except UnicodeDecodeError:
                continue
        return x.decode("utf-8", "replace")

    # Pandas/NumPy types
    if isinstance(x, (np.integer,)):
        return str(int(x))
    if isinstance(x, (np.floating,)):
        f = float(x)
        if np.isnan(f) or np.isinf(f):
            return ""
        return str(f)
    if isinstance(x, (datetime, date, time, pd.Timestamp)):
        return x.isoformat()

    # Fallback
    return str(x)

def _json_default(o):
    # pandas / numpy / datetimes → JSON-safe
    if isinstance(o, (datetime, date, time)):
        return o.isoformat()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    if isinstance(o, np.ndarray):
        return o.tolist()
    # pandas NA / NaT etc.
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass
    # last resort
    return str(o)

load_dotenv()  # loads OPENAI_API_KEY from .env if present

def get_llm():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=key)

def generate_ai_descriptions(df: pd.DataFrame, sample_size: int = 3) -> dict:
    """Ask AI to describe each column based on a sample of the data."""
    llm = get_llm()

    # ---- 1) Take sample and convert all values to JSON-safe types ----
    def safe_value(val):
        if isinstance(val, (pd.Timestamp, np.datetime64)):
            return str(val)
        if isinstance(val, (list, tuple, np.ndarray)):
            return [safe_value(v) for v in val]
        try:
            if pd.isna(val):
                return None
        except Exception:
            pass
        return val

    # Build a small, JSON-safe sample for the prompt
    n = min(sample_size, len(df)) if len(df) else 0
    sample_df = df.sample(n, random_state=0) if n > 0 else df.head(0)
    sample_records = sample_df.to_dict(orient="records")
    sample_json = json.dumps(sample_records, default=_json_default, indent=2)

    # ---- 2) Build strict JSON-only prompt ----
    prompt = f"""
You are helping define a database schema for a **generic analytics system**.

Here is a sample of the dataset:
{sample_json}

For each column in the dataset, provide a short, precise description.

OUTPUT INSTRUCTIONS:
- Respond ONLY with valid JSON.
- JSON format: {{ "column_name": "description string" }}
- No extra commentary, no markdown, no explanation.
"""

    # ---- 3) Call LLM and log raw output ----
    resp = llm.invoke(prompt)
    print("AI RAW OUTPUT:", repr(resp.content))

    # ---- 4) Parse JSON safely ----
    try:
        return json.loads(resp.content)
    except Exception:
        # Try extracting JSON substring
        match = re.search(r"\{.*\}", resp.content, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        print("⚠️ Failed to parse AI description output. Returning empty descriptions.")
        return {}




def heuristic_schema(df: pd.DataFrame, options_max: int = 20, max_opt_len: int = 50) -> list[dict]:
    """Dynamic schema generation with AI descriptions and intelligent option inclusion."""
    TYPE_OPS = {
        "string": ["regex", "sort"],
        "int":    ["range", "sort"],
        "float":  ["range", "sort"],
        "date":   ["range", "sort"],
        "array":  ["keyword"]
    }

    def infer_type(s: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(s): return "int"
        if pd.api.types.is_float_dtype(s):   return "float"
        try:
            pd.to_datetime(s.dropna().head(50), errors="raise")
            return "date"
        except Exception:
            pass
        if s.dropna().map(lambda x: isinstance(x, (list, tuple))).any():
            return "array"
        return "string"

    # 1) Get AI descriptions for each column
    col_descriptions = generate_ai_descriptions(df)
    print("AI Descriptions:", col_descriptions)

    # 2) Build field list with intelligent options
    fields = []
    for col in df.columns:
        t = infer_type(df[col])
        ops = TYPE_OPS[t]
        opts = []

        if t in ("string", "bool"):  # candidates for categorical
            s = df[col].dropna().map(_to_safe_text).str.strip()
            uniq = s[s != ""].unique()
            uniq = [u for u in uniq if len(u) <= max_opt_len]
            if 1 < len(uniq) <= options_max:
                opts = sorted(map(str, uniq))

        fields.append({
            "name": col,
            "type": t,
            "operations": ops,
            "description": col_descriptions.get(col, ""),
            "options": opts
        })
    return fields



def get_mongo_collections(uri, db_name):
    """Fetch all collection names from the specified MongoDB database."""
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            uuidRepresentation="standard",
        )
        client.admin.command("ping")  # Verify connection
        
        # Get all collection names
        collections = client[db_name].list_collection_names()
        return collections, None
    except Exception as e:
        return [], f"Error connecting to MongoDB: {e}"


def fetch_collection_sample(uri, db_name, collection_name, limit=100):
    """Fetch a sample of documents from the specified collection."""
    try:
        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
            uuidRepresentation="standard",
        )
        client.admin.command("ping")
        pipe = [
            {"$match": {}},
            {"$sample": {"size": int(limit)}},
            {"$project": {"_id": 0}},
        ]
        df = pd.DataFrame(list(client[db_name][collection_name].aggregate(pipe)))
        return df, None
    except Exception as e:
        return None, f"Error fetching data: {e}"


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
if ADMIN_TOKEN:
    if st.text_input("Admin token", type="password") != ADMIN_TOKEN:
        st.stop()

# ─────────────────────────────────────────────────────────────
# Connection (MongoDB) — saved in schema_registry.json
# ─────────────────────────────────────────────────────────────
st.subheader("Database Connection")

cfg = get_connection_config(load_registry())
default_uri = cfg.get("mongo_uri", "mongodb://127.0.0.1:27017")
default_db  = cfg.get("mongo_db",  "test")

c1, c2 = st.columns([3, 1])
with c1:
    uri_input = st.text_input("Mongo URI", value=default_uri, placeholder="mongodb://host:27017")
    db_input  = st.text_input("Database Name", value=default_db, placeholder="sample_mflix")
with c2:
    st.markdown(" ")
    st.markdown(" ")
    if st.button("Save Connection", type="primary"):
        if not uri_input.strip() or not db_input.strip():
            st.error("Please provide both Mongo URI and Database Name.")
        else:
            set_connection_config(uri_input, db_input)
            st.success(f"Saved! mongo_db = **{db_input.strip()}**")
            st.toast("Connection settings updated.", icon="✅")
            
            # Store connection in session state for later use
            st.session_state.mongo_uri = uri_input
            st.session_state.mongo_db = db_input
            
            # Automatically fetch collection names
            collections, error = get_mongo_collections(uri_input, db_input)
            if error:
                st.error(error)
            else:
                st.session_state.available_collections = collections
                st.success(f"Found {len(collections)} collections in database {db_input}")

# Show the effective connection (for sanity)
st.code(json.dumps(get_connection_config(load_registry()), indent=2))
st.divider()


st.title("Admin • Collections & Schemas")
tab_schema, tab_rules, tab_access = st.tabs(["Collections & Schemas", "Core Rules & Prompt", "User Access Control"])

with tab_schema:

    # Keep sample & schema in state so clicks survive reruns
    if "sample_df" not in st.session_state:
        st.session_state.sample_df = None
    if "schema_fields" not in st.session_state:
        st.session_state.schema_fields = []

    reg = load_registry()

    uri = st.text_input(
        "Mongo URI",
        value=os.getenv("MONGO_URI",""),
        help="Use 127.0.0.1 to force IPv4. If this UI runs in Docker, try host.docker.internal (Mac/Win) or 172.17.0.1 (Linux)."
    )

    # global options cap control
    cap_val = st.number_input(
        "Options cap (used when rendering prompt)",
        min_value=1,
        value=int(reg.get("options_max", 20)),
        help="Categories/options per field will be clipped to this number in prompt bullets."
    )
    if st.button("Save options cap"):
        set_options_max(cap_val)
        st.success("Saved options cap")

    st.header("Add / Edit Collection")
    coll_name = st.text_input("Collection name (exact Mongo name)", "")
    coll_desc = st.text_input("Description (short)","")

    st.subheader("Sample Data")
    source = st.radio("Source", ["Mongo", "CSV", "JSON"], horizontal=True)
    df = None

    if source == "Mongo":
        db  = st.text_input("Database", os.getenv("MONGO_DB", ""))
        lim = st.slider("Rows", 10, 500, 100)

        if st.button("Fetch"):
            try:
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000,
                    uuidRepresentation="standard",
                )
                client.admin.command("ping")
                pipe = [
                    {"$match": {}},
                    {"$sample": {"size": int(lim)}},
                    {"$project": {"_id": 0}},
                ]
                df = pd.DataFrame(list(client[db][coll_name].aggregate(pipe)))
                st.session_state.sample_df = df
                st.success(f"Loaded {len(df)} rows")
                st.dataframe(df.head(50), use_container_width=True)
            except Exception as e:
                st.error(f"Mongo connection/sample error: {e}")

    elif source == "CSV":
        up = st.file_uploader("CSV file", type=["csv"])
        if up:
            df = pd.read_csv(up)
            st.session_state.sample_df = df
            st.dataframe(df.head(50), use_container_width=True)

    else:  # JSON
        up = st.file_uploader("JSON (records)", type=["json"])
        if up:
            recs = json.load(up)
            df = pd.DataFrame(recs)
            st.session_state.sample_df = df
            st.dataframe(df.head(50), use_container_width=True)


    st.subheader("Schema")
    TYPE_OPS = {"string":["regex","sort"], "int":["range","sort"], "float":["range","sort"], "date":["range","sort"], "array":["keyword"]}

    def infer_type(s: pd.Series) -> str:
        import pandas as pd
        if pd.api.types.is_integer_dtype(s): return "int"
        if pd.api.types.is_float_dtype(s):   return "float"
        try:
            pd.to_datetime(s.dropna().head(50), errors="raise"); return "date"
        except: pass
        if s.dropna().map(lambda x: isinstance(x,(list,tuple))).any(): return "array"
        return "string"

    fields = []
    generate = st.button("Generate schema (heuristic)")
    upload   = st.file_uploader("…or upload schema JSON", type=["json"], key="schemajson")

    cap = load_registry().get("options_max", 20)

    if generate:
        df0 = st.session_state.sample_df
        if df0 is None or df0.empty:
            st.error("Load sample data first.")
        else:
            fields = heuristic_schema(df0, options_max=load_registry().get("options_max", 20))
            st.session_state.schema_fields = fields
            st.success("Schema generated with AI descriptions and intelligent options.")


    # if generate:
    #     df0 = st.session_state.sample_df
    #     if df0 is None or df0.empty:
    #         st.error("Load sample data first (Mongo/CSV/JSON) before generating schema.")
    #     else:
    #         fields = []
    #         for col in df0.columns:
    #             t = infer_type(df0[col])
    #             ops = TYPE_OPS[t]
    #             opts = []
    #             if t == "string":
    #                 uniq = df0[col].dropna().astype(str).str.strip().unique()
    #                 opts = sorted(map(str, uniq))[:cap]  # IMPORTANT: cap to admin setting
    #             fields.append({
    #                 "name": col,
    #                 "type": t,
    #                 "operations": ops,
    #                 "description": "",
    #                 "options": opts
    #             })
    #         st.session_state.schema_fields = fields

    if upload:
        try:
            st.session_state.schema_fields = json.load(upload).get("fields", [])
            st.info(f"Loaded {len(st.session_state['schema_fields'])} fields from JSON.")
        except Exception as e:
            st.error(f"Invalid schema JSON: {e}")


    if st.session_state.schema_fields:
        st.subheader("Edit Generated Schema")

        sch = pd.DataFrame(st.session_state.schema_fields)

        # Convert options and operations to comma-separated strings for editing
        sch["options"] = sch["options"].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else "")
        sch["operations"] = sch["operations"].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else "")

        # Editable type dropdown
        sch["type"] = sch["type"].astype("category").cat.set_categories(list(TYPE_OPS.keys()))

        edited = st.data_editor(
            sch,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Field Name", required=True),
                "type": st.column_config.SelectboxColumn("Type", options=list(TYPE_OPS.keys())),
                "operations": st.column_config.TextColumn("Operations (comma-separated)"),
                "description": st.column_config.TextColumn("Description"),
                "options": st.column_config.TextColumn("Options (comma-separated)")
            },
            hide_index=True
        )

        # Convert edited table back into correct structure
        def parse_list(val):
            if not val or pd.isna(val):
                return []
            return [v.strip() for v in str(val).split(",") if v.strip()]

        st.session_state.schema_fields = [
            {
                "name": row["name"],
                "type": row["type"],
                "operations": parse_list(row["operations"]),
                "description": row["description"],
                "options": parse_list(row["options"])
            }
            for _, row in edited.iterrows()
        ]

        if st.button("Save collection"):
            if not coll_name.strip():
                st.warning("Please enter a collection name.")
            else:
                reg = load_registry()
                reg["collections"][coll_name] = {
                    "description": coll_desc,
                    "fields": st.session_state.schema_fields
                }
                save_registry(reg)
                st.success(f"Saved schema for {coll_name}")


    st.header("Existing Collections")
    colls = list_collections()
    st.write({k: {"fields": len(v.get("fields",[]))} for k,v in colls.items()})
    del_name = st.selectbox("Delete collection", ["(none)"] + list(colls.keys()))
    if del_name != "(none)" and st.button("Delete"):
        delete_collection(del_name); st.success("Deleted")

    from schema_registry import get_all_fields, get_descriptions
    reg = load_registry()
    preview = render_schema_section_all(
        get_all_fields(reg), 
        get_descriptions(reg), 
        reg.get("options_max", 20)
    )
    st.code(preview)



from schema_registry import (
    get_core_rules_config, set_core_rules_config,
    get_all_fields, get_descriptions, get_collection_names,
    get_user_match_context, set_user_match_context, load_registry
)
from core_rules import render_core_rules, render_schema_section_all

with tab_rules:
    st.subheader("Core Rules (Admin-editable)")

    # 1) Mode + editor
    cfg = get_core_rules_config()
    mode = st.radio("Mode", ["auto", "append", "override"], index=["auto","append","override"].index(cfg["mode"]),
                    help="auto: generated from current collections; append: add your text after auto; override: use only your text")
    custom = st.text_area("Custom rules (Markdown or plain text)",
                          value=cfg["custom_text"], height=180,
                          placeholder="Write additional/override rules here…")

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("Save Core Rules"):
            set_core_rules_config(mode, custom)
            st.success("Core rules saved.")
    with colB:
        st.caption("These settings apply instantly to new queries.")

    # 2) Optional: admin “extra add-up” match context
    st.divider()
    st.subheader("Optional: Extra Context")
    cur_ctx = get_user_match_context()
    new_ctx = st.text_area("Extra context (shown as a separate system message)", value=cur_ctx, height=140)
    if st.button("Save Extra Context"):
        set_user_match_context(new_ctx)
        st.success("Extra context saved.")

    # 3) Effective rules preview
    st.divider()
    st.subheader("Effective Rules Preview")
    reg = load_registry()
    coll_names = get_collection_names(reg)
    auto_rules = render_core_rules(coll_names)
    if mode == "override" and custom.strip():
        effective = custom.strip()
    elif mode == "append" and custom.strip():
        effective = (auto_rules + "\n\n" + custom.strip()).strip()
    else:
        effective = auto_rules
    st.code(effective)

    # 4) Full Search-Agent prompt preview (what’s actually sent)
    st.divider()
    st.subheader("Search Agent – Prompt Inspector")

    # schema bullets (all collections, with cap)
    all_fields = get_all_fields(reg)
    descs      = get_descriptions(reg)
    cap        = reg.get("options_max", 20)
    schema_section = render_schema_section_all(all_fields, descs, cap)

    # base system prompt template (import from your agent if available)
    try:
        import importlib, search_agent_new as SA
        importlib.reload(SA)  # pick up latest registry
        base_system = SA.SYSTEM_PROMPT
        match_ctx   = SA.MATCH_CONTEXT
    except Exception:
        base_system = "You are an expert MongoDB query planner for the following collections:\n\n" + schema_section
        match_ctx   = ""

    # Compose the exact messages we send at runtime
    prompt_preview = "\n\n".join([
        "【SYSTEM #1】Search SYSTEM_PROMPT\n" + base_system,
        "【SYSTEM #2】CORE_RULES_TEXT (effective)\n" + effective,
        "【SYSTEM #3】MATCH_CONTEXT (admin extra)\n" + (new_ctx or cur_ctx or "(empty)"),
        "【SYSTEM #4】MEMORY_PROMPT\n(constructed at runtime; recent Q/A, pronoun rules)",
        "【HUMAN】{query}"
    ])
    st.code(prompt_preview, language="markdown")

    st.info("Tip: In the Chat UI, open “Prompt sent to LLM” to see the **response-gen** prompt for a given question.")

# Import user access functions
from schema_registry import (
    get_user_access_config, set_user_access, delete_user_access,
    get_user_collections, get_all_users
)

with tab_access:
    st.subheader("User Access Control")
    st.write("Manage which collections each user can access.")
    
    # Get current registry and available collections
    reg = load_registry()
    all_collections = list(reg.get("collections", {}).keys())
    
    # Display no collections warning if needed
    if not all_collections:
        st.warning("No collections available. Add collections in the 'Collections & Schemas' tab first.")
    
    # Get current user access configuration
    user_access_config = get_user_access_config()
    
    # Section for adding/editing user access
    st.markdown("### Add or Edit User Access")
    
    # User selection or creation
    col1, col2 = st.columns([3, 1])
    with col1:
        # Get existing users plus option for new user
        existing_users = get_all_users()
        user_options = ["Add New User"] + existing_users
        selected_user_option = st.selectbox("Select User", user_options)
        
        if selected_user_option == "Add New User":
            # New user input
            user_id = st.text_input("New User ID", placeholder="Enter user ID or name")
            is_new_user = True
        else:
            # Existing user
            user_id = selected_user_option
            is_new_user = False
    
    # Only show the rest if collections exist
    if all_collections:
        # Collection access selection
        st.markdown("#### Select Collections")
        st.write("Choose which collections this user can access:")
        
        # Get current access for this user
        user_collections = get_user_collections(user_id) if not is_new_user else []
        
        # Create checkboxes for each collection
        selected_collections = []
        for collection in all_collections:
            is_selected = collection in user_collections
            if st.checkbox(collection, value=is_selected, key=f"access_{user_id}_{collection}"):
                selected_collections.append(collection)
        
        # Save button
        if st.button("Save User Access", type="primary"):
            if user_id:
                set_user_access(user_id, selected_collections)
                st.success(f"Access settings saved for user '{user_id}'")
                st.toast(f"User '{user_id}' now has access to {len(selected_collections)} collections", icon="✅")
            else:
                st.error("Please enter a User ID")
        
        # Current access summary
        st.divider()
        st.markdown("### Current User Access")
        
        user_access_df = []
        for u_id, collections in user_access_config.items():
            user_access_df.append({
                "User ID": u_id,
                "Collections": ", ".join(collections) if collections else "None",
                "Count": len(collections)
            })
        
        if user_access_df:
            st.dataframe(
                pd.DataFrame(user_access_df),
                use_container_width=True,
                column_config={
                    "User ID": st.column_config.TextColumn("User ID"),
                    "Collections": st.column_config.TextColumn("Accessible Collections"),
                    "Count": st.column_config.NumberColumn("Count")
                }
            )
        else:
            st.info("No user access settings defined yet.")
        
        # Delete user section
        st.divider()
        st.markdown("### Delete User Access")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            user_to_delete = st.selectbox(
                "Select User to Delete",
                [""] + existing_users,
                index=0,
                placeholder="Select a user"
            )
        
        with col2:
            st.markdown("&nbsp;")  # Spacer
            st.markdown("&nbsp;")  # Spacer
            if user_to_delete and st.button("Delete User", type="secondary"):
                delete_user_access(user_to_delete)
                st.success(f"Access settings deleted for user '{user_to_delete}'")
                st.toast(f"User '{user_to_delete}' removed", icon="🗑️")
                st.rerun()
    else:
        st.info("Please add collections in the 'Collections & Schemas' tab before setting up user access.")


