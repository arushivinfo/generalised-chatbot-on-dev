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

# Configure Streamlit for full width
st.set_page_config(
    page_title="Admin Schema UI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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




# Initialize session state for connection
if "connection_established" not in st.session_state:
    st.session_state.connection_established = False



# ─────────────────────────────────────────────────────────────
# Centralized Connection Management
# ─────────────────────────────────────────────────────────────
if not st.session_state.connection_established:
    st.title("🔧 Admin Schema Management")
    st.subheader("Database Connection Setup")
    
    cfg = get_connection_config(load_registry())
    default_uri = cfg.get("mongo_uri", "mongodb://127.0.0.1:27017")
    default_db = cfg.get("mongo_db", "test")
    
    with st.container():
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            uri_input = st.text_input(
                "MongoDB URI", 
                value=default_uri, 
                placeholder="mongodb://host:27017",
                help="Use 127.0.0.1 to force IPv4. If this UI runs in Docker, try host.docker.internal (Mac/Win) or 172.17.0.1 (Linux)."
            )
        
        with col2:
            db_input = st.text_input(
                "Database Name", 
                value=default_db, 
                placeholder="sample_mflix"
            )
        
        with col3:
            st.markdown("&nbsp;")  # Spacer
            if st.button("🔗 Connect & Continue", type="primary", use_container_width=True):
                if not uri_input.strip() or not db_input.strip():
                    st.error("Please provide both MongoDB URI and Database Name.")
                else:
                    # Test connection
                    collections, error = get_mongo_collections(uri_input.strip(), db_input.strip())
                    if error:
                        st.error(f"Connection failed: {error}")
                    else:
                        # Save connection and proceed
                        set_connection_config(uri_input.strip(), db_input.strip())
                        st.session_state.mongo_uri = uri_input.strip()
                        st.session_state.mongo_db = db_input.strip()
                        st.session_state.available_collections = collections
                        st.session_state.connection_established = True
                        st.success(f"✅ Connected! Found {len(collections)} collections in database '{db_input.strip()}'")
                        st.rerun()

    # Show current saved connection for reference
    current_config = get_connection_config(load_registry())
    if current_config.get("mongo_uri"):
        st.info(f"💾 Saved connection: {current_config['mongo_db']} @ {current_config['mongo_uri']}")
    
    st.stop()  # Don't show the rest until connected

# Show connection status in sidebar
with st.sidebar:
    st.success(f"🟢 Connected to: **{st.session_state.mongo_db}**")
    st.caption(f"URI: {st.session_state.mongo_uri}")
    if st.button("🔄 Change Connection"):
        st.session_state.connection_established = False
        st.rerun()
    
    # Show available collections
    if "available_collections" in st.session_state:
        st.subheader("Available Collections")
        for coll in st.session_state.available_collections[:10]:  # Show first 10
            st.caption(f"📄 {coll}")
        if len(st.session_state.available_collections) > 10:
            st.caption(f"... and {len(st.session_state.available_collections) - 10} more")

# ─────────────────────────────────────────────────────────────
# Main Application (Full Width)
# ─────────────────────────────────────────────────────────────
st.title("🔧 Admin • Collections & Schemas")

# Keep sample & schema in state so clicks survive reruns
if "sample_df" not in st.session_state:
    st.session_state.sample_df = None
if "schema_fields" not in st.session_state:
    st.session_state.schema_fields = []

tab_schema, tab_rules, tab_access, tab_rls, tab_eval = st.tabs(["📊 Collections & Schemas", "📝 Core Rules & Prompt", "👥 User Access Control", "🔒 Row-Level Security", "🔍 Batch Evaluations"])

with tab_schema:
    # Global options cap control (moved to top for better UX)
    with st.expander("⚙️ Global Settings", expanded=False):
        reg = load_registry()
        cap_val = st.number_input(
            "Options cap (used when rendering prompt)",
            min_value=1,
            value=int(reg.get("options_max", 20)),
            help="Categories/options per field will be clipped to this number in prompt bullets."
        )
        if st.button("💾 Save Options Cap"):
            set_options_max(cap_val)
            st.success("✅ Saved options cap")

    # Collection management in two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Add / Edit Collection")
        
        # Collection selector for editing existing
        existing_colls = list(list_collections().keys())
        edit_mode = st.selectbox("Mode", ["Create New", "Edit Existing"], key="collection_mode")
        
        if edit_mode == "Edit Existing" and existing_colls:
            selected_coll = st.selectbox("Select Collection to Edit", existing_colls)
            # Pre-fill with existing data
            existing_data = list_collections()[selected_coll]
            coll_name = selected_coll
            coll_desc = st.text_input("Description (short)", value=existing_data.get("description", ""))
            # Load existing schema
            st.session_state.schema_fields = existing_data.get("fields", [])
        else:
            # Collection name input with dropdown for available collections
            available_collections = st.session_state.get("available_collections", [])
            
            # Create options: manual input + available collections
            collection_options = ["Type manually..."] + available_collections
            
            selected_option = st.selectbox(
                "Collection Name", 
                options=collection_options,
                help="Select from available MongoDB collections or type manually"
            )
            
            if selected_option == "Type manually...":
                coll_name = st.text_input(
                    "Enter collection name", 
                    placeholder="Enter exact MongoDB collection name"
                )
            else:
                coll_name = selected_option
                st.info(f"Selected collection: **{coll_name}**")
            
            coll_desc = st.text_input("Description (short)", "")
        
        st.subheader("📊 Sample Data")
        source = st.radio("Data Source", ["MongoDB", "CSV Upload", "JSON Upload"], horizontal=True)
        df = None

        if source == "MongoDB":
            # Show current collection being used
            if coll_name:
                st.info(f"Will fetch data from: **{coll_name}**")
            
            lim = st.slider("Sample Size", 10, 500, 100)

            if st.button("🔄 Fetch Sample Data", type="primary"):
                if not coll_name:
                    st.error("Please select or enter a collection name")
                else:
                    df, error = fetch_collection_sample(
                        st.session_state.mongo_uri, 
                        st.session_state.mongo_db, 
                        coll_name, 
                        lim
                    )
                    if error:
                        st.error(error)
                    else:
                        st.session_state.sample_df = df
                        st.success(f"✅ Loaded {len(df)} rows from {coll_name}")

        elif source == "CSV Upload":
            up = st.file_uploader("📁 Upload CSV file", type=["csv"])
            if up:
                df = pd.read_csv(up)
                st.session_state.sample_df = df
                st.success(f"✅ Loaded CSV with {len(df)} rows")

        else:  # JSON Upload
            up = st.file_uploader("📁 Upload JSON file (records format)", type=["json"])
            if up:
                recs = json.load(up)
                df = pd.DataFrame(recs)
                st.session_state.sample_df = df
                st.success(f"✅ Loaded JSON with {len(df)} rows")

    with col2:
        st.header("🧠 Schema Generation")
        
        # Schema generation options
        col_a, col_b = st.columns([1, 1])
        with col_a:
            generate = st.button("🤖 Generate AI Schema", type="primary", use_container_width=True)
        with col_b:
            upload = st.file_uploader("📤 Upload Schema JSON", type=["json"], key="schemajson")

        if generate:
            df0 = st.session_state.sample_df
            if df0 is None or df0.empty:
                st.error("⚠️ Load sample data first.")
            else:
                with st.spinner("🤖 Generating schema with AI descriptions..."):
                    fields = heuristic_schema(df0, options_max=load_registry().get("options_max", 20))
                    st.session_state.schema_fields = fields
                    st.success(f"✅ Schema generated with {len(fields)} fields!")

        if upload:
            try:
                st.session_state.schema_fields = json.load(upload).get("fields", [])
                st.info(f"📥 Loaded {len(st.session_state['schema_fields'])} fields from JSON.")
            except Exception as e:
                st.error(f"❌ Invalid schema JSON: {e}")

    # Show sample data if available (full width)
    if st.session_state.sample_df is not None:
        st.subheader("📋 Sample Data Preview")
        st.dataframe(st.session_state.sample_df.head(50), use_container_width=True, height=300)

    # Schema editor (full width)
    if st.session_state.schema_fields:
        st.subheader("✏️ Edit Schema")

        sch = pd.DataFrame(st.session_state.schema_fields)

        # Convert options and operations to comma-separated strings for editing
        sch["options"] = sch["options"].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else "")
        sch["operations"] = sch["operations"].apply(lambda x: ", ".join(map(str, x)) if isinstance(x, list) else "")

        TYPE_OPS = {"string":["regex","sort"], "int":["range","sort"], "float":["range","sort"], "date":["range","sort"], "array":["keyword"]}
        sch["type"] = sch["type"].astype("category").cat.set_categories(list(TYPE_OPS.keys()))

        edited = st.data_editor(
            sch,
            num_rows="dynamic",
            use_container_width=True,
            height=400,
            column_config={
                "name": st.column_config.TextColumn("Field Name", required=True, width="medium"),
                "type": st.column_config.SelectboxColumn("Type", options=list(TYPE_OPS.keys()), width="small"),
                "operations": st.column_config.TextColumn("Operations", width="medium"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "options": st.column_config.TextColumn("Options", width="large")
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

        # Save button (prominent)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 Save Collection Schema", type="primary", use_container_width=True):
                if not coll_name.strip():
                    st.error("⚠️ Please enter a collection name.")
                else:
                    reg = load_registry()
                    reg["collections"][coll_name] = {
                        "description": coll_desc,
                        "fields": st.session_state.schema_fields
                    }
                    save_registry(reg)
                    st.success(f"✅ Saved schema for **{coll_name}** with {len(st.session_state.schema_fields)} fields!")
                    st.balloons()

    # Existing collections management (full width)
    st.divider()
    st.header("📚 Existing Collections")
    
    colls = list_collections()
    if colls:
        # Create a nice table view
        coll_data = []
        for k, v in colls.items():
            coll_data.append({
                "Collection": k,
                "Description": v.get("description", ""),
                "Fields": len(v.get("fields", [])),
                "Last Modified": "N/A"  # Could add timestamp if needed
            })
        
        st.dataframe(
            pd.DataFrame(coll_data),
            use_container_width=True,
            column_config={
                "Collection": st.column_config.TextColumn("Collection Name", width="medium"),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Fields": st.column_config.NumberColumn("Field Count", width="small"),
                "Last Modified": st.column_config.TextColumn("Last Modified", width="medium")
            }
        )
        
        # Delete collection
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            del_name = st.selectbox("🗑️ Delete Collection", ["(none)"] + list(colls.keys()))
            if del_name != "(none)" and st.button("🗑️ Delete", type="secondary"):
                delete_collection(del_name)
                st.success(f"🗑️ Deleted {del_name}")
                st.rerun()
    else:
        st.info("📝 No collections defined yet. Create your first collection above!")

    # Schema preview (full width)
    st.divider()
    st.subheader("👁️ Schema Preview (as seen by AI)")
    
    from schema_registry import get_all_fields, get_descriptions
    reg = load_registry()
    preview = render_schema_section_all(
        get_all_fields(reg), 
        get_descriptions(reg), 
        reg.get("options_max", 20)
    )
    st.code(preview, language="markdown")

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
            st.info("New users are granted access to all collections by default. You can adjust access as needed.")
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
        user_collections = get_user_collections(user_id) if not is_new_user else all_collections  # Default all collections for new users
        
        # Add select/deselect all buttons
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Select All"):
                st.session_state.select_all_collections = True
                st.session_state.deselect_all_collections = False
                st.rerun()
            if st.button("Deselect All"):
                st.session_state.select_all_collections = False
                st.session_state.deselect_all_collections = True
                st.rerun()
        
        # Initialize session state for select/deselect all
        if 'select_all_collections' not in st.session_state:
            st.session_state.select_all_collections = False
        if 'deselect_all_collections' not in st.session_state:
            st.session_state.deselect_all_collections = False
            
        # Create checkboxes for each collection
        selected_collections = []
        for collection in all_collections:
            # Handle select all/deselect all
            if st.session_state.select_all_collections:
                is_selected = True
            elif st.session_state.deselect_all_collections:
                is_selected = False
            else:
                is_selected = collection in user_collections
                
            if st.checkbox(collection, value=is_selected, key=f"access_{user_id}_{collection}"):
                selected_collections.append(collection)
        
        # Reset select/deselect flags after they've been applied
        if st.session_state.select_all_collections or st.session_state.deselect_all_collections:
            st.session_state.select_all_collections = False
            st.session_state.deselect_all_collections = False
        
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

# Import RLS functions
from schema_registry import (
    get_rls_config, set_rls_config, set_rls_collection_field, 
    get_rls_collection_field, delete_rls_collection_config
)
from rls_engine import RLSFieldDetector, create_rls_interceptor

with tab_rls:
    st.subheader("🔒 Row-Level Security (RLS) Configuration")
    st.write("Configure automatic query filtering to ensure users only see data they own.")
    
    # Get current RLS configuration
    rls_config = get_rls_config()
    
    # Global RLS Settings
    st.markdown("### Global RLS Settings")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        rls_enabled = st.checkbox(
            "Enable Row-Level Security",
            value=rls_config.get("enabled", False),
            help="When enabled, all queries will be automatically filtered by user ownership"
        )
        
        default_user_field = st.text_input(
            "Default User Field",
            value=rls_config.get("default_user_field", "user_id"),
            help="Default field name used to identify row ownership across collections"
        )
        
        enforcement_mode = st.selectbox(
            "Enforcement Mode",
            options=["base_only", "all_involved"],
            index=0 if rls_config.get("enforcement_mode", "base_only") == "base_only" else 1,
            help="base_only: Filter only primary collection; all_involved: Filter all collections in joins"
        )
    
    with col2:
        audit_enabled = st.checkbox(
            "Enable Audit Logging",
            value=rls_config.get("audit_enabled", True),
            help="Log all RLS decisions and filter applications for security audit"
        )
        
        # Bypass roles
        bypass_roles_text = st.text_area(
            "Bypass Roles (one per line)",
            value="\n".join(rls_config.get("bypass_roles", ["admin", "super_user"])),
            height=100,
            help="Users with these roles will bypass RLS filtering"
        )
        
        bypass_roles = [role.strip() for role in bypass_roles_text.split("\n") if role.strip()]
    
    # Save global settings
    if st.button("💾 Save Global RLS Settings", type="primary"):
        new_config = {
            "enabled": rls_enabled,
            "default_user_field": default_user_field,
            "enforcement_mode": enforcement_mode,
            "bypass_roles": bypass_roles,
            "audit_enabled": audit_enabled,
            "collections": rls_config.get("collections", {})
        }
        set_rls_config(new_config)
        st.success("✅ Global RLS settings saved!")
        st.rerun()
    
    st.divider()
    
    # Collection-Specific RLS Configuration
    st.markdown("### Collection-Specific RLS Configuration")
    st.write("Override the default user field for specific collections or let AI detect the best field.")
    
    # Get available collections
    reg = load_registry()
    available_collections = list(reg.get("collections", {}).keys())
    
    if not available_collections:
        st.warning("No collections available. Add collections in the 'Collections & Schemas' tab first.")
    else:
        # Auto-detect fields for all collections
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🤖 Auto-Detect RLS Fields", type="secondary"):
                detector = RLSFieldDetector()
                all_fields = get_all_fields(reg)
                detected_fields = {}
                
                with st.spinner("🤖 Analyzing collections for ownership fields..."):
                    for collection in available_collections:
                        if collection in all_fields:
                            detected_field = detector.detect_ownership_field(
                                collection, all_fields[collection]
                            )
                            if detected_field:
                                detected_fields[collection] = detected_field
                
                if detected_fields:
                    st.session_state.detected_rls_fields = detected_fields
                    st.success(f"✅ Detected ownership fields for {len(detected_fields)} collections!")
                    for coll, field in detected_fields.items():
                        st.info(f"**{coll}**: {field}")
                else:
                    st.warning("⚠️ No suitable ownership fields detected automatically.")
        
        with col2:
            if st.button("📋 Apply All Detected Fields"):
                if "detected_rls_fields" in st.session_state:
                    for collection, field in st.session_state.detected_rls_fields.items():
                        set_rls_collection_field(collection, field, enforcement_mode)
                    st.success(f"✅ Applied RLS fields to {len(st.session_state.detected_rls_fields)} collections!")
                    st.rerun()
                else:
                    st.error("Please run auto-detection first.")
        
        # Individual collection configuration
        st.markdown("#### Manual Configuration")
        
        selected_collection = st.selectbox(
            "Select Collection",
            options=[""] + available_collections,
            help="Choose a collection to configure its RLS field"
        )
        
        if selected_collection:
            # Get current configuration
            current_field = get_rls_collection_field(selected_collection)
            collections_config = rls_config.get("collections", {})
            current_enforcement = "base_only"
            
            if selected_collection in collections_config:
                current_enforcement = collections_config[selected_collection].get("enforcement", "base_only")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Show available fields for this collection
                collection_fields = get_all_fields(reg).get(selected_collection, [])
                field_names = [f["name"] for f in collection_fields]
                
                if current_field in field_names:
                    field_index = field_names.index(current_field)
                else:
                    field_index = 0
                
                new_field = st.selectbox(
                    "RLS Field",
                    options=field_names,
                    index=field_index,
                    help="Field that identifies row ownership for this collection"
                )
            
            with col2:
                new_enforcement = st.selectbox(
                    "Enforcement",
                    options=["base_only", "all_involved"],
                    index=0 if current_enforcement == "base_only" else 1,
                    help="How to enforce RLS for this collection"
                )
            
            with col3:
                st.markdown("&nbsp;")
                if st.button("💾 Save"):
                    set_rls_collection_field(selected_collection, new_field, new_enforcement)
                    st.success(f"✅ Saved RLS config for {selected_collection}")
                    st.rerun()
                
                if st.button("🗑️ Remove"):
                    delete_rls_collection_config(selected_collection)
                    st.success(f"🗑️ Removed RLS config for {selected_collection}")
                    st.rerun()
            
            # Show field details
            if new_field and collection_fields:
                field_info = next((f for f in collection_fields if f["name"] == new_field), None)
                if field_info:
                    st.info(f"**{new_field}** ({field_info['type']}): {field_info.get('description', 'No description')}")
    
    st.divider()
    
    # Current RLS Configuration Summary
    st.markdown("### Current RLS Configuration")
    
    if rls_enabled:
        st.success("🟢 Row-Level Security is **ENABLED**")
    else:
        st.error("🔴 Row-Level Security is **DISABLED**")
    
    # Summary table
    config_summary = []
    collections_config = rls_config.get("collections", {})
    
    for collection in available_collections:
        field = get_rls_collection_field(collection)
        enforcement = "base_only"  # default
        source = "Default"
        
        if collection in collections_config:
            enforcement = collections_config[collection].get("enforcement", "base_only")
            source = "Override"
        
        config_summary.append({
            "Collection": collection,
            "RLS Field": field,
            "Enforcement": enforcement,
            "Source": source
        })
    
    if config_summary:
        st.dataframe(
            pd.DataFrame(config_summary),
            use_container_width=True,
            column_config={
                "Collection": st.column_config.TextColumn("Collection", width="medium"),
                "RLS Field": st.column_config.TextColumn("RLS Field", width="medium"),
                "Enforcement": st.column_config.TextColumn("Enforcement", width="small"),
                "Source": st.column_config.TextColumn("Source", width="small")
            }
        )
    
    # RLS Testing
    st.divider()
    st.markdown("### RLS Query Testing")
    st.write("Test how RLS will modify queries for different users and roles.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        test_user_id = st.text_input("Test User ID", value="user123")
        test_role = st.selectbox("Test Role", options=["user", "admin", "customer", "employee"] + bypass_roles)
        test_collection = st.selectbox("Test Collection", options=available_collections)
    
    with col2:
        test_query = st.text_area(
            "Test Query (JSON)",
            value='{"status": "active"}',
            height=100,
            help="Enter a MongoDB query to see how RLS will modify it"
        )
    
    if st.button("🧪 Test RLS Query Enhancement"):
        try:
            # Parse the test query
            import json
            original_query = json.loads(test_query)
            
            # Create RLS interceptor
            rls = create_rls_interceptor(test_user_id, test_role)
            
            # Test different query types
            enhanced_find = rls.enhance_find_query(test_collection, original_query)
            enhanced_pipeline = rls.enhance_aggregate_pipeline(test_collection, [{"$match": original_query}])
            enhanced_update_filter, _ = rls.enhance_update_query(test_collection, original_query, {"$set": {"updated": True}})
            enhanced_delete = rls.enhance_delete_query(test_collection, original_query)
            
            # Show results
            st.markdown("#### Query Enhancement Results:")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Original Query:**")
                st.code(json.dumps(original_query, indent=2), language="json")
                
                st.markdown("**Enhanced Find:**")
                st.code(json.dumps(enhanced_find, indent=2), language="json")
            
            with col2:
                st.markdown("**Enhanced Update Filter:**")
                st.code(json.dumps(enhanced_update_filter, indent=2), language="json")
                
                st.markdown("**Enhanced Delete:**")
                st.code(json.dumps(enhanced_delete, indent=2), language="json")
            
            st.markdown("**Enhanced Aggregation Pipeline:**")
            st.code(json.dumps(enhanced_pipeline, indent=2), language="json")
            
            # Show audit log
            audit_log = rls.get_audit_log()
            if audit_log:
                st.markdown("**Audit Log:**")
                for entry in audit_log:
                    st.caption(f"[{entry['timestamp']}] {entry['action']}: {entry['message']}")
            
        except json.JSONDecodeError:
            st.error("❌ Invalid JSON in test query")
        except Exception as e:
            st.error(f"❌ Error testing RLS: {e}")

with tab_eval:  # Batch Evaluations
    # if st.session_state['current_tab'] == 'Batch Evaluations':
    try:
        # Import batch evaluation module
        from batch_evaluation import render_batch_evaluation_ui
        render_batch_evaluation_ui()
        
    except ImportError as e:
        st.error(f"Failed to import batch evaluation module: {e}")
    # else:
    #     # Just show a placeholder when not on this tab
    #     st.info("Click to load Batch Evaluation")


