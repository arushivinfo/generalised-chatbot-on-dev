# admin_schema_ui.py
import os, json, pandas as pd, streamlit as st
from datetime import date, datetime, time
from pymongo import MongoClient
from uuid import uuid4
from schema_registry import (
    load_registry, save_registry, list_collections, upsert_collection, delete_collection,
    set_options_max,
    get_connection_config, set_connection_config,
    get_rls_config, set_rls_config, get_rls_layers, set_rls_layers,
    get_available_rls_fields_for_collection, add_collection_relation,
    get_response_prompt_config, set_response_prompt_config
)
from core_rules import render_schema_section_all
from llm_services import call_admin_ai_model
from response_gen import get_admin_prompt_sections
from default_prompts import (
    DEFAULT_PROMPT_SECTIONS,
    DEFAULT_SUGGESTED_QUESTIONS_SETTINGS,
    DEFAULT_ANSWERING_TEMPLATES,
    get_default_prompt_sections,
    get_default_suggested_questions_settings,
    get_default_answering_templates
)
import re
import pandas as pd
import numpy as np
import json
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llm_services import render_llm_model_config, call_narrator_model
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

# def get_llm():
#     key = os.getenv("OPENAI_API_KEY")
#     if not key:
#         return None
#     return ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=key)

def generate_ai_descriptions(df: pd.DataFrame, sample_size: int = 3) -> dict:
    """Ask AI to describe each column based on a sample of the data."""
    # llm = get_llm()

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
    massage = [{"role": "user", "content": prompt}]
    resp = call_admin_ai_model(massage)
    print("AI RAW OUTPUT:", (resp))

    # ---- 4) Parse JSON safely ----
    try:
        return json.loads(resp.content)
    except Exception:
        # Try extracting JSON substring
        match = re.search(r"\{.*\}", resp, re.S)
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

# Initialize session state for prompt sections (load from admin config)
if "prompt_sections" not in st.session_state:
    from schema_registry import get_response_prompt_config
    admin_config = get_response_prompt_config()
    
    # Load from admin config if available, otherwise use defaults
    if admin_config.get("prompt_sections"):
        st.session_state.prompt_sections = admin_config["prompt_sections"]
    else:
        st.session_state.prompt_sections = DEFAULT_PROMPT_SECTIONS.copy()

# Initialize answering templates (load from admin config)
if "answering_templates" not in st.session_state:
    from schema_registry import get_response_prompt_config
    admin_config = get_response_prompt_config()
    
    # Load from admin config if available, otherwise empty list
    st.session_state.answering_templates = admin_config.get("answering_templates", [])

# Default suggested questions settings (exactly as in frontend)
DEFAULT_SUGGESTED_QUESTIONS_PROMPT = """\
Generate relevant follow-up questions based on the user's original question and the assistant's answer. Focus on:
Questions should be short(10-12 words) and simple also highly relevant to the context according to the Question and Answer.
Make questions specific, actionable, and likely to provide valuable insights for the users.
"""

DEFAULT_SUGGESTED_QUESTIONS_COUNT = 3

# Initialize suggested questions settings (load from admin config)
if "suggested_questions_settings" not in st.session_state:
    from schema_registry import get_response_prompt_config
    admin_config = get_response_prompt_config()
    
    # Load from admin config if available, otherwise use defaults
    if admin_config.get("suggested_questions_settings"):
        st.session_state.suggested_questions_settings = admin_config["suggested_questions_settings"]
    else:
        st.session_state.suggested_questions_settings = {
            "enabled": True,
            "custom_prompt": DEFAULT_SUGGESTED_QUESTIONS_PROMPT,
            "max_questions": DEFAULT_SUGGESTED_QUESTIONS_COUNT
        }


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

tab_schema, tab_rules, tab_access, tab_rls, tab_eval, tab_model_management = st.tabs(["📊 Collections & Schemas", "📝 Core Rules & Prompt", "👥 User Access Control", "🔒 Row-Level Security", "🔍 Batch Evaluations", "🛠️ Model Management"])

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
            if st.button("💾 Save Collection & Relationships", type="primary", use_container_width=True):
                if not coll_name.strip():
                    st.error("⚠️ Please enter a collection name.")
                elif not st.session_state.schema_fields:
                    st.error("⚠️ Please add at least one field to the schema.")
                else:
                    from schema_registry import set_collection_relations
                    # Save the main collection schema
                    reg = load_registry()
                    reg["collections"][coll_name] = {
                        "description": coll_desc,
                        "fields": st.session_state.schema_fields
                    }
                    
                    # Save relationships if any
                    if hasattr(st.session_state, 'relations_data') and st.session_state.relations_data:
                        # Filter out empty relationships
                        valid_relations = [
                            rel for rel in st.session_state.relations_data
                            if rel.get("alias") and rel.get("ref_collection") and 
                               rel.get("local_field") and rel.get("foreign_field")
                        ]
                        if valid_relations:
                            reg["collections"][coll_name]["relations"] = valid_relations
                    
                    save_registry(reg)
                    num_relations = len(st.session_state.get('relations_data', []))
                    st.success(f"✅ Saved **{coll_name}** with {len(st.session_state.schema_fields)} fields and {num_relations} relationships!")
                    st.balloons()
                    st.rerun()

        # Relationship configuration
        st.markdown("#### Collection Relationships (For Joins)")
        st.write("Define how this collection relates to other collections for join queries.")
        
        # Get current relationships
        current_relations = []
        if edit_mode == "Edit Existing" and coll_name:
            from schema_registry import get_collection_relations
            current_relations = get_collection_relations(coll_name)
        
        # AI Suggestions Section
        if len(existing_colls) > 1 and coll_name:
            with st.expander("🤖 AI Relationship Suggestions", expanded=False):
                col_ai2 = st.columns([1])[0]
                
                with col_ai2:
                    if st.button("🤖 Generate AI Suggestions", key="ai_suggestions_main"):
                        try:
                            with st.spinner("🤖 Analyzing relationships..."):
                                from ai_relationship_suggester import get_ai_relationship_suggestions
                                
                                # Get current schema data
                                collections_data = {}
                                for existing_coll in existing_colls:
                                    existing_data = list_collections()[existing_coll]
                                    collections_data[existing_coll] = existing_data
                                
                                # Get AI suggestions
                                suggestions = get_ai_relationship_suggestions(collections_data)
                                st.session_state.ai_suggestions = suggestions
                                
                                if suggestions:
                                    st.success(f"✅ Generated {len(suggestions)} relationship suggestions!")
                                else:
                                    st.warning("⚠️ No relationship suggestions generated.")
                                    
                        except Exception as e:
                            st.error(f"❌ Error generating AI suggestions: {e}")
                
                # Display current AI suggestions
                if hasattr(st.session_state, 'ai_suggestions') and st.session_state.ai_suggestions:
                    collection_suggestions = [
                        s for s in st.session_state.ai_suggestions
                        if s.get('source_collection') == coll_name or s.get('target_collection') == coll_name
                    ]
                    if collection_suggestions:
                        st.markdown("**AI Suggestions for this collection:**")
                        for i, suggestion in enumerate(collection_suggestions):
                            confidence = suggestion.get('confidence', 'medium')
                            confidence_color = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}.get(confidence, '⚪')
                            
                            # Display relationship direction properly
                            if suggestion.get('source_collection') == coll_name:
                                direction_text = f"{coll_name}.{suggestion.get('source_field')} → {suggestion.get('target_collection')}.{suggestion.get('target_field')}"
                            else:
                                direction_text = f"{coll_name}.{suggestion.get('target_field')} ← {suggestion.get('source_collection')}.{suggestion.get('source_field')}"
                            
                            st.write(f"{confidence_color} **{suggestion.get('suggested_alias', 'N/A')}**: {direction_text}")
                            st.caption(f"   Cardinality: {suggestion.get('cardinality', 'N/A')} | {suggestion.get('reason', 'AI suggested relationship')}")
                            st.divider()
        
        # Relationship editor
        if 'relations_data' not in st.session_state:
            st.session_state.relations_data = current_relations or []
        
        # Add new relationship button
        if st.button("➕ Add Manual Relationship"):
            st.session_state.relations_data.append({
                "alias": "",
                "ref_collection": "",
                "local_field": "",
                "foreign_field": "",
                "cardinality": "one_to_many",
                "join_type": "left"
            })
        
        # Display relationship editor
        relations_to_remove = []
        available_collections = [c for c in existing_colls if c != coll_name] if coll_name else existing_colls
        
        for i, relation in enumerate(st.session_state.relations_data):
            st.markdown(f"**Relationship {i+1}:**")
            
            col_a, col_b, col_c = st.columns([1, 1, 0.2])
            
            with col_a:
                relation["alias"] = st.text_input(
                    "Alias", 
                    value=relation.get("alias", ""), 
                    key=f"rel_alias_{i}",
                    help="Short name to reference this join"
                )
                relation["ref_collection"] = st.selectbox(
                    "Target Collection",
                    options=[""] + available_collections,
                    index=available_collections.index(relation.get("ref_collection", "")) + 1 if relation.get("ref_collection") in available_collections else 0,
                    key=f"rel_collection_{i}"
                )
            
            with col_b:
                relation["local_field"] = st.text_input(
                    "Local Field", 
                    value=relation.get("local_field", ""), 
                    key=f"rel_local_{i}",
                    help="Field in this collection"
                )
                relation["foreign_field"] = st.text_input(
                    "Foreign Field", 
                    value=relation.get("foreign_field", ""), 
                    key=f"rel_foreign_{i}",
                    help="Field in target collection"
                )
            
            with col_c:
                st.markdown("&nbsp;")
                if st.button("🗑️", key=f"del_rel_{i}", help="Delete relationship"):
                    relations_to_remove.append(i)
            
            col_d, col_e = st.columns([1, 1])
            with col_d:
                relation["cardinality"] = st.selectbox(
                    "Cardinality",
                    options=["one_to_many", "many_to_one", "one_to_one"],
                    index=["one_to_many", "many_to_one", "one_to_one"].index(relation.get("cardinality", "one_to_many")),
                    key=f"rel_card_{i}"
                )
            
            with col_e:
                relation["join_type"] = st.selectbox(
                    "Join Type",
                    options=["left", "inner"],
                    index=["left", "inner"].index(relation.get("join_type", "left")),
                    key=f"rel_type_{i}"
                )
            
            st.divider()
        
        # Remove relationships marked for deletion
        for i in sorted(relations_to_remove, reverse=True):
            del st.session_state.relations_data[i]
            st.rerun()

    # Existing collections management (full width)
    st.divider()
    st.header("📚 Existing Collections")
    
    colls = list_collections()
    if colls:
        # Create enhanced table view with relationships
        from schema_registry import get_collection_relations
        
        coll_data = []
        for k, v in colls.items():
            relations = get_collection_relations(k)
            relation_summary = f"{len(relations)} relationships" if relations else "No relationships"
            
            coll_data.append({
                "Collection": k,
                "Description": v.get("description", ""),
                "Fields": len(v.get("fields", [])),
                "Relationships": relation_summary,
                "Last Modified": "N/A"  # Could add timestamp if needed
            })
        
        st.dataframe(
            pd.DataFrame(coll_data),
            use_container_width=True,
            column_config={
                "Collection": st.column_config.TextColumn("Collection Name", width="medium"),
                "Description": st.column_config.TextColumn("Description", width="large"),
                "Fields": st.column_config.NumberColumn("Field Count", width="small"),
                "Relationships": st.column_config.TextColumn("Relationships", width="medium"),
                "Last Modified": st.column_config.TextColumn("Last Modified", width="medium")
            }
        )
        
        # Relationship summary section
        with st.expander("🔗 Relationship Summary", expanded=False):
            from schema_registry import get_relationship_summary
            summary = get_relationship_summary()
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                st.metric("Total Relationships", summary['total_relationships'])
            
            with col2:
                st.metric("Collections with Relations", summary['collections_with_relations'])
            
            with col3:
                if summary['relationship_patterns']:
                    st.markdown("**Relationship Patterns:**")
                    for pattern, count in summary['relationship_patterns'].items():
                        st.caption(f"• {pattern} ({count}x)")
            
            # Display detailed relationships with containers instead of nested expanders
            if summary['collections']:
                st.markdown("**Detailed Relationships:**")
                for collection, relations in summary['collections'].items():
                    st.markdown(f"### 📄 {collection} ({len(relations)} relations)")
                    
                    for i, rel in enumerate(relations):
                        with st.container():
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                st.write(f"**{rel['alias']}** → {rel['ref_collection']}")
                                st.caption(f"Join: {collection}.{rel['local_field']} = {rel['ref_collection']}.{rel['foreign_field']}")
                                st.caption(f"Type: {rel['cardinality']}, {rel['join_type']} join")
                            
                            with col_b:
                                if st.button(f"Edit", key=f"edit_rel_{collection}_{i}"):
                                    # Set up editing mode for this relationship
                                    st.session_state.edit_relationship = {
                                        'collection': collection,
                                        'index': i,
                                        'relation': rel
                                    }
                                    st.rerun()
                            
                            st.divider()
        
        # Global relationship actions
        st.markdown("### Global Relationship Actions")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🤖 Suggest All Relationships", type="secondary"):
                try:
                    from ai_relationship_suggester import get_ai_relationship_suggestions
                    from schema_registry import bulk_add_relationships
                    
                    # Prepare all collections data
                    reg = load_registry()
                    collections_data = {}
                    
                    for col_name in colls.keys():
                        if col_name in reg.get("collections", {}):
                            collections_data[col_name] = {
                                "description": reg["collections"][col_name].get("description", ""),
                                "fields": reg["collections"][col_name].get("fields", [])
                            }
                    
                    with st.spinner("🤖 AI analyzing all collection relationships..."):
                        suggestions = get_ai_relationship_suggestions(collections_data)
                    
                    if suggestions:
                        st.session_state.global_ai_suggestions = suggestions
                        total_suggestions = len(suggestions)
                        st.success(f"✅ Generated {total_suggestions} relationship suggestions!")
                    else:
                        st.warning("⚠️ No global relationship suggestions generated")
                
                except Exception as e:
                    st.error(f"❌ Error getting global suggestions: {e}")
        
        with col2:
            if hasattr(st.session_state, 'global_ai_suggestions') and st.session_state.global_ai_suggestions:
                if st.button("📝 Apply All Suggestions", type="primary"):
                    try:
                        # Apply suggestions by adding them to appropriate collections
                        added_count = 0
                        
                        for suggestion in st.session_state.global_ai_suggestions:
                            source_collection = suggestion.get('source_collection')
                            target_collection = suggestion.get('target_collection')
                            
                            # Add relationship to source collection
                            if source_collection in colls:
                                clean_suggestion = {
                                    'alias': suggestion.get('suggested_alias', target_collection),
                                    'ref_collection': target_collection,
                                    'local_field': suggestion.get('source_field'),
                                    'foreign_field': suggestion.get('target_field'),
                                    'cardinality': suggestion.get('cardinality', 'one_to_many'),
                                    'join_type': suggestion.get('join_type', 'left')
                                }
                                
                                if add_collection_relation(source_collection, clean_suggestion):
                                    added_count += 1
                        
                        st.success(f"✅ Added {added_count} relationships!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error applying suggestions: {e}")
        
        with col3:
            # Show preview of suggestions
            if hasattr(st.session_state, 'global_ai_suggestions') and st.session_state.global_ai_suggestions:
                st.markdown("**AI Suggestions Preview:**")
                for suggestion in st.session_state.global_ai_suggestions:
                    source_col = suggestion.get('source_collection', 'N/A')
                    target_col = suggestion.get('target_collection', 'N/A') 
                    alias = suggestion.get('suggested_alias', target_col)
                    st.caption(f"• **{source_col}** → {alias} ({target_col})")
        
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
    
    # Relationship Editor Modal
    if hasattr(st.session_state, 'edit_relationship'):
        edit_data = st.session_state.edit_relationship
        
        with st.expander(f"✏️ Edit Relationship: {edit_data['collection']}", expanded=True):
            rel = edit_data['relation']
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                new_alias = st.text_input("Alias", value=rel.get("alias", ""), key="edit_alias")
                new_ref_collection = st.selectbox(
                    "Target Collection",
                    options=list(colls.keys()),
                    index=list(colls.keys()).index(rel.get("ref_collection", "")) if rel.get("ref_collection") in colls else 0,
                    key="edit_ref_collection"
                )
            
            with col2:
                new_local_field = st.text_input("Local Field", value=rel.get("local_field", ""), key="edit_local_field")
                new_foreign_field = st.text_input("Foreign Field", value=rel.get("foreign_field", ""), key="edit_foreign_field")
            
            col3, col4 = st.columns([1, 1])
            
            with col3:
                new_cardinality = st.selectbox(
                    "Cardinality",
                    options=["one_to_many", "many_to_one", "one_to_one"],
                    index=["one_to_many", "many_to_one", "one_to_one"].index(rel.get("cardinality", "one_to_many")),
                    key="edit_cardinality"
                )
            
            with col4:
                new_join_type = st.selectbox(
                    "Join Type",
                    options=["left", "inner"],
                    index=["left", "inner"].index(rel.get("join_type", "left")),
                    key="edit_join_type"
                )
            
            col_save, col_cancel, col_delete = st.columns([1, 1, 1])
            
            with col_save:
                if st.button("💾 Save Changes", type="primary"):
                    updated_relation = {
                        "alias": new_alias,
                        "ref_collection": new_ref_collection,
                        "local_field": new_local_field,
                        "foreign_field": new_foreign_field,
                        "cardinality": new_cardinality,
                        "join_type": new_join_type
                    }
                    
                    from schema_registry import update_collection_relationship
                    if update_collection_relationship(edit_data['collection'], edit_data['index'], updated_relation):
                        st.success("✅ Relationship updated successfully!")
                        del st.session_state.edit_relationship
                        st.rerun()
                    else:
                        st.error("❌ Failed to update relationship")
            
            with col_cancel:
                if st.button("❌ Cancel"):
                    del st.session_state.edit_relationship
                    st.rerun()
            
            with col_delete:
                if st.button("🗑️ Delete Relation", type="secondary"):
                    from schema_registry import remove_collection_relation
                    if remove_collection_relation(edit_data['collection'], edit_data['index']):
                        st.success("🗑️ Relationship deleted!")
                        del st.session_state.edit_relationship
                        st.rerun()
                    else:
                        st.error("❌ Failed to delete relationship")

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
    st.subheader("Core Rules & Prompts Configuration")
    
    # Create sub-tabs for different prompt types
    search_tab, response_tab = st.tabs(["🔍 Search Agent Prompts", "💬 Response Generation Prompts"])
    with search_tab:
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

    with response_tab:
        st.markdown("### Prompt Sections")
        for section, text in list(st.session_state.prompt_sections.items()):
            key = f"prompt_{section}"
            if key not in st.session_state:
                st.session_state[key] = text
            st.text_area(section, key=key, height=180)

        if st.button("Save Prompt Sections"):
            # Update session state from form inputs
            for section in list(st.session_state.prompt_sections.keys()):
                st.session_state.prompt_sections[section] = st.session_state.get(f"prompt_{section}", "")
            
            # Save to admin registry
            from schema_registry import set_response_prompt_config
            set_response_prompt_config(
                st.session_state.prompt_sections,
                st.session_state.suggested_questions_settings,
                st.session_state.answering_templates
            )
            st.success("Prompt sections saved to admin configuration")

        st.divider()

        # NEW: Suggested Questions Settings Section
        st.markdown("### Suggested Questions Settings")
        
        # Enable/Disable suggested questions
        st.session_state.suggested_questions_settings["enabled"] = st.checkbox(
            "Enable Suggested Questions", 
            value=st.session_state.suggested_questions_settings.get("enabled", True),
            help="Show follow-up question suggestions after each response"
        )
        
        if st.session_state.suggested_questions_settings["enabled"]:
            # Number of suggested questions
            st.session_state.suggested_questions_settings["max_questions"] = st.slider(
                "Number of Suggested Questions",
                min_value=1,
                max_value=5,
                value=st.session_state.suggested_questions_settings.get("max_questions", 3),
                help="How many suggested questions to display"
            )
            
            # Custom prompt for suggested questions
            st.markdown("#### Custom Prompt for Suggested Questions")
            if "suggested_questions_prompt_key" not in st.session_state:
                st.session_state.suggested_questions_prompt_key = st.session_state.suggested_questions_settings.get("custom_prompt", "")
            
            custom_prompt = st.text_area(
                "Instructions for generating suggested questions",
                key="suggested_questions_prompt_key",
                height=200,
                help="This prompt will be used to generate follow-up questions. Be specific about the type of questions you want."
            )
            st.session_state.suggested_questions_settings["custom_prompt"] = custom_prompt
            
            # Save button for suggested questions settings
            if st.button("Save Suggested Questions Settings"):
                # Save to admin registry
                from schema_registry import set_response_prompt_config
                set_response_prompt_config(
                    st.session_state.prompt_sections,
                    st.session_state.suggested_questions_settings,
                    st.session_state.answering_templates
                )
                st.success("Suggested questions settings saved to admin configuration")

        st.divider()
        
        st.markdown("### Answering Templates (appended to **Formatting** at runtime)")

    # Render existing templates with edit/toggle/delete
    to_delete = []
    for i, tpl in enumerate(st.session_state.answering_templates):
        # Handle both old and new template structures
        template_name = tpl.get('name', f'Template {i+1}')
        template_enabled = tpl.get('enabled', True)  # Default to enabled if not specified
        template_id = tpl.get('id', f'template_{i}')  # Generate ID if missing
        
        with st.expander(f"{'✅' if template_enabled else '⏸️'} {template_name}", expanded=False):
            col1, col2 = st.columns([3,1])
            with col1:
                tpl['enabled'] = st.checkbox("Enabled", value=template_enabled, key=f"tpl_enabled_{template_id}")
            with col2:
                if st.button("Delete", key=f"tpl_del_{template_id}"):
                    to_delete.append(template_id)

            tpl['name'] = st.text_input("Template Name", value=template_name, key=f"tpl_name_{template_id}")
            
            # Handle both 'text' and 'template' fields
            template_content = tpl.get('text', tpl.get('template', ''))
            tpl['text'] = st.text_area("Template Content", value=template_content, key=f"tpl_text_{template_id}", height=160)
            
            # Ensure the template has an ID
            tpl['id'] = template_id

    if to_delete:
        st.session_state.answering_templates = [t for t in st.session_state.answering_templates if t["id"] not in to_delete]
        
        # Auto-save after deletion
        from schema_registry import set_response_prompt_config
        set_response_prompt_config(
            st.session_state.prompt_sections,
            st.session_state.suggested_questions_settings,
            st.session_state.answering_templates
        )
        st.success("Template(s) deleted and saved to admin configuration.")


    st.markdown("#### Add New Template")

    # Ensure keys exist before widgets render
    if "new_tpl_name" not in st.session_state:
        st.session_state.new_tpl_name = ""
    if "new_tpl_text" not in st.session_state:
        st.session_state.new_tpl_text = ""

    new_name = st.text_input(
        "New Template Name",
        key="new_tpl_name",
        placeholder="e.g., For Venue Queries",
    )
    new_text = st.text_area(
        "New Template Content",
        key="new_tpl_text",
        height=140,
    )

    def _add_template_cb():
        name = st.session_state.get("new_tpl_name", "").strip()
        text = st.session_state.get("new_tpl_text", "").strip()
        if not name or not text:
            st.warning("Please provide both a name and content.")
            return
        st.session_state.answering_templates.append(
            {"id": str(uuid4()), "name": name, "enabled": True, "text": text}
        )
        
        # Auto-save after adding template
        from schema_registry import set_response_prompt_config
        set_response_prompt_config(
            st.session_state.prompt_sections,
            st.session_state.suggested_questions_settings,
            st.session_state.answering_templates
        )
        
        # Reset fields safely inside callback
        st.session_state.new_tpl_name = ""
        st.session_state.new_tpl_text = ""
        st.success("Template added and saved to admin configuration.")

    # Save all template changes button
    if st.button("💾 Save All Template Changes"):
        # Save current template modifications to admin registry
        from schema_registry import set_response_prompt_config
        set_response_prompt_config(
            st.session_state.prompt_sections,
            st.session_state.suggested_questions_settings,
            st.session_state.answering_templates
        )
        st.success("All template changes saved to admin configuration!")

    st.button("Add Template", on_click=_add_template_cb)

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
    else:
        # Debug RLS field detection
        st.markdown("#### 🔍 RLS Field Detection Debug")
        with st.expander("Debug Information", expanded=False):
            from schema_registry import debug_rls_field_detection
            
            if st.button("🔍 Run RLS Field Debug Analysis"):
                with st.spinner("Analyzing RLS field detection..."):
                    debug_rls_field_detection()
                    st.success("Debug analysis complete - check console output")
        
        st.divider()
        
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
            "collections": rls_config.get("collections", {}),
            "rls_layers": rls_config.get("rls_layers", [])  # Preserve layers
        }
        set_rls_config(new_config)
        st.success("✅ Global RLS settings saved!")
        st.rerun()
    
    st.divider()
    
    # Multi-Layer Authentication Configuration
    st.markdown("### Multi-Layer Authentication")
    st.write("Configure multiple authentication fields (user_id, department, team_id, etc.)")
    
    # Get current RLS layers
    from schema_registry import get_rls_layers, set_rls_layers, get_available_rls_fields_for_collection
    rls_layers = get_rls_layers()
    
    # Add new layer form
    with st.expander("➕ Add New Authentication Layer", expanded=len(rls_layers) == 0):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            field_name = st.text_input(
                "Field Name",
                placeholder="e.g., user_id, department, team_id",
                help="The database field name used for authentication"
            )
        
        with col2:
            display_name = st.text_input(
                "Display Name",
                placeholder="e.g., User ID, Department, Team",
                help="User-friendly name shown in the frontend"
            )
            
        with col3:
            enabled = st.checkbox(
                "Enabled",
                value=True,
                help="Enable this authentication layer"
            )
        
        if st.button("Add Layer", type="secondary"):
            if field_name and display_name:
                new_layer = {
                    "field_name": field_name,
                    "display_name": display_name,
                    "enabled": enabled,
                    "order": len(rls_layers) + 1
                }
                
                # Check if field already exists
                existing_fields = [layer.get("field_name") for layer in rls_layers]
                if field_name in existing_fields:
                    st.error(f"Field '{field_name}' already exists in authentication layers")
                else:
                    rls_layers.append(new_layer)
                    set_rls_layers(rls_layers)
                    st.success(f"✅ Added authentication layer: {display_name}")
                    st.rerun()
            else:
                st.error("Please fill in both Field Name and Display Name")
    
    # Display current layers
    if rls_layers:
        st.markdown("### Current Authentication Layers")
        
        layers_to_remove = []
        for i, layer in enumerate(rls_layers):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                
                with col1:
                    st.text(f"Field: {layer.get('field_name', '')}")
                
                with col2:
                    st.text(f"Display: {layer.get('display_name', '')}")
                
                with col3:
                    new_enabled = st.checkbox(
                        f"Enabled",
                        value=layer.get("enabled", True),
                        key=f"enabled_{i}"
                    )
                    if new_enabled != layer.get("enabled", True):
                        rls_layers[i]["enabled"] = new_enabled
                
                with col4:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete layer"):
                        layers_to_remove.append(i)
        
        # Remove layers marked for deletion
        if layers_to_remove:
            for i in sorted(layers_to_remove, reverse=True):
                del rls_layers[i]
            set_rls_layers(rls_layers)
            st.rerun()
        
        # Save changes button
        if st.button("💾 Save Layer Changes"):
            set_rls_layers(rls_layers)
            st.success("✅ Authentication layers updated successfully!")
            st.rerun()
    else:
        st.info("No authentication layers configured. Add one above to get started.")        # Field Availability Matrix
        if rls_layers and available_collections:
            st.markdown("### Field Availability Matrix")
            st.write("Shows which collections have which authentication fields available.")
            
            # Create availability matrix
            availability_data = []
            
            for collection in available_collections:
                # Use enhanced field validation
                from schema_registry import is_valid_rls_field
                collection_fields_info = get_all_fields(reg).get(collection, [])
                available_fields = []
                
                # Check each field to see if it's suitable for RLS
                for field_info in collection_fields_info:
                    if is_valid_rls_field(field_info):
                        available_fields.append(field_info["name"])
                
                row = {"Collection": collection}
                
                for layer in rls_layers:
                    field_name = layer.get("field_name", "")
                    if field_name in available_fields:
                        row[layer.get("display_name", field_name)] = "✅ Available"
                    else:
                        row[layer.get("display_name", field_name)] = "❌ Missing"
                
                availability_data.append(row)
            
            if availability_data:
                availability_df = pd.DataFrame(availability_data)
                st.dataframe(availability_df, use_container_width=True)
    
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

with tab_model_management:
    try:
        render_llm_model_config()
    except Exception as e:
        st.error(f"Failed to load model management: {e}")
        st.write("Please ensure llm_services.py is available and properly configured.")

