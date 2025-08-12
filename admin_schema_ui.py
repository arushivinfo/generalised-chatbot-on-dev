# admin_schema_ui.py
import os, json, pandas as pd, streamlit as st
from pymongo import MongoClient
from schema_registry import (
    load_registry, save_registry, list_collections, upsert_collection, delete_collection,
    set_options_max
)
from core_rules import render_schema_section_all


ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
if ADMIN_TOKEN:
    if st.text_input("Admin token", type="password") != ADMIN_TOKEN:
        st.stop()

st.title("Admin • Collections & Schemas")

# Keep sample & schema in state so clicks survive reruns
if "sample_df" not in st.session_state:
    st.session_state.sample_df = None
if "schema_fields" not in st.session_state:
    st.session_state.schema_fields = []

reg = load_registry()

uri = st.text_input(
    "Mongo URI",
    value=os.getenv("MONGO_URI","mongodb://127.0.0.1:27017"),
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
role = st.selectbox("Role (for core rules)", ["matches", "players", "venues", "upcoming_match", "other"])
coll_name = st.text_input("Collection name (exact Mongo name)", "")
coll_desc = st.text_input("Description (short)","")

st.subheader("Sample Data")
source = st.radio("Source", ["Mongo", "CSV", "JSON"], horizontal=True)
df = None

if source == "Mongo":
    db  = st.text_input("Database", os.getenv("MONGO_DB", "sports_feed_stg"))
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
        st.error("Load sample data first (Mongo/CSV/JSON) before generating schema.")
    else:
        fields = []
        for col in df0.columns:
            t = infer_type(df0[col])
            ops = TYPE_OPS[t]
            opts = []
            if t == "string":
                uniq = df0[col].dropna().astype(str).str.strip().unique()
                opts = sorted(map(str, uniq))[:cap]  # IMPORTANT: cap to admin setting
            fields.append({
                "name": col,
                "type": t,
                "operations": ops,
                "description": "",
                "options": opts
            })
        st.session_state.schema_fields = fields

if upload:
    try:
        st.session_state.schema_fields = json.load(upload).get("fields", [])
        st.info(f"Loaded {len(st.session_state['schema_fields'])} fields from JSON.")
    except Exception as e:
        st.error(f"Invalid schema JSON: {e}")


if st.session_state.schema_fields:
    sch = pd.DataFrame(st.session_state.schema_fields)
    sch["type"] = sch["type"].astype("category").cat.set_categories(list(TYPE_OPS.keys()))
    sch["operations"] = sch["operations"].apply(lambda x: x if isinstance(x, list) else [])
    edited = st.data_editor(sch, num_rows="dynamic", use_container_width=True)

    # keep working copy updated across reruns
    st.session_state.schema_fields = edited.to_dict(orient="records")

    if st.button("Save collection"):
        if not coll_name.strip():
            st.warning("Please enter a collection name.")
        else:
            reg = load_registry()
            reg["collections"][coll_name] = {
                "role": role,
                "description": coll_desc,
                "fields": st.session_state.schema_fields
            }
            save_registry(reg)
            st.success(f"Saved schema for {coll_name} (role: {role})")

st.header("Existing Collections")
colls = list_collections()
st.write({k: {"role": v["role"], "fields": len(v.get("fields",[]))} for k,v in colls.items()})
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