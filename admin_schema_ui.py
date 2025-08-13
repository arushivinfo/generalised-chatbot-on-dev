# admin_schema_ui.py
import os, json, pandas as pd, streamlit as st
from pymongo import MongoClient
from schema_registry import (
    load_registry, save_registry, list_collections, upsert_collection, delete_collection,
    set_options_max
)
from core_rules import render_schema_section_all

import re
import pandas as pd
import numpy as np
import json
from langchain_openai import ChatOpenAI

def generate_ai_descriptions(df: pd.DataFrame, sample_size: int = 3) -> dict:
    """Ask AI to describe each column based on a sample of the data."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ---- 1) Take sample and convert all values to JSON-safe types ----
    sample_df = df.head(sample_size).copy()

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

    sample_df = sample_df.applymap(safe_value)
    sample_data = sample_df.to_dict(orient="records")

    # ---- 2) Build strict JSON-only prompt ----
    prompt = f"""
You are helping define a database schema for a cricket/fantasy sports analytics system.

Here is a sample of the dataset:
{json.dumps(sample_data, indent=2)}

For each column in the dataset, provide a short but precise description of what it represents.

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
            uniq = df[col].dropna().astype(str).str.strip().unique()
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



ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")
if ADMIN_TOKEN:
    if st.text_input("Admin token", type="password") != ADMIN_TOKEN:
        st.stop()

st.title("Admin • Collections & Schemas")
tab_schema, tab_rules = st.tabs(["Collections & Schemas", "Core Rules & Prompt"])

with tab_schema:

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



from schema_registry import (
    get_core_rules_config, set_core_rules_config,
    get_all_fields, get_descriptions, get_core_coll_map,
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
    st.subheader("Optional: Extra Match Context")
    cur_ctx = get_user_match_context()
    new_ctx = st.text_area("Extra context (shown as a separate system message)", value=cur_ctx, height=140)
    if st.button("Save Extra Context"):
        set_user_match_context(new_ctx)
        st.success("Extra context saved.")

    # 3) Effective rules preview
    st.divider()
    st.subheader("Effective Rules Preview")
    reg = load_registry()
    core_map = get_core_coll_map(reg)
    auto_rules = render_core_rules(core_map)
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
