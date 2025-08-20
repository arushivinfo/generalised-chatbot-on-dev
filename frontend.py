# cric_chat_ui.py
# Front-end for Perfect Lineup Chatbot 🏏   (uses search_agent_new + response_gen)
from uuid import uuid4
import uuid, json, inspect
from pathlib import Path
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

# --- BACK-END IMPORTS --------------------------------------------
from search_agent_new import run_search_agent          # returns (spec, rows_text)
from response_gen import DEFAULT_PROMPT_SECTIONS, compose_prompt,get_memory_prompt2
from response_gen import narrator                      # ChatOpenAI instance

# --- Collection metadata for rows formatting ---
# COLL_META = {
#     "players":        ("players_filtered_90696",      "players_filtered collection – contains historical data of players"),
#     "matches":        ("matches_filtered_90696",      "matches_filtered collection – contains historical match summaries"),
#     "venues":         ("venues_filtered_90696",       "venues_filtered collection – contains venue and ground stats"),
#     "upcoming_match": ("upcoming_match_90696_summary","upcoming_match collection – contains prediction data and details of upcoming match"),
# }
from schema_registry import load_registry, get_collection_names, get_descriptions

def _coll_meta_from_registry():
    reg = load_registry()
    names = get_collection_names(reg)       # {'matches': '<coll>', ...}
    desc = get_descriptions(reg)        # {'<coll>': 'short description', ...}
    return {name: (name, (desc.get(name) or "no description")) for name in names}


COLL_META = _coll_meta_from_registry()

def _normalize_coll_key(name: str) -> str:
    """
    Map raw collection names to canonical keys used in chosen_collections.
    """
    name = (name or "").lower()
    if name.startswith("upcoming_match"): return "upcoming_match"
    if name.startswith("players"):        return "players"
    if name.startswith("venues"):         return "venues"
    if name.startswith("matches"):        return "matches"
    return name  # fallback

def build_rows_for_prompt(dbg: dict) -> str:
    """Render full objects (pretty JSON) per collection."""
    cols_order = dbg.get("chosen_collections", []) or []
    results    = dbg.get("results", []) or []
    blocks     = []
    for i, coll in enumerate(cols_order):
        res  = results[i] if i < len(results) else {}
        docs = res.get("docs", []) or []
        if not docs:
            blocks.append(f"# {coll}\n(no rows)")
            continue
        blob = "\n\n".join("```json\n" + json.dumps(d, indent=2, default=str) + "\n```" for d in docs[:30])
        blocks.append(f"# {coll}\n{blob}")
    return "\n\n".join(blocks) or "(no rows)"


if "answering_templates" not in st.session_state:
    # Start empty; you can add any generic templates from the UI
    st.session_state.answering_templates = []


# --- 1.  callback for “thinking” pane ----------------------------

class StreamlitStepHandler(BaseCallbackHandler):
    """Collects agent/tool events and renders them to a container."""
    def __init__(self, container):
        self.container = container
        self.lines = []

    # ------------- callbacks -------------
    def on_agent_action(self, action, **_):
        self.lines.append(f"**Thought**: {action.log}")

    def on_tool_start(self, tool, *_args, **_):
        """Handle the start of a tool call.

        ``tool`` may be a Tool instance or a serialized dict depending on the
        langchain version. Accept *args to remain forwards compatible.
        """
        name = getattr(tool, "name", None)
        if name is None and isinstance(tool, dict):
            name = tool.get("name")
        self.lines.append(f"**Action**: {name}")

    def on_tool_end(self, output, **_):
        self.lines.append("**Observation**: *(tool output received)*")

    def on_agent_finish(self, finish, **_):
        self.lines.append("**Finish**")

    # ------------- renderer -------------
    def render(self):
        # Join lines once per render to avoid flicker
        self.container.markdown("\n\n".join(self.lines), unsafe_allow_html=True)


class StreamlitCB(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.lines = []

    def _push(self, txt): self.lines.append(txt)

    def on_agent_action(self, action, **kw): self._push(f"**Thought**: {action.log}")
    def on_tool_start  (self, tool, *_args, **kw):
        name = getattr(tool, "name", None)
        if name is None and isinstance(tool, dict):
            name = tool.get("name")
        self._push(f"**Action**: {name}")
    def on_tool_end    (self, output, **kw):  self._push("**Observation**: data retrieved")
    def on_agent_finish(self, finish, **kw):  self._push("**Finish**")

    def render(self):
        joined = "\n\n".join(self.lines) or "No steps"
        self.container.markdown(joined, unsafe_allow_html=True)

def get_cb(container):
    cb = StreamlitCB(container)
    ctx = get_script_run_ctx()
    for name, fn in inspect.getmembers(cb, inspect.ismethod):
        if name.startswith("on_"):
            setattr(cb, name, lambda *a, _fn=fn, **k: add_script_run_ctx(ctx) or _fn(*a, **k))
    return cb

# --- 2.  Page styling (copied from your old frontend) -------------
st.set_page_config(page_title="Data QA Chat", page_icon="🧠", layout="wide")

st.markdown(Path("frontend.css").read_text() if Path("frontend.css").exists() else """<style>
.chat-container{background:#fff;border-radius:8px;padding:10px;margin-bottom:20px;min-height:60vh;overflow-y:auto}
.stChatMessage{margin-bottom:15px}.stChatMessage>div{border-radius:10px;padding:10px;max-width:80%}
.stChatMessage.user>div{background:#007bff;color:#fff;margin-left:auto}
.stChatMessage.assistant>div{background:#e9ecef;color:#333}
.stChatInput{position:sticky;bottom:0;background:#fff;padding:10px;border-radius:20px;box-shadow:0 -2px 4px rgba(0,0,0,.1)}
.stButton>button{background:#dc3545;color:#fff;border-radius:8px;padding:8px 16px;border:none}.stButton>button:hover{background:#b02a37}
.title{text-align:center;color:#007bff;font-size:2em;margin-bottom:10px}
.subtitle{text-align:center;color:#6c757d;font-size:1.1em;margin-bottom:20px}
@media(max-width:768px){.stApp{padding:10px}.chat-container{min-height:50vh}.stChatMessage>div{max-width:90%}}
</style>""", unsafe_allow_html=True)

# --- 3.  Session state -------------------------------------------
if "chat" not in st.session_state: st.session_state.chat = []
if "sid"  not in st.session_state: st.session_state.sid  = str(uuid.uuid4())
# store prompt blocks so edits persist
if "prompt_sections" not in st.session_state:
    # start with the library defaults so text areas show initial values
    st.session_state.prompt_sections = DEFAULT_PROMPT_SECTIONS.copy()
# if "mem_agent" not in st.session_state:
#     from memory_agent import MemoryAgent
#     st.session_state.mem_agent = MemoryAgent(k=5)

# --- 4.  Header ---------------------------------------------------
st.markdown('<h1 class="title">Data QA Chat</h1>', unsafe_allow_html=True)

if st.button("Clear Chat"): st.session_state.chat = []; st.rerun()

chat_tab, prompt_tab = st.tabs(["Chat", "Prompt Settings"])

with chat_tab:
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    q = st.chat_input("Ask about your datasets…")

    if q:
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            resp_container = st.empty()

            step_box   = st.expander("Intermediate steps", expanded=False)
            step_cb    = StreamlitStepHandler(step_box)

            # STEP 0 – **no rewrite**: just use the original question
            standalone_q = q
            step_box.markdown("**Step**: Using original question (no rewrite) 🔍", unsafe_allow_html=True)

            # STEP 1 – structured search on the rewritten question
            with st.spinner("Planning and retrieving documents 📂️..."):
                spec, rows_text, dbg = run_search_agent(standalone_q, callbacks=[step_cb],history=get_memory_prompt2(1))
                step_cb.render()  # show steps up to now

            dbg_box = st.expander("Debug info", expanded=False)
            debug_content = {
                "Query Spec": spec,  # Includes filters, sort, and limit
                "MongoDB Filters": dbg.get("filters", []),
                "Chosen Collections": dbg.get("chosen_collections", []),
                "Raw Results": rows_text
            }
            dbg_box.code(json.dumps(debug_content, indent=2, default=str), language="json")

            # STEP 2  – narrative answer
            with st.spinner("Generating your answer 🔥..."):
                
                # lang = st.session_state.mem_agent.last_lang
                from lang_detect import LangDetectAgent
                mem = LangDetectAgent()  # Create a new instance for each query
                lang = mem.detect_language(q)
                print('Detected_languadge:',lang)  # Detect language of the question

                rows_clean = rows_text or "(no rows)"

                # 1) Build a structured rows blob from chosen collections (no raw JSON)
                rows_clean = build_rows_for_prompt(dbg)

                # 2) Append enabled “Answering Templates” ONLY into PROMPT_FORMATTING
                sections_rt = dict(st.session_state.prompt_sections)   # shallow copy
                formatting_text = sections_rt.get("formatting", "")

                enabled_templates = [
                    t["text"] for t in st.session_state.answering_templates
                    if t.get("enabled") and t.get("text","").strip()
                ]
                if enabled_templates:
                    formatting_text = (formatting_text.rstrip() + "\n\n" + "\n\n".join(enabled_templates)).strip()
                sections_rt["formatting"] = formatting_text  # PROMPT_FORMATTING only

                # 3) Compose final prompt with structured rows + augmented formatting
                prompt = compose_prompt(
                    sections_rt,  # make sure to use sections_rt, not original
                    question=q,
                    rows=rows_clean,
                    language=lang,
                    memory_context=get_memory_prompt2(1)  # get last 3 memories
                )

                prompt_debug = st.expander("Prompt sent to LLM", expanded=False)
                prompt_debug.code(prompt, language="markdown")

                stream = narrator.stream(
                    [HumanMessage(content=prompt)],
                    config={"callbacks": [step_cb]}
                )
                answer = ""
                for chunk in stream:
                    token = getattr(chunk, "content", "")
                    answer += token
                    resp_container.markdown(answer)

                step_cb.render()

            st.session_state.chat.append(("assistant", answer))

            # Save memory to cache
            from cache_memory import save_to_cache
            save_to_cache(q, answer)

with prompt_tab:
    st.markdown("### Prompt Sections")
    for section, text in list(st.session_state.prompt_sections.items()):
        key = f"prompt_{section}"
        if key not in st.session_state:
            st.session_state[key] = text
        st.text_area(section, key=key, height=180)

    if st.button("Save Prompt Sections"):
        for section in list(st.session_state.prompt_sections.keys()):
            st.session_state.prompt_sections[section] = st.session_state.get(f"prompt_{section}", "")
        st.success("Prompt sections updated")

    st.divider()
    st.markdown("### Answering Templates (appended to **Formatting** at runtime)")

    # Render existing templates with edit/toggle/delete
    to_delete = []
    for i, tpl in enumerate(st.session_state.answering_templates):
        with st.expander(f"{'✅' if tpl['enabled'] else '⏸️'} {tpl['name']}", expanded=False):
            col1, col2 = st.columns([3,1])
            with col1:
                tpl['enabled'] = st.checkbox("Enabled", value=tpl['enabled'], key=f"tpl_enabled_{tpl['id']}")
            with col2:
                if st.button("Delete", key=f"tpl_del_{tpl['id']}"):
                    to_delete.append(tpl['id'])

            tpl['name'] = st.text_input("Template Name", value=tpl['name'], key=f"tpl_name_{tpl['id']}")
            tpl['text'] = st.text_area("Template Content", value=tpl['text'], key=f"tpl_text_{tpl['id']}", height=160)

    if to_delete:
        st.session_state.answering_templates = [t for t in st.session_state.answering_templates if t["id"] not in to_delete]
        st.success("Template(s) deleted.")


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
        # Reset fields safely inside callback
        st.session_state.new_tpl_name = ""
        st.session_state.new_tpl_text = ""
        st.success("Template added.")

    st.button("Add Template", on_click=_add_template_cb)

from cache_memory import view_cache

st.write("Current cache:", view_cache())

    # st.markdown("#### Add New Template")
    # new_name = st.text_input("New Template Name", key="new_tpl_name", placeholder="e.g., For Venue Queries")
    # new_text = st.text_area("New Template Content", key="new_tpl_text", height=140)
    # if st.button("Add Template"):
    #     if new_name.strip() and new_text.strip():
    #         st.session_state.answering_templates.append(
    #             {"id": str(uuid4()), "name": new_name.strip(), "enabled": True, "text": new_text.strip()}
    #         )
    #         st.session_state.new_tpl_name = ""
    #         st.session_state.new_tpl_text = ""
    #         st.success("Template added.")
    #     else:
    #         st.warning("Please provide both a name and content.")