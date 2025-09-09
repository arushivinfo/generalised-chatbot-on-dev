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
from search_agent_new import run_search_agent        # returns (spec, rows_text)
#sections_rt = dict(st.session_state.prompt_sections)   # shallow copy
from cache_memory import get_memory_prompt  # Use our new function that supports session_id
from response_gen import DEFAULT_PROMPT_SECTIONS, compose_prompt, get_suggested_questions
from response_gen import narrator                      # ChatOpenAI instance

from schema_registry import load_registry, get_collection_names, get_descriptions


# Default suggested questions settings
DEFAULT_SUGGESTED_QUESTIONS_PROMPT = """\
Generate relevant follow-up questions based on the user's original question and the assistant's answer. Focus on:
Questions should be short(10-12 words) and simple also highly relevant to the context according to the Question and Answer.
Make questions specific, actionable, and likely to provide valuable insights for the users.
"""

DEFAULT_SUGGESTED_QUESTIONS_COUNT = 3

# Initialize suggested questions settings
if "suggested_questions_settings" not in st.session_state:
    st.session_state.suggested_questions_settings = {
        "enabled": True,
        "custom_prompt": DEFAULT_SUGGESTED_QUESTIONS_PROMPT,
        "max_questions": DEFAULT_SUGGESTED_QUESTIONS_COUNT
    }

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
        blob = "\n\n".join("```json\n" + json.dumps(d, indent=2, default=str) + "\n```" for d in docs)
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

# Initialize session state variables before using them
if "chat" not in st.session_state: st.session_state.chat = []
if "sid"  not in st.session_state: st.session_state.sid  = str(uuid.uuid4())
# Make session ID visible in sidebar
if "display_session_id" not in st.session_state: st.session_state.display_session_id = True
# store prompt blocks so edits persist
if "prompt_sections" not in st.session_state:
    # start with the library defaults so text areas show initial values
    st.session_state.prompt_sections = DEFAULT_PROMPT_SECTIONS.copy()

with st.sidebar:
    st.markdown("<h2 style='text-align:left;'>User Panel</h2>", unsafe_allow_html=True)
    
    # Initialize user_name in session state if missing
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "team_id" not in st.session_state:
        st.session_state.team_id = ""
        
    # User inputs (keep these separate from RLS)
    user_name = st.text_input("User Name", value=st.session_state.user_name, key="user_name_sidebar")
    st.session_state.user_name = user_name
    team_id = st.text_input("Team ID", value=st.session_state.team_id, key="team_id_sidebar")
    st.session_state.team_id = team_id
    
    # Show user's collection access if they have a username
    if user_name and user_name != "anonymous":
        try:
            from schema_registry import get_user_collections
            accessible_collections = get_user_collections(user_name)
            
            if accessible_collections:
                with st.expander("Your Collection Access", expanded=False):
                    st.write("You have access to these collections:")
                    for coll in accessible_collections:
                        st.write(f"• {coll}")
            else:
                st.info("You have no specific collection access permissions.")
        except Exception as e:
            st.error(f"Error checking collection access: {e}")
    
    # RLS User ID Selection (separate from user name/team)
    st.divider()
    st.markdown("### 🔒 Row-Level Security (RLS)")
    st.caption("Select which user's data you want to see (for testing RLS)")
    
    # Get RLS configuration to show bypass roles
    try:
        from schema_registry import get_rls_config
        rls_config = get_rls_config()
        bypass_roles = rls_config.get("bypass_roles", ["admin", "super_user"])
        
        # Show current RLS settings
        if rls_config.get("enabled", False):
            st.success("🟢 RLS is enabled")
            st.caption(f"Bypass roles: {', '.join(bypass_roles)}")
        else:
            st.warning("🟡 RLS is disabled in admin settings")
            
    except Exception as e:
        st.error(f"Error loading RLS config: {e}")
        bypass_roles = ["admin", "super_user"]
    
    # User role selection
    all_roles = ["user", "customer", "employee"] + bypass_roles
    user_role = st.selectbox(
        "User Role",
        options=all_roles,
        index=0,
        help=f"Select user role. Bypass roles ({', '.join(bypass_roles)}) will see all data regardless of RLS user ID"
    )
    
    # Store user role in session state
    st.session_state.user_role = user_role
    
    # Get available user_ids from database
    try:
        from schema_registry import get_connection_config, load_registry
        from pymongo import MongoClient
        
        cfg = get_connection_config(load_registry())
        if cfg.get("mongo_uri") and cfg.get("mongo_db"):
            client = MongoClient(cfg["mongo_uri"])
            db = client[cfg["mongo_db"]]
            
            # Get unique user_ids from collections
            user_ids = set()
            if "user_leave_list" in db.list_collection_names():
                user_ids.update(db.user_leave_list.distinct("user_id"))
            # Add other collections as needed
            
            available_user_ids = sorted([str(uid) for uid in user_ids if uid is not None])
        else:
            available_user_ids = ["1", "85", "25", "42", "73"]
    except Exception as e:
        st.error(f"Could not fetch user IDs: {e}")
        available_user_ids = ["1", "85", "25", "42", "73"]
    
    # RLS User ID selection
    rls_user_id = st.selectbox(
        "RLS Filter User ID",
        options=[""] + available_user_ids,
        index=0,
        help="Select user_id for RLS filtering. Bypass roles will see all data regardless of this selection"
    )
    
    # Show what RLS will do
    if rls_user_id:
        if user_role in bypass_roles:
            st.warning(f"🔓 Role '{user_role}' bypasses RLS - will see all data")
        else:
            st.info(f"🔒 RLS Active - Only shows data where user_id = {rls_user_id}")
        
        # Store RLS user ID in session for search agent
        st.session_state.rls_user_id = rls_user_id
    else:
        st.caption("No RLS user selected - will see all data")
        st.session_state.rls_user_id = None

    # Session ID management
    st.divider()
    st.markdown("<h4>Session Management</h4>", unsafe_allow_html=True)
    
    # Import the function to get unique session IDs
    from cache_memory import get_unique_session_ids
    
    # Display current session ID with better styling
    current_sid = st.session_state.sid
    st.markdown(f"**Active Session**: `{current_sid}`")
    
    # Get available session IDs for the current user/team
    user_id = st.session_state.get("user_name", "anonymous")
    team_id = st.session_state.get("team_id", "default_team")
    available_sessions = get_unique_session_ids(user_id, team_id)
    
    # Add a visual indicator for the active session
    session_status_container = st.empty()
    if current_sid in available_sessions:
        session_status_container.success("Using saved session")
    else:
        session_status_container.info("Using new session")
    
    # Initialize selected_session variable
    selected_session = "Current Session"
    
    # Only show dropdown if there are previous sessions
    if available_sessions:
        # Add "Current" option at the top
        session_options = ["Current Session"] + available_sessions
        
        # Add key to track the previously selected session for detecting changes
        if "previous_selected_session" not in st.session_state:
            st.session_state.previous_selected_session = "Current Session"
            
        selected_session = st.selectbox(
            "Select Previous Session", 
            options=session_options,
            help="Choose a previous session to continue"
        )
        
        # Auto-load chat history when session selection changes
        if selected_session != st.session_state.previous_selected_session:
            # Save the new selection to detect future changes
            st.session_state.previous_selected_session = selected_session
            
            if selected_session == "Current Session":
                # Switching back to current session
                if st.session_state.sid != current_sid:
                    # Restore the current session ID
                    st.session_state.sid = current_sid
                    # Clear chat to start fresh
                    st.session_state.chat = []
                    st.toast("Switched to new session", icon="🆕")
                    st.rerun()
            else:
                # Loading a previous session
                from cache_memory import load_chat_history
                
                # Load chat history for this session
                chat_history = load_chat_history(user_id, team_id, selected_session)
                
                # Update chat state with loaded history and set the session ID
                if chat_history:  # Only update if we actually found chat history
                    st.session_state.chat = chat_history
                    st.session_state.sid = selected_session
                    st.toast(f"Loaded session with {len(chat_history)//2} messages", icon="✅")
                    st.rerun()  # Refresh to show the loaded chat
    else:
        st.info("No previous sessions found. Start chatting to create a new session.")
    from cache_memory import load_chat_history, get_session_details
    # Apply the selected session ID if it's not "Current Session"
    if selected_session != "Current Session" or selected_session in available_sessions:
        # Import the functions to load chat history and get session details
        from cache_memory import load_chat_history, get_session_details, rename_session, delete_session
        
        # Get session details for display
        if selected_session in available_sessions:
            # Show session management UI regardless of whether we have details
            with st.expander("Session Management", expanded=True):
                session_details = get_session_details(selected_session)
                
                # Show session details if available
                if session_details:
                    st.write(f"**Session ID:** {selected_session}")
                    #st.write(f"**Messages:** {session_details['message_count']}")
                    st.write(f"**Started:** {session_details['started'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Last active:** {session_details['last_active'].strftime('%Y-%m-%d %H:%M')}")
                    # st.write(f"**First question:** {session_details['first_query']}")
                    # st.write(f"**Last question:** {session_details['last_query']}")
                else:
                    st.write(f"**Session ID:** {selected_session}")
                    st.warning("Detailed session information not available")
                
                # Add session renaming feature - always show
                st.divider()
                st.write("**Rename Session**")
                new_session_id = st.text_input(
                    "New Session ID", 
                    key=f"rename_{selected_session}", 
                    placeholder="Enter a memorable name"
                )
                if st.button("Rename Session", key=f"rename_btn_{selected_session}"):
                    if new_session_id and new_session_id != selected_session:
                        renamed = rename_session(selected_session, new_session_id)
                        if renamed > 0:
                            st.success(f"Renamed session! {renamed} messages updated.")
                            # Update current session ID if we're renaming the active session
                            if selected_session == current_sid:
                                st.session_state.sid = new_session_id
                            st.rerun()
                        else:
                            st.error("Failed to rename session.")
                
                # Add session deletion feature - always show
                st.divider()
                st.write("**Delete Session**")
                col1, col2 = st.columns([3, 1])
                with col1:
                    confirm_delete = st.checkbox(f"Confirm deletion of session '{selected_session}'", key=f"confirm_delete_{selected_session}")
                with col2:
                    if st.button("Delete Session", key=f"delete_{selected_session}", disabled=not confirm_delete, type="primary", help="This action cannot be undone"):
                        if confirm_delete:
                            deleted = delete_session(selected_session, user_id, team_id)
                            if deleted > 0:
                                st.success(f"Deleted session! {deleted} messages removed.")
                                # If we're deleting the current session, create a new one
                                if selected_session == current_sid:
                                    st.session_state.sid = str(uuid.uuid4())
                                    st.session_state.chat = []
                                st.session_state.previous_selected_session = "Current Session"
                                st.rerun()
                            else:
                                st.error("Failed to delete session or session was empty.")        # Add button to refresh the session (in case it was updated in another tab/window)
        with st.container():
            refresh_col1, refresh_col2 = st.columns([1, 3])
            with refresh_col1:
                if st.button("🔄 Refresh", key="refresh_session_btn", help="Refresh this session"):
                    # Load chat history for this session
                    chat_history = load_chat_history(user_id, team_id, selected_session)
                    
                    # Update chat state with loaded history
                    st.session_state.chat = chat_history
                    
                    # Display a success message and reload
                    st.success(f"Refreshed session {selected_session} with {len(chat_history)//2} messages")
                    st.rerun()
    
    # Option to generate a new session ID
    st.divider()
    
    # Add new session input and button
    if "new_session_name" not in st.session_state:
        st.session_state.new_session_name = ""
    
    new_session_col1, new_session_col2 = st.columns([3, 1])
    
    with new_session_col1:
        new_session_name = st.text_input(
            "New Session Name (optional)",
            key="new_session_name_input",
            value=st.session_state.new_session_name,
            placeholder="Enter a name or leave blank for auto-ID"
        )
    
    with new_session_col2:
        if st.button("New Session", help="Start a fresh session with a new ID"):
            # Use custom name if provided, otherwise generate UUID
            if new_session_name.strip():
                st.session_state.sid = new_session_name.strip()
                st.session_state.new_session_name = ""  # Clear the input
            else:
                st.session_state.sid = str(uuid.uuid4())
            
            # Clear chat history and refresh
            st.session_state.chat = []
            st.session_state.previous_selected_session = "Current Session"
            st.toast(f"Created new session: {st.session_state.sid}", icon="🆕")
            st.rerun()
    
    # Toggle visibility of session ID in chat memory
    st.session_state.display_session_id = st.checkbox(
        "Track Sessions", 
        value=st.session_state.display_session_id,
        help="Store session ID with chat history"
    )
    
    st.divider()
    st.markdown("<h4>Chat Cache</h4>", unsafe_allow_html=True)

    # Import view_cache here or earlier if you want
    from cache_memory import view_cache, clear_cache
    
    # Create cache view filter options
    cache_filter = st.radio("Cache view filter:", 
                           ["Current Session Only", "Current User Only","All Sessions"],
                           horizontal=True)
    
    # Apply appropriate filters based on selection
    if cache_filter == "Current Session Only":
        cache_entries = view_cache(
            user_id=st.session_state.user_name, 
            team_id=st.session_state.team_id,
            session_id=st.session_state.sid
        )
        st.write(f"Current session cache ({len(cache_entries)} entries):")
    elif cache_filter == "Current User Only":
        cache_entries = view_cache(
            user_id=st.session_state.user_name, 
            team_id=st.session_state.team_id
        )
        st.write(f"User cache ({len(cache_entries)} entries):")
    else:
        cache_entries = view_cache()
        st.write(f"All cache entries ({len(cache_entries)} total):")
        
    if cache_entries:
        st.write(cache_entries)  # Show latest entry
        
    # Add button to clear filtered cache
    if st.button("Clear Filtered Cache", help="Clear cache based on current filter"):
        if cache_filter == "All Sessions":
            cleared = clear_cache()
            
        elif cache_filter == "Current User Only":
            cleared = clear_cache(
                user_id=st.session_state.user_name, 
                team_id=st.session_state.team_id
            )
        else:
            cleared = clear_cache(
                user_id=st.session_state.user_name, 
                team_id=st.session_state.team_id,
                session_id=st.session_state.sid
            )
            
        st.success(f"Cleared {cleared} cache entries")
        st.rerun()

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
# Session state variables are now initialized at the top of the file
# if "mem_agent" not in st.session_state:
#     from memory_agent import MemoryAgent
#     st.session_state.mem_agent = MemoryAgent(k=5)

# --- 4.  Header ---------------------------------------------------
st.markdown('<h1 class="title">Data QA Chat</h1>', unsafe_allow_html=True)

if st.button("Clear Chat"): st.session_state.chat = []; st.rerun()

chat_tab, prompt_tab = st.tabs(["Chat", "Prompt Settings"])
import time
with chat_tab:
    t1= time.time()
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    
    # Clear the skip_suggestions flag for new interactions
    if "skip_suggestions" in st.session_state:
        del st.session_state.skip_suggestions

    # Debug: Check if we have a pending question
    if st.session_state.pending_question:
        print(f"Found pending question: {st.session_state.pending_question}")
    else:
        print("No pending question found")
        
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

    # Always show the chat input, but handle pending questions first
    chat_input_q = st.chat_input("Ask about players, venues, fantasy picks…")
    
    # Determine which question to process
    if st.session_state.pending_question:
        q = st.session_state.pending_question
        print(f"Processing pending question: {q}")
        st.session_state.pending_question = None  # Clear immediately
        print("Cleared pending_question")
    else:
        q = chat_input_q
        if q:
            print(f"Got question from chat input: {q}")


    if q:
        st.session_state.chat.append(("user", q))
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            resp_container = st.empty()

            step_box   = st.expander("Intermediate steps", expanded=False)
            step_cb    = StreamlitStepHandler(step_box)
            if st.session_state.get("pending_question"):
                del st.session_state.pending_question


            # STEP 0 – **no rewrite**: just use the original question
            standalone_q = q
            step_box.markdown("**Step**: Using original question (no rewrite) 🔍", unsafe_allow_html=True)

            # STEP 1 – structured search on the rewritten question
            with st.spinner("Planning and retrieving documents 📂️..."):
                user_id = st.session_state.get("user_name", "anonymous")
                team_id = st.session_state.get("team_id", "default_team")
                # Get session ID if tracking sessions
                session_id = st.session_state.sid if st.session_state.display_session_id else None
                spec, rows_text, dbg = run_search_agent(user_id, team_id, standalone_q, callbacks=[step_cb], session_id=session_id)
                step_cb.render()  # show steps up to now

            # Check if the user attempted to access restricted collections
            attempted_restricted = False
            if rows_text and rows_text.startswith("⚠️ Access denied:"):
                attempted_restricted = True
                st.warning(rows_text)
                
            # Prepare debug info
            dbg_box = st.expander("Debug info", expanded=False)
            debug_content = {
                "Query Spec": spec,  # Includes filters, sort, and limit
                "MongoDB Filters": dbg.get("filters", []),
                "Chosen Collections": dbg.get("chosen_collections", []),
                "Restricted Collections": dbg.get("restricted_collections", []),
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

                # 1) Build a structured rows blob from chosen collections (no raw JSON)
                rows_clean = build_rows_for_prompt(dbg)
                
                # Check for access denied messages
                access_denied = rows_text and rows_text.startswith("⚠️ Access denied:")

                if access_denied:
                    # Use a custom error message for access denied
                    answer = f"{rows_text}\n\nPlease contact an administrator if you need access to this collection."
                    resp_container.markdown(answer)
                    st.session_state.chat.append(("assistant", answer))
                    
                    # Save memory to cache
                    from cache_memory import save_to_cache
                    session_id = st.session_state.sid if st.session_state.display_session_id else None
                    save_to_cache(q, answer, user_id, team_id, session_id=session_id)
                    st.session_state.skip_suggestions = True
                else:
                    # Only execute this block if access is not denied
                    # 2) Append enabled "Answering Templates" ONLY into PROMPT_FORMATTING
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
                    # Get session ID if tracking sessions
                    session_id = st.session_state.sid if st.session_state.display_session_id else None
                    
                    prompt = compose_prompt(
                        sections_rt,  # make sure to use sections_rt, not original
                        question=q,
                        rows=rows_clean,
                        language=lang,
                        memory_context=get_memory_prompt(3, user_id, team_id, session_id),
                        source_query=dbg.get("filters", )  # get last 3 memories
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
                    session_id = st.session_state.sid if st.session_state.display_session_id else None
                    answer_for_memory = answer
                    import re
                    answer_for_memory = re.sub(r'```[\s\S]*?```', '', answer)
                    answer_for_memory = re.sub(r'Source Query:[\s\S]*$', '', answer_for_memory).strip()
                    save_to_cache(q, answer_for_memory, user_id, team_id, session_id=session_id)

            # NEW: Generate and display suggested questions (only if enabled and access wasn't denied)
            if st.session_state.suggested_questions_settings.get("enabled", True) and not st.session_state.get("skip_suggestions", False):
                with st.spinner("Generating suggested questions..."):
                    try:
                        # Use custom settings for suggested questions
                        custom_prompt = st.session_state.suggested_questions_settings.get("custom_prompt", DEFAULT_SUGGESTED_QUESTIONS_PROMPT)
                        max_questions = st.session_state.suggested_questions_settings.get("max_questions", DEFAULT_SUGGESTED_QUESTIONS_COUNT)
                        
                        # Pass custom prompt to get_suggested_questions function
                        # Note: You may need to modify get_suggested_questions to accept custom_prompt parameter
                        suggested_questions = get_suggested_questions(
                            q, answer, 
                            max_questions=max_questions,
                            custom_prompt=custom_prompt  # This parameter may need to be added to the function
                        )
                        
                        if suggested_questions:
                            st.markdown("### 💡 You might also want to ask:")
                            
                            # Create columns for better layout
                            cols = st.columns(len(suggested_questions))
                            
                            # Create a callback function for button clicks
                            def set_pending_question(suggestion_text):
                                st.session_state.pending_question = suggestion_text
                                print(f"Callback: Set pending_question to: {suggestion_text}")
                            
                            for i, suggestion in enumerate(suggested_questions):
                                print(f"Suggestion {i}: {suggestion}")  # Debugging line
                                with cols[i]:
                                    # Use a unique key for each button to avoid conflicts
                                    button_key = f"suggest_{len(st.session_state.chat)}_{i}_{hash(suggestion)}"
                                    if st.button(
                                        f"❓ {suggestion}", 
                                        key=button_key, 
                                        help="Click to ask this question",
                                        on_click=set_pending_question,
                                        args=(suggestion,)
                                    ):
                                        print(f"Button clicked for suggestion: {suggestion}")  # This will execute
                                        # The callback will handle setting the pending question
                                        pass  # Remove st.rerun() from here
                    except Exception as e:
                        st.error(f"Error generating suggestions: {e}")
                    t2 = time.time()
                    print(f"Total time for processing question: {t2 - t1:.6f} seconds")

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
            value=st.session_state.suggested_questions_settings.get("max_questions", DEFAULT_SUGGESTED_QUESTIONS_COUNT),
            help="How many suggested questions to display"
        )
        
        # Custom prompt for suggested questions
        st.markdown("#### Custom Prompt for Suggested Questions")
        if "suggested_questions_prompt_key" not in st.session_state:
            st.session_state.suggested_questions_prompt_key = st.session_state.suggested_questions_settings.get("custom_prompt", DEFAULT_SUGGESTED_QUESTIONS_PROMPT)
        
        custom_prompt = st.text_area(
            "Instructions for generating suggested questions",
            key="suggested_questions_prompt_key",
            height=200,
            help="This prompt will be used to generate follow-up questions. Be specific about the type of questions you want."
        )
        st.session_state.suggested_questions_settings["custom_prompt"] = custom_prompt
        
        # Save button for suggested questions settings
        if st.button("Save Suggested Questions Settings"):
            st.success("Suggested questions settings updated")
        
        

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