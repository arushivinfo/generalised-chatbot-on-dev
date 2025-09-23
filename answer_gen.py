import re
import json
import time
import streamlit as st
from typing import Dict, Any, Tuple, Optional, List
from uuid import uuid4

# Import required modules
from search_agent_new import run_search_agent
from response_gen import compose_prompt, get_suggested_questions
from cache_memory import save_to_cache, get_memory_prompt
from llm_services import call_narrator_model
from lang_detect import LangDetectAgent
from schema_registry import get_response_prompt_config

# Move StreamlitStepHandler here to avoid circular import
class StreamlitStepHandler:
    """Step handler for Streamlit UI"""
    def __init__(self, step_box):
        self.step_box = step_box
        self.steps = []
    
    def __call__(self, step_text: str):
        """Add a step to the display"""
        self.steps.append(step_text)
        self.render()
    
    def render(self):
        """Render all steps in the step box"""
        if self.steps:
            content = "\n\n".join(f"**Step {i+1}**: {step}" for i, step in enumerate(self.steps))
            self.step_box.markdown(content, unsafe_allow_html=True)

class ResponseHandler:
    """Handles the complete flow from question to response and suggested questions."""
    
    def __init__(self):
        self.lang_detector = LangDetectAgent()
    
    def load_admin_prompt_settings(self) -> bool:
        """Load prompt settings from admin configuration."""
        try:
            admin_config = get_response_prompt_config()
            
            # Initialize session state if needed
            if "prompt_sections" not in st.session_state:
                from default_prompts import get_default_prompt_sections
                st.session_state.prompt_sections = get_default_prompt_sections()
            
            if "suggested_questions_settings" not in st.session_state:
                from default_prompts import get_default_suggested_questions_settings
                st.session_state.suggested_questions_settings = get_default_suggested_questions_settings()
            
            if "answering_templates" not in st.session_state:
                from default_prompts import get_default_answering_templates
                st.session_state.answering_templates = get_default_answering_templates()
            
            # Update prompt sections
            if admin_config.get("prompt_sections"):
                st.session_state.prompt_sections = admin_config["prompt_sections"]
            
            # Update suggested questions settings
            if admin_config.get("suggested_questions_settings"):
                st.session_state.suggested_questions_settings = admin_config["suggested_questions_settings"]
            
            # Update answering templates
            st.session_state.answering_templates = admin_config.get("answering_templates", [])
            
            return True
        except Exception as e:
            print(f"Error loading admin prompt settings: {e}")
            return False
    
    def build_rows_for_prompt(self, dbg: dict) -> str:
        """Render full objects (pretty JSON) per collection."""
        cols_order = dbg.get("chosen_collections", []) or []
        results = dbg.get("results", []) or []
        blocks = []
        
        for i, coll in enumerate(cols_order):
            res = results[i] if i < len(results) else {}
            docs = res.get("docs", []) or []
            if not docs:
                blocks.append(f"# {coll}\n(no rows)")
                continue
            blob = "\n\n".join("```json\n" + json.dumps(d, indent=2, default=str) + "\n```" for d in docs)
            blocks.append(f"# {coll}\n{blob}")
        
        return "\n\n".join(blocks) or "(no rows)"
    
    def generate_debug_content(self, spec: dict, dbg: dict, rows_text: str) -> dict:
        """Generate debug information for the query."""
        # Get bypass roles for RLS debug info
        try:
            from schema_registry import get_rls_config
            rls_config = get_rls_config()
            bypass_roles = rls_config.get("bypass_roles", ["admin", "super_user"])
        except:
            bypass_roles = ["admin", "super_user"]
        
        user_role = st.session_state.get("user_role", "user")
        
        return {
            "Query Spec": spec,
            "MongoDB Filters": dbg.get("filters", []),
            "Chosen Collections": dbg.get("chosen_collections", []),
            "Restricted Collections": dbg.get("restricted_collections", []),
            "RLS Status": {
                "rls_values": st.session_state.get("rls_values", {}),
                "user_role": user_role,
                "bypass_active": user_role in bypass_roles
            },
            "Raw Results": rows_text[:500] + "..." if len(rows_text) > 500 else rows_text
        }
    
    def process_access_denied(self, rows_text: str, question: str, user_id: str, team_id: str, session_id: Optional[str]) -> str:
        """Handle access denied scenarios."""
        answer = f"{rows_text}\n\nPlease contact an administrator if you need access to this collection."
        
        # Clean the answer for memory - remove code blocks and source query sections
        answer_for_memory = re.sub(r'```[\s\S]*?```', '', answer)
        answer_for_memory = re.sub(r'Source Query[:\s]*[\s\S]*$', '', answer_for_memory, flags=re.IGNORECASE).strip()
        answer_for_memory = re.sub(r'\{[^{}]*\}', '', answer_for_memory)
        
        # Save to cache
        save_to_cache(question, answer_for_memory, user_id, team_id, session_id=session_id)
        
        # Skip suggestions for access denied
        st.session_state.skip_suggestions = True
        
        return answer
    
    def generate_response(self, question: str, rows_clean: str, lang: str, user_id: str, team_id: str, session_id: Optional[str]) -> Tuple[str, List[dict]]:
        """Generate the AI response using the composed prompt."""
        # Load latest admin prompt settings
        self.load_admin_prompt_settings()
        
        # Append enabled "Answering Templates" ONLY into formatting section
        sections_rt = dict(st.session_state.prompt_sections)  # shallow copy
        
        formatting_text = sections_rt.get("formatting", "")
        
        enabled_templates = [
            t["text"] for t in st.session_state.answering_templates
            if t.get("enabled") and t.get("text", "").strip()
        ]
        
        if enabled_templates:
            formatting_text = (formatting_text.rstrip() + "\n\n" + "\n\n".join(enabled_templates)).strip()
        
        sections_rt["formatting"] = formatting_text
        
        # Compose final prompt with structured rows + augmented formatting
        prompt = compose_prompt(
            sections_rt,
            question=question,
            rows=rows_clean,
            language=lang,
            memory_context=get_memory_prompt(3, user_id, team_id, session_id)
        )
        
        print(f"Final prompt composed: {prompt[:200]}...")
        
        # Convert to proper message format for call_narrator_model
        messages = [{"role": "user", "content": prompt}]
        
        return prompt, messages
    
    def stream_response(self, messages: List[dict], resp_container) -> str:
        """Stream the AI response and return the complete answer."""
        # Stream the answer
        stream = call_narrator_model(messages, stream=True)
        print("Streaming response...")
        
        answer = ""
        for chunk in stream:
            if chunk:  # Only process non-empty chunks
                answer += chunk
                if resp_container:
                    resp_container.markdown(answer)
        
        return answer
    
    def generate_suggested_questions(self, question: str, answer: str, rows_clean: str) -> Optional[List[str]]:
        """Generate suggested follow-up questions."""
        # Load latest admin settings for suggested questions
        self.load_admin_prompt_settings()
        
        if not st.session_state.suggested_questions_settings.get("enabled", True):
            return None
        
        if st.session_state.get("skip_suggestions", False):
            return None
        
        try:
            custom_prompt = st.session_state.suggested_questions_settings.get("custom_prompt", "")
            max_questions = st.session_state.suggested_questions_settings.get("max_questions", 3)
            
            suggested_questions = get_suggested_questions(
                question, answer, Row_data=rows_clean,
                max_questions=max_questions,
                custom_prompt=custom_prompt
            )
            
            return suggested_questions
            
        except Exception as e:
            st.error(f"Error generating suggestions: {e}")
            return None
    
    def display_suggested_questions(self, suggested_questions: List[str]):
        """Display suggested questions with clickable buttons."""
        if not suggested_questions:
            return
        
        st.markdown("### 💡 You might also want to ask:")
        
        # Create columns for better layout
        cols = st.columns(len(suggested_questions))
        
        # Create a callback function for button clicks
        def set_pending_question(suggestion_text):
            st.session_state.pending_question = suggestion_text
            print(f"Callback: Set pending_question to: {suggestion_text}")
        
        for i, suggestion in enumerate(suggested_questions):
            print(f"Suggestion {i}: {suggestion}")
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
                    print(f"Button clicked for suggestion: {suggestion}")

def process_question_complete(
    question: str, 
    resp_container, 
    step_box, 
    dbg_box, 
    prompt_debug
) -> Tuple[str, bool]:
    """
    Complete question processing pipeline from search to suggested questions.
    
    Args:
        question: User's question
        resp_container: Streamlit container for response
        step_box: Streamlit expander for intermediate steps
        dbg_box: Streamlit expander for debug info
        prompt_debug: Streamlit expander for prompt debug
    
    Returns:
        Tuple of (answer, success_flag)
    """
    start_time = time.time()
    handler = ResponseHandler()
    
    # Clear skip suggestions flag for new interactions
    if "skip_suggestions" in st.session_state:
        del st.session_state.skip_suggestions
    
    try:
        # Get user context
        user_id = st.session_state.get("user_name", "anonymous")
        team_id = st.session_state.get("team_id", "default_team")
        session_id = st.session_state.sid if st.session_state.display_session_id else None
        
        # Step 1: Use original question (no rewrite)
        standalone_q = question
        step_box.markdown("**Step**: Using original question (no rewrite) 🔍", unsafe_allow_html=True)
        
        # Step 2: Structured search on the question
        with st.spinner("Planning and retrieving documents 📂️..."):
            step_cb = StreamlitStepHandler(step_box)
            spec, rows_text, dbg = run_search_agent(user_id, team_id, standalone_q, callbacks=[step_cb], session_id=session_id)
            step_cb.render()
        
        # Check for access restrictions
        access_denied = rows_text and rows_text.startswith("⚠️ Access denied:")
        if access_denied:
            st.warning(rows_text)
        
        # Generate and display debug info
        debug_content = handler.generate_debug_content(spec, dbg, rows_text)
        dbg_box.code(json.dumps(debug_content, indent=2, default=str), language="json")
        
        # Step 3: Generate narrative answer
        with st.spinner("Generating your answer 🔥..."):
            # Detect language
            lang = handler.lang_detector.detect_language(question)
            print(f'Detected language: {lang}')
            
            # Build structured rows from chosen collections
            rows_clean = handler.build_rows_for_prompt(dbg)
            
            if access_denied:
                # Handle access denied
                answer = handler.process_access_denied(rows_text, question, user_id, team_id, session_id)
                resp_container.markdown(answer)
                return answer, False
            
            else:
                # Generate normal response
                prompt, messages = handler.generate_response(question, rows_clean, lang, user_id, team_id, session_id)
                
                # Display debug info
                prompt_debug.code(prompt, language="markdown")
                
                # Stream the response
                answer = handler.stream_response(messages, resp_container)
                
                # Save to cache
                save_to_cache(question, answer, user_id, team_id, session_id=session_id)
                
                # Step 4: Generate suggested questions
                with st.spinner("Generating suggested questions..."):
                    suggested_questions = handler.generate_suggested_questions(question, answer, rows_clean)
                    if suggested_questions:
                        handler.display_suggested_questions(suggested_questions)
                
                end_time = time.time()
                print(f"Total time for processing question: {end_time - start_time:.6f} seconds")
                
                return answer, True
                
    except Exception as e:
        st.error(f"Error processing question: {e}")
        print(f"Error in process_question_complete: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}", False