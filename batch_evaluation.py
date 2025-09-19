import os
from dotenv import load_dotenv

# Force reload environment variables and override any cached values
load_dotenv(override=True)

# Debug: Print what API key is being loaded
print(f"DEBUG: API Key ending: {os.getenv('OPENAI_API_KEY', 'NOT_FOUND')[-10:]}")

# If it's still the wrong key, force set it manually
current_key = os.getenv('OPENAI_API_KEY', '')
if current_key.endswith('8OMA'):
    print("WARNING: Detected old API key, forcing update...")
    os.environ['OPENAI_API_KEY'] = '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'
 
    print(f"Updated API Key ending: {os.getenv('OPENAI_API_KEY', 'NOT_FOUND')[-10:]}")

# Rest of your imports...
import re
import streamlit as st
import pandas as pd
import json
import time
import datetime
import os
import sys
from typing import List, Dict, Any, Optional, Tuple  # Added Tuple here

# Add this at the top to prevent page config conflicts when importing other modules
os.environ["STREAMLIT_DISABLE_SET_PAGE_CONFIG"] = "true"

# Import the centralized response handler
from answer_gen import ResponseHandler

from schema_registry import load_registry, get_descriptions, get_response_prompt_config
from schema_registry import load_registry, get_collection_names, get_descriptions

def _coll_meta_from_registry():
    reg = load_registry()
    names = get_collection_names(reg)       # {'matches': '<coll>', ...}
    desc = get_descriptions(reg)        # {'<coll>': 'short description', ...}
    return {name: (name, (desc.get(name) or "no description")) for name in names}

import pandas as pd
COLL_META = _coll_meta_from_registry()

class BatchResponseHandler(ResponseHandler):
    """Modified ResponseHandler for batch processing without suggested questions and memory."""
    
    def __init__(self):
        super().__init__()
    
    def generate_response(self, question: str, rows_clean: str, lang: str, user_id: str = 'admin', team_id: str = 'admin', session_id: Optional[str] = 'admin') -> Tuple[str, List[dict]]:
        """Generate the AI response using the composed prompt - batch version without memory."""
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
        
        # Import compose_prompt here to avoid circular imports
        from response_gen import compose_prompt
        
        # Compose final prompt with structured rows + augmented formatting (NO MEMORY for batch)
        prompt = compose_prompt(
            sections_rt,
            question=question,
            rows=rows_clean,
            language=lang,
            memory_context=""  # Empty memory context for batch processing
        )
        
        print(f"Final prompt composed: {prompt[:200]}...")
        
        # Convert to proper message format for call_narrator_model
        messages = [{"role": "user", "content": prompt}]
        
        return prompt, messages
    
    def stream_response_batch(self, messages: List[dict]) -> str:
        """Non-streaming version for batch processing."""
        from llm_services import call_narrator_model
        
        # Get answer directly without streaming
        answer = call_narrator_model(messages, stream=False)
        print("Batch response generated...")
        
        return answer

def process_question_batch(question: str) -> Tuple[str, bool, dict, str]:
    """
    Simplified question processing for batch evaluation using centralized ResponseHandler.
    
    Args:
        question: User's question
    
    Returns:
        Tuple of (answer, success_flag, debug_info, spec)
    """
    start_time = time.time()
    handler = BatchResponseHandler()
    
    try:
        # Get user context (fixed for batch)
        user_id = 'admin'
        team_id = 'admin' 
        session_id = 'admin'
        
        # Step 1: Structured search on the question
        from search_agent_new import run_search_agent
        
        spec, rows_text, dbg = run_search_agent(user_id, team_id, question, callbacks=[], session_id=session_id)
        
        # Check for access restrictions
        access_denied = rows_text and rows_text.startswith("⚠️ Access denied:")
        if access_denied:
            return f"{rows_text}\n\nPlease contact an administrator if you need access to this collection.", False, dbg, spec
        
        # Step 2: Generate narrative answer
        # Detect language
        lang = handler.lang_detector.detect_language(question)
        print(f'Detected language: {lang}')
        
        # Build structured rows from chosen collections
        rows_clean = handler.build_rows_for_prompt(dbg)
        
        # Generate response
        prompt, messages = handler.generate_response(question, rows_clean, lang, user_id, team_id, session_id)
        
        # Get the response (non-streaming for batch)
        answer = handler.stream_response_batch(messages)
        
        # NO cache saving for batch processing
        # NO suggested questions for batch processing
        
        end_time = time.time()
        print(f"Total time for processing question: {end_time - start_time:.6f} seconds")
        
        return answer, True, dbg, spec
        
    except Exception as e:
        print(f"Error in process_question_batch: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}", False, {}, {}

def process_query_for_batch(question):
    """
    Updated version using centralized ResponseHandler.
    """
    try:
        # Use the centralized batch processing function
        answer, success, dbg, spec = process_question_batch(question)
        
        # Build debug info for compatibility
        debug_info = {
            "filters": dbg.get("filters", []),
            "chosen_collections": dbg.get("chosen_collections", []),
            "restricted_collections": dbg.get("restricted_collections", [])
        }
        
        # Build rows text for compatibility
        handler = BatchResponseHandler()
        rows_text = handler.build_rows_for_prompt(dbg)
        
        return answer, spec, debug_info, rows_text, dbg, []
    
    except Exception as e:
        # Handle errors
        return f"Error: {str(e)}", {}, {"error": str(e)}, "", {}, []

def ensure_page_config():
    """Only set page config if this is run as main script"""
    if __name__ == "__main__":
        # Remove the environment variable if running as main script
        if "STREAMLIT_DISABLE_SET_PAGE_CONFIG" in os.environ:
            del os.environ["STREAMLIT_DISABLE_SET_PAGE_CONFIG"]
            
        st.set_page_config(
            page_title="Batch Evaluation",
            page_icon="🔍",
            layout="wide",
        )

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_question(question, idx, total):
    """Helper to process a single question safely"""
    try:
        start_time = time.time()
        answer, spec, debug_info, rows_text, dbg, _ = process_query_for_batch(question)
        end_time = time.time()

        answer_str = str(answer).strip() if answer else ""

        failure_patterns = [
            "No relevant data found for this query",
            "Please specify players, matches, or metrics",
            "I don't have enough information",
            "I don't have data on",
            "No information available",
            "I don't have specific information",
            "unable to find relevant data",
            "Please specify players, matches, or metrics related to the given fixture"
        ]
        contains_failure_pattern = any(
            pattern.lower() in answer_str.lower() 
            for pattern in failure_patterns
        ) if answer_str else False

        is_success = bool(
            answer_str and 
            len(answer_str) > 10 and
            not answer_str.startswith("⚠️") and 
            not answer_str.startswith("Error:") and
            not contains_failure_pattern
        )

        return {
            "question": question,
            "answer": answer_str,
            "spec": spec,
            "debug_info": debug_info,
            "rows_text": rows_text,
            "success": is_success,
            "time_taken": round(end_time - start_time, 2),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        return {
            "question": question,
            "answer": f"ERROR: {str(e)}",
            "spec": {},
            "debug_info": {"error": str(e)},
            "rows_text": "",
            "success": False,
            "time_taken": 0,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def process_all_questions_parallel(questions: List[str]) -> List[Dict[str, Any]]:
    """Process all questions in parallel without batching"""
    total = len(questions)
    results = []

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    completed_count = 0

    status_text.text(f"Processing {total} questions in parallel using centralized ResponseHandler...")

    # Run all questions in parallel with automatic worker management
    max_workers = min(total, 20)  # Automatically set workers based on question count, max 20
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_question, question, idx, total): (idx, question)
            for idx, question in enumerate(questions)
        }

        for future in as_completed(future_to_idx):
            idx, q = future_to_idx[future]
            try:
                result = future.result()
                results.append((idx, result))  # Keep track of original index
                completed_count += 1
                
                # Update progress bar as each completes
                progress_bar.progress(completed_count / total)
                status_text.text(f"Completed {completed_count}/{total} questions...")
                
            except Exception as e:
                result = {
                    "question": q,
                    "answer": f"ERROR: {str(e)}",
                    "spec": {},
                    "debug_info": {"error": str(e)},
                    "rows_text": "",
                    "success": False,
                    "time_taken": 0,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                results.append((idx, result))
                completed_count += 1
                progress_bar.progress(completed_count / total)

    # Sort results by original index to maintain question order
    results.sort(key=lambda x: x[0])
    final_results = [result for _, result in results]

    progress_bar.progress(1.0)
    status_text.text(f"Completed processing all {total} questions using centralized system!")
    return final_results

def render_batch_evaluation_ui():
    """Render the batch evaluation UI"""
    # Only initialize session state if needed
    if "eval_questions" not in st.session_state:
        st.session_state.eval_questions = []
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = []
    if "eval_running" not in st.session_state:
        st.session_state.eval_running = False
    if "eval_complete" not in st.session_state:
        st.session_state.eval_complete = False
    
    # Display info about centralized system
    st.info("🎯 **Using Centralized Response System**: This batch evaluation now uses the same response generation pipeline as the main chat interface, ensuring consistent results.")
    
    # Step 1: Load Questions
    st.header("Step 1: Load Questions")
    upload_col1, upload_col2 = st.columns([1, 1])
    
    with upload_col1:
        questions_file = st.file_uploader("Upload CSV or JSON file", type=["csv", "json"])
        
        if questions_file is not None:
            try:
                if questions_file.name.endswith('.csv'):
                    questions_df = pd.read_csv(questions_file)
                    if 'question' not in questions_df.columns:
                        st.error("CSV file must have a 'question' column")
                    else:
                        st.session_state.eval_questions = questions_df['question'].tolist()
                else:  # JSON
                    questions_data = json.load(questions_file)
                    if isinstance(questions_data, list):
                        if all(isinstance(q, str) for q in questions_data):
                            st.session_state.eval_questions = questions_data
                        elif all(isinstance(q, dict) and 'question' in q for q in questions_data):
                            st.session_state.eval_questions = [q['question'] for q in questions_data]
                        else:
                            st.error("JSON must be a list of questions or objects with a 'question' field")
                    else:
                        st.error("JSON must be a list of questions")
                
                st.success(f"Loaded {len(st.session_state.eval_questions)} questions")
            except Exception as e:
                st.error(f"Error loading questions: {str(e)}")
    
    with upload_col2:
        st.write("Or enter questions manually:")
        manual_questions = st.text_area("One question per line", height=150)
        
        if st.button("Add Questions"):
            new_questions = [q.strip() for q in manual_questions.split('\n') if q.strip()]
            if new_questions:
                st.session_state.eval_questions.extend(new_questions)
                st.success(f"Added {len(new_questions)} questions")
    
    # Display loaded questions
    if st.session_state.eval_questions:
        with st.expander(f"Preview Questions ({len(st.session_state.eval_questions)})"):
            for i, q in enumerate(st.session_state.eval_questions):
                st.write(f"{i+1}. {q}")
    
    # Step 2: Execution Controls
    st.header("Step 2: Run Evaluation")
    
    if st.session_state.eval_complete:
        st.success("✅ Evaluation completed!")
        
        # Display metrics
        total = len(st.session_state.eval_results)
        success_count = sum(1 for r in st.session_state.eval_results if r.get("success", False))
        fail_count = total - success_count
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Total Questions", total)
        metrics_col2.metric("Successful", success_count)
        metrics_col3.metric("Failed", fail_count)
        
        # Add failure analysis
        if fail_count > 0:
            st.subheader("Failure Analysis")
            failure_types = {}
            for result in st.session_state.eval_results:
                if not result.get("success", False):
                    answer = result.get("answer", "")
                    if "No relevant data found" in answer:
                        failure_types["No Data Found"] = failure_types.get("No Data Found", 0) + 1
                    elif "Please specify players, matches, or metrics" in answer:
                        failure_types["Needs Clarification"] = failure_types.get("Needs Clarification", 0) + 1
                    elif "JSON parse error" in answer or "'filters'" in answer:
                        failure_types["JSON Parse Error"] = failure_types.get("JSON Parse Error", 0) + 1
                    elif "Search failed" in answer:
                        failure_types["Search Failed"] = failure_types.get("Search Failed", 0) + 1
                    elif answer.startswith("Error:"):
                        failure_types["System Error"] = failure_types.get("System Error", 0) + 1
                    else:
                        failure_types["Other"] = failure_types.get("Other", 0) + 1
            
            failure_cols = st.columns(len(failure_types))
            for i, (failure_type, count) in enumerate(failure_types.items()):
                failure_cols[i].metric(failure_type, count)
        
        if st.button("Reset and Start New Evaluation"):
            st.session_state.eval_results = []
            st.session_state.eval_running = False
            st.session_state.eval_complete = False
            st.rerun()
    
    elif st.session_state.eval_running:
        st.info("Evaluation in progress... Please wait.")
    
    else:
        # Simple start button without any configuration options
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:  # Center the button
            if st.button("🚀 Start Evaluation", disabled=len(st.session_state.eval_questions) == 0, 
                        help=f"Process all {len(st.session_state.eval_questions)} questions using centralized ResponseHandler"):
                if st.session_state.eval_questions:
                    try:
                        st.session_state.eval_running = True
                        
                        # Process all questions in parallel using centralized system
                        results = process_all_questions_parallel(st.session_state.eval_questions)
                        
                        # Update session state
                        st.session_state.eval_results = results
                        st.session_state.eval_complete = True
                    finally:
                        st.session_state.eval_running = False
                    
                    # Rerun once at the end
                    st.rerun()
        
        with col3:
            if st.button("🗑️ Clear Questions"):
                st.session_state.eval_questions = []
                st.session_state.eval_results = []
                st.success("Questions cleared")

    # Rest of the function remains the same...
    # Step 3: Results
    st.header("Step 3: View Results")
    
    if st.session_state.eval_results:
        # Export options
        export_col1, export_col2 = st.columns(2)
        
        with export_col1:
            # Export full results as JSON
            json_str = json.dumps(st.session_state.eval_results, indent=2, default=str)
            st.download_button(
                label="Export Full Results (JSON)",
                data=json_str,
                file_name=f"eval_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
        
        with export_col2:
            # Export summary as CSV
            summary_df = pd.DataFrame([{
                "question": r["question"],
                "success": r["success"],
                "time_taken": r.get("time_taken", 0),
                "answer_length": len(r["answer"]) if isinstance(r["answer"], str) else 0,
                "timestamp": r.get("timestamp", "")
            } for r in st.session_state.eval_results])
            
            csv_data = summary_df.to_csv(index=False)
            st.download_button(
                label="Export Summary (CSV)",
                data=csv_data,
                file_name=f"eval_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        # Display individual results
        st.subheader("Detailed Results")
        
        # Add filter options
        filter_type = st.radio("Filter results", ["All", "Successful", "Failed"], horizontal=True)
        
        filtered_results = st.session_state.eval_results
        if filter_type == "Successful":
            filtered_results = [r for r in filtered_results if r.get("success", False)]
        elif filter_type == "Failed":
            filtered_results = [r for r in filtered_results if not r.get("success", False)]
        
        # Display each result in an expander
        for i, result in enumerate(filtered_results):
            success_icon = "✅" if result.get("success", False) else "❌"
            
            # Add failure reason analysis
            failure_reason = ""
            if not result.get("success", False):
                answer = result.get("answer", "")
                if "No relevant data found" in answer:
                    failure_reason = " [No Data Found]"
                elif "Please specify players, matches, or metrics" in answer:
                    failure_reason = " [Needs Clarification]"
                elif "JSON parse error" in answer or "'filters'" in answer:
                    failure_reason = " [JSON Parse Error]"
                elif "Search failed" in answer:
                    failure_reason = " [Search Failed]"
                elif answer.startswith("Error:"):
                    failure_reason = " [System Error]"
                else:
                    failure_reason = " [Other Error]"
            
            with st.expander(f"{success_icon} Q{i+1}{failure_reason}: {result['question']}"):
                st.markdown("**Question:**")
                st.write(result["question"])
                
                st.markdown("**Answer:**")
                st.write(result["answer"])
                
                st.markdown("**Metrics:**")
                st.write(f"Status: {'Success' if result.get('success', False) else 'Failed'}")
                if failure_reason:
                    st.write(f"Failure Type: {failure_reason.strip('[]')}")
                st.write(f"Time: {result.get('time_taken', 'N/A')} seconds")
                
                # Show search specification
                st.markdown("**Search Specification:**")
                debug_content = {
                    "Query Spec": result["spec"],
                    "MongoDB Filters": result["debug_info"].get("filters", []),
                    "Chosen Collections": result["debug_info"].get("chosen_collections", [])
                }
                st.json(debug_content)
    else:
        st.info("No results yet. Run an evaluation to see results here.")

if __name__ == "__main__":
    ensure_page_config()
    render_batch_evaluation_ui()