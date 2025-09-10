import re
import streamlit as st
import pandas as pd
import json
import time
import datetime
import os
import sys
from typing import List, Dict, Any, Optional
import openai
from response_gen import DEFAULT_PROMPT_SECTIONS,PROMPT_Q_AND_ROWS

# Add this at the top to prevent page config conflicts when importing other modules
os.environ["STREAMLIT_DISABLE_SET_PAGE_CONFIG"] = "true"
from schema_registry import load_registry, get_descriptions
from schema_registry import load_registry, get_collection_names, get_descriptions
def _coll_meta_from_registry():
    reg = load_registry()
    names = get_collection_names(reg)       # {'matches': '<coll>', ...}
    desc = get_descriptions(reg)        # {'<coll>': 'short description', ...}
    return {name: (name, (desc.get(name) or "no description")) for name in names}


import pandas as pd
COLL_META = _coll_meta_from_registry()
# Direct implementation of essential functions without importing frontend.py
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
import openai
from uuid import uuid4
def process_query_for_batch(question):
    """
    Simplified version of process_query_core that doesn't require importing frontend.py

    """
    
    try:
        # Import necessary components directly
        from search_agent_new import run_search_agent,get_memory_prompt
        from response_gen import narrator, compose_prompt
        from cache_memory import  save_to_cache
        from langchain_core.messages import HumanMessage
        from lang_detect import LangDetectAgent
        
        if "answering_templates" not in st.session_state:
            # Start empty; you can add any generic templates from the UI
            st.session_state.answering_templates = []
        # Create debug dict
        debug_info = {}
        callbacks = []
        if "prompt_sections" not in st.session_state:
            # start with the library defaults so text areas show initial values
            st.session_state.prompt_sections = DEFAULT_PROMPT_SECTIONS.copy()
        # Get memory context
        standalone_q = question
        # STEP 1 — Run search agent to get structured data
        spec, rows_text, dbg = run_search_agent('admin', 'admin', standalone_q, session_id='admin', callbacks=[])
        
        # STEP 2 — Generate narrative answer
        # Detect language
        mem = LangDetectAgent()
        lang = mem.detect_language(question)
        
        # Build structured rows for prompt
        rows_clean = build_rows_for_prompt(dbg)
        
        # Get prompt sections from session state or use defaults
        sections_rt = dict(st.session_state.prompt_sections)
        formatting_text = sections_rt.get("formatting", "")
        
        enabled_templates = [
            t["text"] for t in st.session_state.answering_templates
            if t.get("enabled") and t.get("text", "").strip()
        ]
        if enabled_templates:
            formatting_text = (formatting_text.rstrip() + "\n\n" + "\n\n".join(enabled_templates)).strip()
        sections_rt["formatting"] = formatting_text
        # print('sections_rt:', sections_rt)
        # Compose final prompt
        prompt = compose_prompt(
                        sections_rt,  # make sure to use sections_rt, not original
                        question=question,
                        rows=rows_clean,
                        language=lang,
                        memory_context=get_memory_prompt(3, 'admin', 'admin', 'admin'),
                        source_query=dbg.get("filters", )  # get last 3 memories
                    )
        #print("Final prompt for narrator:", prompt)
        
        # Get answer directly
        t0 = time.time()
        result = narrator.invoke([HumanMessage(content=prompt)])
        client = openai.OpenAI()  # Uses your OPENAI_API_KEY env variable
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        answer = response.choices[0].message.content
        # answer = response['choices'][0]['message']['content']
        t1 = time.time()
        print(f"Narrator response time: {t1 - t0:.2f} seconds")
    
        
        # Save to memory cache
        answer_for_memory = re.sub(r'```[\s\S]*?```', '', answer)
        answer_for_memory = re.sub(r'Source Query[:\s]*[\s\S]*$', '', answer_for_memory, flags=re.IGNORECASE).strip()
        save_to_cache(question, answer_for_memory, 'admin', 'admin', session_id='admin')

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
def process_batch(questions: List[str], batch_size: int = 5, max_workers: int = 5) -> List[Dict[str, Any]]:
    """Process questions in parallel batches"""
    total = len(questions)
    results = []

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    for i in range(0, total, batch_size):
        batch = questions[i:i+batch_size]
        batch_results = []

        batch_num = i // batch_size + 1
        total_batches = (total - 1) // batch_size + 1
        status_text.text(f"Processing batch {batch_num}/{total_batches} ({i+1}-{min(i+batch_size, total)}/{total})")

        # Run batch in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_question, question, i+j, total): (i+j, question)
                for j, question in enumerate(batch)
            }

            for future in as_completed(future_to_idx):
                idx, q = future_to_idx[future]
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    batch_results.append({
                        "question": q,
                        "answer": f"ERROR: {str(e)}",
                        "spec": {},
                        "debug_info": {"error": str(e)},
                        "rows_text": "",
                        "success": False,
                        "time_taken": 0,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                # Update progress bar as each finishes
                progress_bar.progress((idx + 1) / total)

        results.extend(batch_results)

    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {total} questions!")
    return results

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
        avg_time = sum(r.get("time_taken", 0) for r in st.session_state.eval_results) / max(total, 1)
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        metrics_col1.metric("Total Questions", total)
        metrics_col2.metric("Successful", success_count)
        metrics_col3.metric("Failed", fail_count)
        metrics_col4.metric("Avg. Time (sec)", f"{avg_time:.2f}")
        
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
        # Controls to start evaluation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=20, value=5, 
                                        help="Number of questions to process before updating UI")
        
        with col2:
            if st.button("Start Evaluation", disabled=len(st.session_state.eval_questions) == 0):
                if st.session_state.eval_questions:
                    try:
                        st.session_state.eval_running = True
                        
                        # Process questions in batches
                        results = process_batch(st.session_state.eval_questions, batch_size=batch_size)
                        
                        # Update session state
                        st.session_state.eval_results = results
                        st.session_state.eval_complete = True
                    finally:
                        st.session_state.eval_running = False
                    
                    # Rerun once at the end
                    st.rerun()
        
        with col3:
            if st.button("Clear Questions"):
                st.session_state.eval_questions = []
                st.session_state.eval_results = []
                st.success("Questions cleared")
    
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
                st.write(f"Answer length: {len(result['answer']) if isinstance(result['answer'], str) else 0} chars")
                
                # Show search specification
                st.markdown("**Search Specification:**")
                debug_content = {
                    "Query Spec": result["spec"],
                    "MongoDB Filters": result["debug_info"].get("filters", []),
                    "Chosen Collections": result["debug_info"].get("chosen_collections", []),
                    "Raw Results": result.get("rows_text", "(no rows)")
                }
                st.json(debug_content)
                
                # Show debug info
                # st.markdown("**Debug Information:**")
                # st.json(result["debug_info"])
    else:
        st.info("No results yet. Run an evaluation to see results here.")

if __name__ == "__main__":
    ensure_page_config()
    render_batch_evaluation_ui()