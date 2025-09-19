# response_gen.py  – natural-language wrapper (Perplexity-style)

import sys, json, textwrap
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# reuse the structured search
from search_agent_new import run_search_agent       # returns (spec, rows_text)
from cache_memory import get_memory_prompt  # Use enhanced version with session_id support

from lang_detect import LangDetectAgent       # ① import
mem = LangDetectAgent()      
from llm_services import  call_narrator_model
# response_gen.py  (only the prompt build bits)
from schema_registry import load_registry, get_collection_names, get_response_prompt_config
from core_rules import render_core_rules, render_match_context

# Import all default prompts from centralized file
from default_prompts import (
    DEFAULT_PROMPT_SECTIONS,
    PROMPT_Q_AND_ROWS,
    DEFAULT_SUGGESTED_QUESTIONS_PROMPT,
    DEFAULT_SUGGESTED_QUESTIONS_COUNT,
    compose_question_and_rows_prompt
)

# ---------- LLM for narration ----------
# narrator = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
from cache_memory import save_to_cache, get_last_memories

reg = load_registry()
CORE_RULES_TEXT = render_core_rules(get_collection_names(reg))

# Your MATCH_CONTEXT remains user-editable "extra add-up":
MATCH_CONTEXT = render_match_context("")


reg = load_registry()
CORE_RULES_TEXT = render_core_rules(get_collection_names(reg))

def get_admin_prompt_sections():
    """Get prompt sections from admin configuration"""
    config = get_response_prompt_config()
    if config and 'prompt_sections' in config:
        return config['prompt_sections']
    return DEFAULT_PROMPT_SECTIONS

def compose_prompt(sections, question, rows, language, memory_context):
    """Join prompt sections and append the question/rows block."""
    body_tmpl = "\n\n".join(sections.values())
    # supply BOTH keys used in your templates
    body = body_tmpl.format(language=language, core_rules=CORE_RULES_TEXT)
    qa = PROMPT_Q_AND_ROWS.format(question=question, rows=rows, memory_context=memory_context,language=language)
    return f"{qa}\n\n{body}"





def get_suggested_questions(q: str, answer: str, Row_data: str, max_questions=3, custom_prompt=None) -> list[str]:
    """
    Generate suggested follow-up questions using admin configuration and dedicated narrator function
    """
    # Get admin configuration for suggested questions
    config = get_response_prompt_config()
    if config and 'suggested_questions_settings' in config:
        settings = config['suggested_questions_settings']
        max_questions = settings.get('count', max_questions)
        base_prompt = settings.get('prompt', DEFAULT_SUGGESTED_QUESTIONS_PROMPT)
    else:
        base_prompt = DEFAULT_SUGGESTED_QUESTIONS_PROMPT
    
    prompt = f"""
    {base_prompt}
    
    Based on the answer below, suggest up to {max_questions} relevant follow-up questions.
    
    Query text:
    {q}
    Row data:
    {Row_data}
    Answer text:
    \"\"\"
    {answer}
    \"\"\"

    Suggested questions:
    -
    """
    
    # Add custom prompt if provided
    final_prompt = f"{prompt}\n\n{custom_prompt}" if custom_prompt else prompt
    
    try:
        # Use dedicated narrator function (non-streaming)
        messages = [{"role": "user", "content": final_prompt}]
        text = call_narrator_model(messages, stream=False)
        print('Suggested Questions#########################################################')
        print(f"Raw response for suggested questions: {text}")  # Debugging line
        # text = response.content.strip()
        

        questions = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("-"):
                q = line[1:].strip()
                if q:
                    questions.append(q)
            if len(questions) >= max_questions:
                break

        return questions

        
    except Exception as e:
        print(f"Error generating suggested questions: {e}")
        return []



# MATCH_CONTEXT = """
# ### MATCH CONTEXT – KEEP AS SEPARATE SYSTEM MESSAGE ###
# This assistant covers **one fixture only**:

# • Fixture  : London County Cricket(Home Team) vs CFS Pinnacle Pro(Away Team)
# • League   : South Africa tour of Australia
# • Ground   : Marrara Cricket Ground (MCG 2), Darwin, Australia
# • Team UIDs: 1242411↔ 1241976 (either side can be home/away)

# Full squad (25):
# London County Cricket (LCC)
# Bilal Muhammad, Ali Raza, Hamza Iqbal, Hameed Ahmadzai, Fahim Baharami, Safwan Manzoor, 
# Zaafer Butt, Yash Tyagi, Ketan Garg, Edress Kamawal, Sadeed Ahmad, Bakhtiar Khan, Abubakar Ahmad, 
# Hector Mclvor, Keegan Fernandes, Haider Zaidi, Luke Giffin, Eli Shenoy, Taha Muhammad, Aqib Mehmood, 
# Farrukh Tahir, Shahbaz Azizullah, Saqib Mehmood, Jack Hunter Lees, Ismail Baharami, Abhimanyu Pandey

# CFS Pinnacle Pro (CPP)
# Jay Chavda, Murad Khan, Kaleb Baldwin, Nathan Weekes, Vansh Lama, Azlan Kumar, Donnel Sylvester, 
# Faris Haider, Arya Khedekar, Prab Singh, Raihan Hussain, Josh Hayward, Arun Patel, Leyton Thres, 
# Ahmad Afzal, James Harvey, Hardik More, Sanay Sadhwani, Mustafa Qureshi, Shajeeth Sivananthan, 
# Adwaaith Sundharam, Ahsan Chaudhry, Reehan Magoon, Micah Thomas, Sai Kotturu, Ralph Figgins, 
# Kavish Patil, Aahaan Srivastav, Gorang Sharma

# 🛈 If the user says “this match / venue / league / team / these players”, resolve the reference to **this fixture** unless they clearly mention something else.
# """


# PROMPT_MAIN = textwrap.dedent("""\
#     You are Perfect Lineup AI—an energetic cricket & fantasy analyst. Give clear, confident answers grounded only in hard numbers and transparent calculations; no speculation.
#     Use the provided match and venue data to answer the user's question in a lively, confident tone with cricket lingo and emojis.

#     Given a user question, chat history, and combined match and venue data, create a cohesive, fantasy-cricket response with:
#     - ⚡ Deep knowledge of recent form.
#     - 📊 Crisp stats with markdown tables.(ONLY IF RELEAVANT MATCHES ARE AVAILABLE IN CONTEXT)
#     - Always calculate metrics like avg inning scores, avg wickets, toss impact from the given scores data of the matches.
#     - 🏏 Lively tone with cricket lingo and emoji callouts.
#     - 🎯 Accurate data from the provided context.
#     - For statistical questions (e.g., average score, chasing success), calculate metrics dynamically from the provided match data and show the calculation.

#     REMEMBER:
#     • Do NOT output raw JSON; this must read like fluent prose.
#     • If the rows section starts with “⚠️”, politely explain no matching
#       records and suggest how the user might refine the query.
# """)

# PROMPT_TONE_STYLE = textwrap.dedent("""\
#     # Tone & Style
#     - Lively, confident, bold; sprinkle 🔥 ✅ ❌ 🧠 📊.
#     - Use cricket lingo (e.g., “death-over threat,” “fantasy lock,” “clean striker”).
#     - Use language: {language}(hi->hindi, en-> formal english)

#     # Efficiency
#     compute and display the math used for calculations.
#     do not show latex, show maths in normal langauge.

#     # Guardrails
#     • Zero speculation—every assertion ties to a number.
#     • Never expose raw JSON; present clean figures only.
#     • Trim fluff; keep tokens lean.
# """)

# PROMPT_FORMATTING = textwrap.dedent("""\
#     ### Formatting:
                                    
#     Defult Guidelines:
#     - **Summary line**: Quick takeaway.
#     - **Markdown tables**: Specific for types of questions
#    - **Bullet points**: 2–5 insights.
#     - **Narrative**: For context.
#     - **Verdict**: Clear fantasy advice.
                                                          
#     For questions about players, use:
#     - **Markdown tables**: For player stats (include columns: Player Name, Team, Role, Avg Fantasy Points, Matches) - **ONLY IF RELEVANT PLAYERS ARE PROVIDED IN CONTEXT**                             

#     For questions relted to matches, use:
#     - **Markdown tables**: For match stats (include columns: Date, Match Title, Team batting first, Chasing Team, Scores[both innings], Result) - **ONLY  AND ONLY IF RELEVANT MATCHES ARE PROVIDED IN CONTEXT**
                                    
#     IMPORTANT:
#     - You must answer ONLY using the data returned by the database/tool.
#     - If the answer is not in the returned data, reply: "No data available for this query."
#     - Do NOT use your own knowledge or make up any facts.
# """)

# PROMPT_Q_AND_ROWS = textwrap.dedent("""\
#     <User Question>
#     {question}

#     <Rows (what you retrieved)>
#     {rows}
# """)
