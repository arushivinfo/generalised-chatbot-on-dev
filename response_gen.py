# response_gen.py  – natural-language wrapper (Perplexity-style)

import sys, json, textwrap
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# reuse the structured search
from search_agent_new import run_search_agent          # returns (spec, rows_text)

from lang_detect import LangDetectAgent       # ① import
mem = LangDetectAgent()      

# response_gen.py  (only the prompt build bits)
from core_rules import render_core_rules, render_match_context
from search_agent_new import COLL_MAP  # central source of truth for names

# ---------- LLM for narration ----------
narrator = ChatOpenAI(model="o3-mini")
from cache_memory import save_to_cache, get_last_memories

MATCH_CONTEXT = textwrap.dedent("""\
    **Match Context** (System Message):
    This assistant covers **only** the following fixture:
    - **Fixture**: Australia(AUS) vs South Africa(SA)
    - **League**: South Africa tour of Australia
    - **Ground**: Marrara Cricket Ground (MCG 2), Darwin, Australia
    - **Team UIDs**: 5 (AUS) ↔ 19 (SA)

    **Squads**:
    **Australia (AUS):**
    Mitchell Owen, Adam Zampa, Travis Head, Ben Dwarshuis, Matthew Short, Josh Inglis, 
    Matthew Kuhnemann, Sean Abbott, Glenn Maxwell, Mitchell Marsh, 
    Josh Hazlewood, Cameron Green, Tim David, Aaron Hardie, Nathan Ellis

    **South Africa (SA):**
    Dewald Brevis, Kwena Maphaka, Lhuan dre Pretorius, Kagiso Rabada, Nqabayomzi Peter, 
    Aiden Markram, Lungisani Ngidi, Rassie van der Dussen, George Linde, Senuran Muthusamy, 
    Prenelan Subrayen, Nandre Burger, Corbin Bosch, Ryan Rickelton, Tristan Stubbs


    **Reference Resolution**:
    - Terms like “this match,” “venue,” “league,” or “these players” refer to the above fixture unless the user explicitly mentions another match, team, or player.
    - If the user references a different match or unlisted player, respond: “This query is outside the provided fixture data. Please specify details related to AUS vs SA.”
""")

CORE_RULES_TEXT = render_core_rules(COLL_MAP)

# Your MATCH_CONTEXT remains user-editable "extra add-up":
MATCH_CONTEXT = render_match_context(MATCH_CONTEXT)


PROMPT_MAIN = textwrap.dedent("""\
    You are Perfect Lineup AI, a cricket and fantasy analyst. Deliver clear, confident, data-driven answers using only the provided match and venue data. Use a lively tone with cricket lingo (e.g., “death-over specialist,” “fantasy gem”) and emojis (⚡📊🏏).

    **Task**:
    - Answer the user’s question using only the data in `rows` and `MATCH_CONTEXT`.
    - Calculate metrics (e.g., average innings score, wickets per match, toss win impact) dynamically from `rows` data. Show calculations in plain language (e.g., “Total runs = 150 + 160 = 310; Average = 310/2 = 155”).
    - If `rows` is empty or starts with “⚠️”, respond: “No relevant data found for this query. Please specify players, matches, or metrics related to the given fixture.”
    - For player queries, focus on stats like runs, wickets, or fantasy points from `rows`.
    - For match queries, analyze scores, results, or toss outcomes from `rows`.
                              
    **Core rule for all questions**:
    To ensure history-backed, rich answers, whenever you answer any venue related question, prefer and cite the recent matches at the venue, player performances at the venue, and the match context provided above.
                              
    **Core rule to answer prediction-related questions**:
    For questions which can be related to predcition, first look into the preiction data inside upcoming_match collection, then using that information, use the historical data to answer the question. 
    
    **Guardrails**:
    - Do NOT speculate (e.g., don’t assume a player’s form without data).
    - Do NOT use external knowledge or make up stats.
    - Do NOT output raw JSON; present clean prose with markdown tables or bullet points as specified.
""")

PROMPT_TONE_STYLE = textwrap.dedent("""\
    # Tone & Style
    - Lively, confident, bold; sprinkle 🔥 ✅ ❌ 🧠 📊.
    - Use cricket lingo (e.g., “death-over threat,” “fantasy lock,” “clean striker”).
    - Use language: {language}
    - Always answer in the user's language: {language} 
    - If the Language is not detected or comming as NONE, then Send a massage  ' Language is not detected Please use another language   ."
    - always answer the question on the language of the question.(most important)
    - Use emojis to highlight key points.

    **Calculations**:
    - Show math in plain language (e.g., “Average score = (120 + 140) / 2 = 130”).
    - Avoid LaTeX or complex notation.
    - For questions which can be related to predcition, first look into the preiction data inside upcoming_match collection, then using that information, use the historical data to answer the question. 

    **Guardrails**:
    - Every claim must tie to a number in `rows`.
    - Avoid fluff (e.g., don’t repeat the question unnecessarily).
""")

PROMPT_FORMATTING = textwrap.dedent("""\
    **Formatting**:
    - **Summary**: Start with a one-sentence takeaway.
    - **Narrative**: Provide 2–3 sentences of context or analysis.
    - **Bullet Points**: List 3 key insights (e.g., top performer, toss impact).
    - **Verdict**: End with clear fantasy advice (e.g., “Pick Player X as captain”).
                            
    **Guardrails**:
    - Only use tables if relevant data is in `rows`.
    - If no data fits the query, respond: “No relevant data found.”
""")

def get_memory_prompt2(n):
    memories = get_last_memories(n)
    filtered = [m for m in memories if "no data" not in m["answer"].lower()]
    if not filtered:
        return ""
    mem_text = "\n\n".join(
        [f"Previous Q: {m['query']}\nPrevious A: {m['answer']}" for m in filtered]
    )
    return (
        "### RECENT MEMORY CONTEXT\n"
        "If any of the last 3 answers below say 'no data available' or similar, ignore that answer for reasoning.\n"
        f"{mem_text}\n"
        "Just use the other memories to answer the question.\n."
    )
memory_context = get_memory_prompt2(1)
print("Memory context for prompt(response_gen):", memory_context)  # Debugging line

PROMPT_Q_AND_ROWS = textwrap.dedent("""\
    <User Question>
    {question}

    <Rows (what you retrieved)>
    {rows}
                                    
    <Memory Context>
    {memory_context} \n\n"Ignore any of the last 3 answers that say 'no data available' or similar for reasoning."
                                    
    <Language>
    {language} , Answer in this language.
                                    
""")

DEFAULT_PROMPT_SECTIONS = {
    "main": PROMPT_MAIN,
    "tone_style": PROMPT_TONE_STYLE,
    "formatting": PROMPT_FORMATTING,
    'memory_context': memory_context
}


def compose_prompt(sections, question, rows, language, memory_context):
    """Join prompt sections and append the question/rows block."""
    body = "\n\n".join(sections.values()).format(language=language)
    qa = PROMPT_Q_AND_ROWS.format(question=question, rows=rows, memory_context=memory_context,language=language)
    return f"{body}\n\n{qa}"


# ---------- driver ----------
# def answer_question(query: str):
#     from lang_detect import MemoryAgent
#     mem = MemoryAgent(k=5)
#     standalone_q = mem.process(query)
#     spec, rows_text = run_search_agent(standalone_q)


#     rows_clean = rows_text or "(no rows)"
#     msg = NL_PROMPT.format(question=query, rows=rows_clean)

#     reply = narrator.invoke([
#         SystemMessage(content="You are Perplexity-style sports analyst."),
#         HumanMessage(content=msg)
#     ])

#     return reply.content, spec, rows_text

def answer_question(query: str, streaming: bool = False, prompt_sections=None, history=None):
    # STEP 0 – detect language only (standalone rewrite ignored)
    lang = mem.detect_language(query)

    # STEP 1 – structured DB search using the original query
    spec, rows_text, dbg = run_search_agent(query,history=memory_context)

    # STEP 2 – narrative answer in the detected language
    rows_clean = rows_text or "(no rows)"
    sections = prompt_sections or DEFAULT_PROMPT_SECTIONS
    prompt = compose_prompt(sections, question=query, rows=rows_clean, language=lang)

    # Add last 3 memories to the prompt
    memories = get_last_memories(1)
    if memories:
        mem_text = "\n\n".join(
            [f"Previous Q: {m['query']}\nPrevious A: {m['answer']}" for m in memories]
        )
        prompt = f"{mem_text}\n\n{prompt}"

    messages = [
        SystemMessage(content="You are Perfect Lineup sports analyst."),
        SystemMessage(content=MATCH_CONTEXT.strip()),  # 🆕  pass match context
        HumanMessage(content=prompt),
    ]

    if streaming:
        def _gen():
            answer_parts = []
            for chunk in narrator.stream(messages):
                token = getattr(chunk, "content", "")
                answer_parts.append(token)
                yield token
            full = "".join(answer_parts)
            mem.update(query, full)
            save_to_cache(query, full)
        return _gen(), spec, rows_text

    reply = narrator.invoke(messages)

    # keep conversation memory fresh
    mem.update(query, reply.content)
    save_to_cache(query, reply.content)
    return reply.content, spec, rows_text



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python response_gen.py \"<your question>\"")
        sys.exit(1)

    q = sys.argv[1]
    answer, spec, raw = answer_question(q,history = memory_context)

    print("\n=== NATURAL-LANGUAGE ANSWER ===\n")
    print(answer)

    # optional: persist run
    Path("last_nl_run.json").write_text(json.dumps({
        "query": q,
        "spec": spec,
        "rows": raw,
        "answer": answer
    }, indent=2))






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
