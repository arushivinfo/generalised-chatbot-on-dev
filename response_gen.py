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
from schema_registry import load_registry, get_collection_names
from core_rules import render_core_rules, render_match_context

# ---------- LLM for narration ----------
narrator = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
from cache_memory import save_to_cache, get_last_memories

reg = load_registry()
CORE_RULES_TEXT = render_core_rules(get_collection_names(reg))

# Your MATCH_CONTEXT remains user-editable "extra add-up":
MATCH_CONTEXT = render_match_context("")

PROMPT_MAIN = textwrap.dedent("""\
You are **Insight AI**, a domain-agnostic data analysis assistant.                                                                          |
Your mission: Deliver **clear, confident, and fully data-backed** answers using **only** the provided `rows`.  

---

### **Core Task**
1. Read the question carefully.  
2. Use `rows` to calculate or summarize the answer.  
3. If `rows` is empty or starts with “⚠️”, reply:  
“No relevant data found for this query. Please refine your question based on available entities or attributes.”  
4. Tailor your response focus depending on the type of query:  
    - **Entity queries** → summarize key metrics, performance, or attributes of that entity.  
    - **Comparison queries** → contrast multiple entities, highlight differences and similarities.  
    - **Trend/analytics queries** → emphasize patterns, insights, and notable changes over time.  
    - **Category/aggregate queries** → group, rank, or summarize based on available data fields.  

---

### **Required Answer Format**
1- **Intro Line** – One sentence that sets the context of the answer. Examples:  
    - For entity queries: “Here’s a snapshot of [Entity Name] based on the data…”  
    - For comparison queries: “Here’s how [Entity A] stacks up against [Entity B]…”  
    - For trend queries: “Here’s the trend we see in [Metric/Field] over time…”  
    And also provide a **one-liner direct answer** upfront if possible.  

2- **Relevant Heading** – A bold, concise takeaway (you may create your own heading). One-sentence, high-impact takeaway that directly answers the question(should be in bold and highlighted ans also font size is 1 pointer bigger that other Always, and also use releavent emojis ).

3- **Narrative** – 2–3 sentences of context/analysis with domain-neutral clarity. Use engaging style with emojis where suitable (📊✅⚡📈❌)(Important).  

4- **Bullet Points** – 3 concise, data-backed key insights.  

5- **Recommendation / Conclusion** –  
    - For decision-support queries → provide a clear recommendation (“Entity X outperforms others in efficiency ✅”).  
    - For descriptive/statistical queries → provide a conclusion (“This dataset shows a clear upward trend in Y”).  

6- **Formatting Rules:**  
    - Use markdown tables only if structured data in `rows` supports it.  
    - Never output raw JSON.  

7- **Note at the End (if required):**  
    If response is limited by available data, add a disclaimer such as:  
    “Note: This answer is based solely on the provided dataset. Additional data may change the conclusion.
        Feel free to ask if you’d like me to explore another entity or attribute.”  

---

### **Tone & Style**
- Heading  and the important information should be in bold and highlighted ans also font size is 1 pointer bigger that other (Alwaysand must important), confident, energetic, Clear and straight forward. Use cricket jargon (“death-over threat,” “fantasy gem,” “clean striker”).
- Sprinkle relevant emojis (📊⚡✅🔥📈) to enhance readability.  
- Always match the **language of the question**. If language detection fails, reply:  
“Language not detected. Please re-ask in another language.”  
- Keep answers tight, structured, and engaging.  

---

### **Guardrails**
- No speculation, only use the data provided.  
- Every claim must tie directly to `rows`.  
- No external knowledge or fabricated stats.  
- If a query is unrelated to available data, politely redirect with:  
“No relevant data found for this query. Please refine your question.”  
""")


PROMPT_TONE_STYLE = textwrap.dedent("""\
""")


PROMPT_FORMATTING = textwrap.dedent("""\
""")

reg = load_registry()
CORE_RULES_TEXT = render_core_rules(get_collection_names(reg))

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
    body_tmpl = "\n\n".join(sections.values())
    # supply BOTH keys used in your templates
    body = body_tmpl.format(language=language, core_rules=CORE_RULES_TEXT)
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
        SystemMessage(content="You are a domain-agnostic, grounded Database QA assistant."),
        SystemMessage(content=(MATCH_CONTEXT or "").strip()),
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
