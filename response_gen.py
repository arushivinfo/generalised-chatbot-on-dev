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
Your mission: Deliver **clear, confident, and fully data-backed** answers using **only** the provided `rows`and `MEMORY_CONTEXT`.  

---

### **Core Task**
1. Read the question carefully.  
2. Use `rows`+ `MEMORY_CONTEXT` to calculate or summarize the answer. 
    - Use `MEMORY_CONTEXT` for the follow up questions(Detect by context).
3.  If `rows` is empty or starts with “⚠️”,then use `MEMORY_CONTEXT` and if it is still not found then reply:  
“No relevant data found for this query. Please refine your question based on available entities or attributes.”  
4. If the query refers to an **ambiguous entity** (e.g., “Vivek” but dataset has multiple `Vivek`s with different surnames/IDs),  
   then **return results for all matching entities** instead of assuming one. Present them in a clear comparative or list format. ✅  
5. Tailor your response focus depending on the type of query:  
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
- Heading  and the important information should be in **bold** and highlighted and also font size is 1 pointer bigger that other (Alwaysand must important), confident, energetic, Clear and straight forward. Use cricket jargon (“death-over threat,” “fantasy gem,” “clean striker”).
- Sprinkle relevant emojis (📊⚡✅🔥📈) to enhance readability.  
- Always match the **language of the question**(language can be hinglish or any other language). If language detection fails, reply:  
“Language not detected. Please re-ask in another language.”  
- Keep answers tight, structured, and engaging.  

---

### **Guardrails**
- No speculation, only use the data provided.  
- Every claim must tie directly to `rows`.  
- No external knowledge or fabricated stats.  
- If `rows` is empty or irrelevant, THEN ANSWER USING THE MEMORY CONTEXT.
- If a query is unrelated to available data, politely redirect with:  
“No relevant data found for this query. Please refine your question.”  
""")



# reg = load_registry()
# CORE_RULES_TEXT = render_core_rules(get_collection_names(reg))

 # last 3 Q&A pairs
# print("Memory context for prompt(response_gen):", memory_context)  # Debugging line

PROMPT_Q_AND_ROWS = textwrap.dedent("""\
    <User Question>
    {question}

    <Rows (what you retrieved)>
    {rows}
                                    
    <Memory Context>
    {memory_context} \n\n"Ignore any of the last answers that say 'no data available' or similar for reasoning."
                                    
    <Language>
    {language} , Answer in this language.
                IF the language if hinglish means the acent is Hindi but the script is English then reply in Hinglish only.
    <SOURCE QUERY>
    {source_query}, In the end of the answer,Show this source query as the sourse of the data you are using to answer the question.And Dont show the query in sigle line give query in the actual query format(You can use code block for this and not in list or array keep as it is).
                                    
""")

DEFAULT_PROMPT_SECTIONS = {
    "main": PROMPT_MAIN,
    # 'memory_context': memory_context
}


def compose_prompt(sections, question, rows, language, memory_context,source_query):
     
    """Join prompt sections and append the question/rows block."""
    body_tmpl = "\n\n".join(sections.values())
    # supply BOTH keys used in your templates
    body = body_tmpl.format(language=language, core_rules=CORE_RULES_TEXT)
    qa = PROMPT_Q_AND_ROWS.format(question=question, rows=rows, memory_context=memory_context,language=language,source_query=source_query)
    return f"{qa}\n\n{body}"



def get_suggested_questions(q:str,answer: str, max_questions=3,custom_prompt=None) -> list[str]:
    """
    Given the current assistant's answer, generate up to `max_questions`
    
    """
    prompt = f"""
    You are a suggested question assistant. Based on the answer below, suggest up to {max_questions} relevant follow-up questions
    a user might want to ask next to continue the conversation.  List each question as a bullet point starting with '-'.
    Query text:
    {q}
    Answer text:
    \"\"\"
    {answer}
    \"\"\"

    Suggested questions:
    -
    """
      # Debugging line
    final_prompt = f"{prompt}\n\n{custom_prompt}"
     # Debugging line
        # Use your existing ChatOpenAI instance (narrator) for generation
    response = narrator.generate([[HumanMessage(content=final_prompt)]])
    text = response.generations[0][0].text.strip()

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

