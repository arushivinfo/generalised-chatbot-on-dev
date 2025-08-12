from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PROMPT_TMPL = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Detect the ISO-639-1 language code and the language name of the user's question. "
            "Reply only with the code and name, such as 'en: English', 'hi: Hindi', 'hi-en: Hinglish'.",
        ),
        ("human", "{question}"),
    ]
)

class LangDetectAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    def detect_language(self, question: str) -> str:
        """Detect language of a single user question. Returns a string 'code: name', or 'NONE' if undetectable.
        Special case: If question looks like Hinglish (mix of Hindi and English), return 'hi-en: Hinglish'."""
        try:
            hinglish_keywords = ["kon", "hoga", "kya", "kaun", "hai", "kaise", "kyun", "ky", "kaha", "kab", "kyunki"]
            if any(word in question.lower() for word in hinglish_keywords) and any(c.isalpha() and ord(c) < 128 for c in question):
                return "hi-en: Hinglish"
            response = self.llm.invoke(PROMPT_TMPL.format(question=question))
            result = response.content.strip()
        except Exception:
            return "NONE"
        if not result:
            return "NONE"
        if ':' in result:
            code, name = result.split(':', 1)
            code = code.strip() or "NONE"
            name = name.strip() or "NONE"
            if code.lower() in ["hi-en", "hinglish"] or "hinglish" in name.lower():
                return "hi-en: Hinglish"
            return f"{code}: {name}"
        else:
            return "NONE"