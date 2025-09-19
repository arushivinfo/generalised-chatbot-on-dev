from dotenv import load_dotenv
from llm_services import call_narrator_model

load_dotenv()



class LangDetectAgent:
    def __init__(self):
        # No need for LLM initialization - using call_narrator_model directly
        pass

    def detect_language(self, question: str) -> str:
        """Detect language of a single user question. Returns a string 'code: name', or 'NONE' if undetectable.
        Special case: If question looks like Hinglish (mix of Hindi and English), return 'Hinglish: Hinglish'."""
        try:
            hinglish_keywords = ["kon", "hoga", "kya", "kaun", "hai", "kaise", "kyun", "ky", "kaha", "kab", "kyunki"]
            if any(word in question.lower() for word in hinglish_keywords) and any(c.isalpha() and ord(c) < 128 for c in question):
                return "Hinglish: Hinglish"
            
            # Use call_narrator_model instead of LangChain
            prompt = f"""Detect the ISO-639-1 language code and the language name of the user's question. 
            Reply only with the code and name, such as 'en: English', 'hi: Hindi', 'Hinglish: Hinglish'.

            Question: {question}"""
            
            
            messages = [{"role": "user", "content": prompt}]
            result = call_narrator_model(messages, stream=False, temperature=0.0)
                
        except Exception:
            return "NONE"
        if not result:
            return "NONE"
        if ':' in result:
            code, name = result.split(':', 1)
            code = code.strip() or "NONE"
            name = name.strip() or "NONE"
            if code.lower() in ["hi-en", "hinglish"] or "hinglish" in name.lower():
                return "Hinglish: Hinglish"
            return f"{code}: {name}"
        else:
            return "NONE"