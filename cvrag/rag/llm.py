from langchain_groq import ChatGroq

from ..core.config import GROQ_MODEL, LLM_TEMPERATURE

_llm = None


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)
    return _llm
