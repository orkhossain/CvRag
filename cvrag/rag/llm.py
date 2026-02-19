from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_ollama import ChatOllama

from ..core.config import GROQ_MODEL, HF_MODEL, LLM_PROVIDER, LLM_TEMPERATURE, OLLAMA_MODEL

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        if LLM_PROVIDER == "groq":
            _llm = ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)
        elif LLM_PROVIDER == "ollama":
            _llm = ChatOllama(model=OLLAMA_MODEL, temperature=LLM_TEMPERATURE)
        else:
            endpoint = HuggingFaceEndpoint(
                repo_id=HF_MODEL,
                temperature=LLM_TEMPERATURE,
            )
            _llm = ChatHuggingFace(llm=endpoint)
    return _llm
