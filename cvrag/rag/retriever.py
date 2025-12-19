from langchain_community.vectorstores import FAISS

from .embeddings import get_embeddings, get_splitter
from .loader import load_cv_docs

_retriever = None


def build_retriever():
    docs = load_cv_docs()
    if not docs:
        return None

    splitter = get_splitter()
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, get_embeddings())

    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "lambda_mult": 0.7,
            "fetch_k": 20,
        },
    )


def init_retriever():
    global _retriever
    _retriever = build_retriever()
    return _retriever


def set_retriever(new_retriever) -> None:
    global _retriever
    _retriever = new_retriever


def get_retriever():
    return _retriever
