import os, json
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----- LangChain RAG -----
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --------- Environment checks ----------
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. In Hugging Face Spaces, add it under Settings → Repository secrets.")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="CV Ask API (HF Spaces)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- RAG setup (build from cv.json if present) ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

def load_cv_docs() -> List[Document]:
    try:
        with open("cv.json", "r", encoding="utf-8") as f:
            cv = json.load(f)
    except FileNotFoundError:
        return []
    docs: List[Document] = []

    def add(text: Optional[str], meta: Dict):
        if text and str(text).strip():
            docs.append(Document(page_content=str(text).strip(), metadata=meta))

    b = cv.get("basics", {})
    add(("SUMMARY: " + b.get("summary", "")) if b.get("summary") else "", {"section": "summary"})

    for s in cv.get("skills", []):
        add(f"SKILL: {s}", {"section": "skill", "skill": s})

    for xp in cv.get("experience", []):
        for h in xp.get("highlights", []):
            add(h, {
                "section": "experience",
                "company": xp.get("company"),
                "role": xp.get("role"),
                "dates": f'{xp.get("start")}-{xp.get("end","present")}',
                "stack": ",".join(xp.get("stack", []))
            })

    for pj in cv.get("projects", []):
        for h in pj.get("highlights", []):
            add(f'{pj.get("name")}: {h}', {
                "section": "project",
                "name": pj.get("name"),
                "stack": ",".join(pj.get("stack", []))
            })
    return docs

def build_retriever():
    docs = load_cv_docs()
    if not docs:
        return None
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, emb)
    return vs.as_retriever(search_kwargs={"k": 8})

retriever = build_retriever()

SYSTEM = """You are a precise career assistant.
Answer ONLY with facts from the provided context chunks.
If a detail is missing, say you don't have it. Be concise."""

def format_docs(_docs: List[Document]) -> str:
    lines = []
    for d in _docs:
        m = d.metadata
        tag = f"[{m.get('section','')}"
        if m.get("company"): tag += f" | {m['company']}"
        if m.get("name"): tag += f" | {m['name']}"
        if m.get("dates"): tag += f" | {m['dates']}"
        tag += "]"
        lines.append(f"- {tag} {d.page_content}")
    return "\n".join(lines)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\nContext:\n{context}"),
    ("user", "{question}")
])
qa_chain = (
    {"context": (retriever or (lambda q: [])) | format_docs, "question": RunnablePassthrough()}
    | QA_PROMPT | llm | StrOutputParser()
)

# ------------ API models ------------
class Q(BaseModel):
    query: str

# ------------ endpoints ------------
@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/ask"], "has_index": retriever is not None}

@app.post("/ask")
def ask(q: Q):
    global retriever
    if retriever is None:
        # try to build once more (e.g., cv.json added after boot)
        retriever = build_retriever()
        if retriever is None:
            raise HTTPException(status_code=400, detail="cv.json not found or empty. Place a cv.json next to app.py and retry.")
    return {"answer": qa_chain.invoke(q.query)}