import os, json
from typing import List, Dict
from fastapi import FastAPI, Header, HTTPException
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

API_TOKEN = os.getenv("API_TOKEN", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. In Hugging Face Spaces, add it under Settings → Repository secrets. Locally: export GROQ_API_KEY=your_key")

app = FastAPI(title="CV RAG API (HF Spaces)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def guard(token: str | None):
    if API_TOKEN and token != f"Bearer {API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------- RAG setup (will build from cv.json if present) ----------
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

    def add(text, meta):
        if text and text.strip():
            docs.append(Document(page_content=text.strip(), metadata=meta))

    b = cv.get("basics", {})
    if b.get("summary"): add("SUMMARY: " + b["summary"], {"section":"summary"})

    for s in cv.get("skills", []):
        add(f"SKILL: {s}", {"section":"skill","skill":s})

    for xp in cv.get("experience", []):
        for h in xp.get("highlights", []):
            add(h, {
                "section":"experience",
                "company": xp.get("company"),
                "role": xp.get("role"),
                "dates": f'{xp.get("start")}-{xp.get("end","present")}',
                "stack": ",".join(xp.get("stack", []))
            })

    for pj in cv.get("projects", []):
        for h in pj.get("highlights", []):
            add(f'{pj.get("name")}: {h}', {
                "section":"project",
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

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\nSummarize for the target role using only the context."),
    ("user", "Target role: {role}\nCompany: {company}\nFocus: {focus}\n\nContext:\n{context}\n\n120 words, bullet points.")
])
summary_chain = (
    {"context": (retriever or (lambda q: [])) | format_docs,
     "role": RunnablePassthrough(), "company": RunnablePassthrough(), "focus": RunnablePassthrough()}
    | SUMMARY_PROMPT | llm | StrOutputParser()
)

COVER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\nDraft a short cover letter (≈180 words) only using the context."),
    ("user", "Company: {company}\nRole: {role}\nEmphasize: {keywords}\n\nContext:\n{context}")
])
cover_chain = (
    {"context": (retriever or (lambda q: [])) | format_docs,
     "company": RunnablePassthrough(), "role": RunnablePassthrough(), "keywords": RunnablePassthrough()}
    | COVER_PROMPT | llm | StrOutputParser()
)

STAR_PROMPT = ChatPromptTemplate.from_template(
    SYSTEM + "\nFrom the context, output 4 STAR bullets (Situation, Task, Action, Result). Quantify results where possible.\nContext:\n{context}"
)
star_chain = ( {"context": (retriever or (lambda q: [])) | format_docs } | STAR_PROMPT | llm | StrOutputParser() )

# ------------ API models ------------
class Q(BaseModel): query: str
class FitReq(BaseModel):
    role: str; company: str = "—"; focus: str = "—"
class CoverReq(BaseModel):
    role: str; company: str; keywords: str = ""

class CVPayload(BaseModel):
    cv: Dict
class CVTextPayload(BaseModel):
    text: str

# ------------ endpoints ------------
@app.get("/")
def root(): return {"ok": True, "endpoints": ["/set-cv","/set-cv-text","/reload","/ask","/fit","/cover","/star"]}

@app.post("/set-cv")
def set_cv(payload: CVPayload, authorization: str | None = Header(default=None)):
    guard(authorization)
    cv = payload.cv
    # persist and rebuild
    with open("cv.json", "w", encoding="utf-8") as f:
        json.dump(cv, f, ensure_ascii=False, indent=2)
    global retriever
    retriever = build_retriever()
    return {"ok": True, "message": "cv.json saved and index rebuilt", "skills": len(cv.get("skills", [])), "experience": len(cv.get("experience", []))}

@app.post("/set-cv-text")
def set_cv_text(payload: CVTextPayload, authorization: str | None = Header(default=None)):
    guard(authorization)
    text = payload.text.strip()
    cv = {
        "basics": {"name": "", "title": "", "location": "", "summary": text, "contacts": {}},
        "skills": [],
        "experience": [],
        "projects": [],
        "education": [],
        "certs": [],
        "languages": []
    }
    with open("cv.json", "w", encoding="utf-8") as f:
        json.dump(cv, f, ensure_ascii=False, indent=2)
    global retriever
    retriever = build_retriever()
    return {"ok": True, "message": "cv.json built from plain text and index rebuilt", "chars": len(text)}

@app.post("/reload")
def reload(authorization: str | None = Header(default=None)):
    guard(authorization)
    global retriever
    retriever = build_retriever()
    return {"ok": True, "message": "retriever rebuilt from cv.json", "has_index": retriever is not None}

@app.post("/ask")
def ask(q: Q, authorization: str | None = Header(default=None)):
    guard(authorization)
    return {"answer": qa_chain.invoke(q.query)}

@app.post("/fit")
def fit(req: FitReq, authorization: str | None = Header(default=None)):
    guard(authorization)
    return {"summary": summary_chain.invoke({"role": req.role, "company": req.company, "focus": req.focus})}

@app.post("/cover")
def cover(req: CoverReq, authorization: str | None = Header(default=None)):
    guard(authorization)
    return {"cover_letter": cover_chain.invoke({"role": req.role, "company": req.company, "keywords": req.keywords})}

@app.get("/star")
def star(authorization: str | None = Header(default=None)):
    guard(authorization)
    return {"bullets": star_chain.invoke({})}