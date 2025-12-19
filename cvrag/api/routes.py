import json

from fastapi import APIRouter, Header

from ..core.config import CV_PATH
from ..core.security import guard
from ..models.schemas import CVPayload, Q
from ..rag.graph import get_agent
from ..rag.loader import load_cv_docs
from ..rag.retriever import build_retriever, set_retriever

router = APIRouter()


@router.get("/")
def root():
    return {
        "ok": True,
        "service": "Enhanced CV RAG API - Unified Ask Endpoint",
        "version": "3.0",
        "primary_endpoint": "/ask",
        "capabilities": [
            "General Q&A about experience and skills",
            "Role-targeted summaries (mention specific roles/companies)",
            "Cover letter generation (ask for cover letters)",
            "STAR format examples (ask for examples/stories)",
            "Technical deep-dives (ask for technical details)",
            "Interview preparation (ask interview-related questions)",
        ],
        "features": [
            "Intent auto-detection from natural language queries",
            "Advanced embedding model (all-mpnet-base-v2)",
            "MMR retrieval for diverse, relevant results",
            "Context-aware response formatting",
            "Multi-strategy document retrieval",
            "LangGraph agent orchestration",
        ],
        "examples": [
            "What experience do I have with Kubernetes?",
            "Create a summary for a Senior DevOps Engineer role at AWS",
            "Write a cover letter for a Full Stack Developer position",
            "Give me STAR examples of leadership",
            "Explain the technical details of the migration project",
            "How should I prepare for questions about cloud architecture?",
        ],
    }


@router.post("/set-cv")
def set_cv(payload: CVPayload, authorization: str | None = Header(default=None)):
    guard(authorization)
    cv = payload.cv

    with CV_PATH.open("w", encoding="utf-8") as handle:
        json.dump(cv, handle, ensure_ascii=False, indent=2)

    set_retriever(build_retriever())

    skills = len(cv.get("skills", [])) if isinstance(cv, dict) else 0
    experience_roles = len(cv.get("experience", [])) if isinstance(cv, dict) else 0
    projects = len(cv.get("projects", [])) if isinstance(cv, dict) else 0

    return {
        "ok": True,
        "message": "CV data saved and search index rebuilt",
        "stats": {
            "skills": skills,
            "experience_roles": experience_roles,
            "projects": projects,
            "total_documents": len(load_cv_docs()),
        },
    }


@router.post("/ask")
def ask(q: Q, authorization: str | None = Header(default=None)):
    guard(authorization)

    if not q.query.strip():
        return {"error": "Query cannot be empty"}

    agent = get_agent()
    result = agent.invoke({"query": q.query.strip()})

    response = {
        "answer": result.get("answer", ""),
        "intent": result.get("intent", "general_qa"),
        "context_used": result.get("context_used", 0),
        "query_enhanced": result.get("query_enhanced", False),
    }

    if result.get("error"):
        response["error"] = True

    return response
