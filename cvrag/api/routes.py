import json

from fastapi import APIRouter, Header
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from ..core.config import BASE_DIR, CV_PATH
from ..core.security import guard
from ..models.schemas import (
    CVPayload,
    ExportRequest,
    ProjectDeepDiveRequest,
    Q,
    QuickSummaryRequest,
    RoleFitRequest,
    StarBankRequest,
)
from ..rag.context import retrieve_context
from ..rag.graph import get_agent
from ..rag.loader import load_cv_docs
from ..rag.llm import get_llm
from ..rag.prompts import PROMPTS
from ..rag.retriever import build_retriever, set_retriever
from ..recruiter.features import (
    build_availability,
    build_certifications,
    build_contact_info,
    build_projects,
    build_references,
    build_skills_matrix,
    load_cv,
)

router = APIRouter()


def _run_prompt(prompt, payload: dict) -> str:
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke(payload)


def _safe_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


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
        "recruiter_endpoints": [
            "POST /recruiter/role-fit",
            "POST /recruiter/quick-summary",
            "GET /recruiter/skills-matrix",
            "POST /recruiter/project-deep-dives",
            "POST /recruiter/star-bank",
            "GET /recruiter/certifications",
            "GET /recruiter/availability",
            "POST /recruiter/export",
            "GET /recruiter/export/pdf",
            "GET /recruiter/contact",
            "GET /recruiter/references",
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
    thread_id = q.session_id or "default"
    result = agent.invoke(
        {"query": q.query.strip(), "messages": [HumanMessage(content=q.query.strip())]},
        config={"configurable": {"thread_id": thread_id}},
    )

    response = {
        "answer": result.get("answer", ""),
        "intent": result.get("intent", "general_qa"),
        "context_used": result.get("context_used", 0),
        "query_enhanced": result.get("query_enhanced", False),
    }

    if result.get("error"):
        response["error"] = True

    return response


@router.post("/recruiter/role-fit")
def role_fit(payload: RoleFitRequest, authorization: str | None = Header(default=None)):
    guard(authorization)

    job_description = payload.job_description.strip()
    if not job_description:
        return {"error": "Job description cannot be empty"}

    query_parts = [job_description]
    if payload.role:
        query_parts.append(payload.role)
    if payload.company:
        query_parts.append(payload.company)
    query = " ".join(query_parts)

    context = retrieve_context(query, intent="role_fit_matcher")
    response = _run_prompt(
        PROMPTS["role_fit_matcher"],
        {
            "context": context["context"],
            "job_description": job_description,
            "role": payload.role or "",
            "company": payload.company or "",
        },
    )

    return {
        "result": _safe_json(response),
        "context_used": context.get("context_used", 0),
        "query_enhanced": context.get("query_enhanced", False),
    }


@router.post("/recruiter/quick-summary")
def quick_summary(
    payload: QuickSummaryRequest, authorization: str | None = Header(default=None)
):
    guard(authorization)

    query_parts = ["recruiter summary"]
    if payload.role_level:
        query_parts.append(payload.role_level)
    if payload.focus:
        query_parts.append(payload.focus)
    query = " ".join(query_parts)

    context = retrieve_context(query, intent="quick_summary")
    response = _run_prompt(
        PROMPTS["quick_summary"],
        {
            "context": context["context"],
            "role_level": payload.role_level or "",
            "focus": payload.focus or "",
        },
    )

    return {
        "summary": response,
        "context_used": context.get("context_used", 0),
        "query_enhanced": context.get("query_enhanced", False),
    }


@router.get("/recruiter/skills-matrix")
def skills_matrix(authorization: str | None = Header(default=None)):
    guard(authorization)
    cv = load_cv()
    if not cv:
        return {"error": "CV data not found", "skills": []}

    return {"skills": build_skills_matrix(cv)}


@router.post("/recruiter/project-deep-dives")
def project_deep_dives(
    payload: ProjectDeepDiveRequest, authorization: str | None = Header(default=None)
):
    guard(authorization)
    cv = load_cv()
    if not cv:
        return {"error": "CV data not found", "projects": []}

    projects = build_projects(cv)
    if payload.project_name:
        projects = [
            project
            for project in projects
            if (project.get("name") or "").lower() == payload.project_name.lower()
        ]

    limit = payload.limit if payload.limit is not None else 3
    projects = projects[: max(limit, 0)]

    context_query = payload.focus or "project deep dive"
    context = retrieve_context(context_query, intent="project_deep_dives")
    response = _run_prompt(
        PROMPTS["project_deep_dives"],
        {
            "context": context["context"],
            "projects": json.dumps(projects, indent=2),
        },
    )

    return {
        "projects": _safe_json(response),
        "context_used": context.get("context_used", 0),
        "query_enhanced": context.get("query_enhanced", False),
    }


@router.post("/recruiter/star-bank")
def star_bank(
    payload: StarBankRequest, authorization: str | None = Header(default=None)
):
    guard(authorization)

    competencies = payload.competencies or []
    count = payload.count if payload.count is not None else 3
    context_query = " ".join(competencies) if competencies else "star examples"
    context = retrieve_context(context_query, intent="star_bank")

    response = _run_prompt(
        PROMPTS["star_bank"],
        {
            "context": context["context"],
            "competencies": ", ".join(competencies) if competencies else "general",
            "count": count,
        },
    )

    return {
        "examples": _safe_json(response),
        "context_used": context.get("context_used", 0),
        "query_enhanced": context.get("query_enhanced", False),
    }


@router.get("/recruiter/certifications")
def certifications(authorization: str | None = Header(default=None)):
    guard(authorization)
    cv = load_cv()
    if not cv:
        return {"error": "CV data not found", "certifications": []}

    return {"certifications": build_certifications(cv)}


@router.get("/recruiter/availability")
def availability(authorization: str | None = Header(default=None)):
    guard(authorization)
    cv = load_cv()
    if not cv:
        return {"error": "CV data not found", "availability": {}}

    return {"availability": build_availability(cv)}


@router.post("/recruiter/export")
def export(payload: ExportRequest, authorization: str | None = Header(default=None)):
    guard(authorization)

    export_format = payload.format.strip().lower()
    if export_format not in {"linkedin", "ats"}:
        return {"error": "Format must be 'linkedin' or 'ats'"}

    intent = "export_linkedin" if export_format == "linkedin" else "export_ats"
    context = retrieve_context(payload.focus or export_format, intent=intent)
    response = _run_prompt(
        PROMPTS[intent],
        {
            "context": context["context"],
            "focus": payload.focus or "",
        },
    )

    return {
        "format": export_format,
        "content": response,
        "context_used": context.get("context_used", 0),
        "query_enhanced": context.get("query_enhanced", False),
    }


@router.get("/recruiter/export/pdf")
def export_pdf(authorization: str | None = Header(default=None)):
    guard(authorization)
    pdf_path = None
    if CV_PATH.suffix.lower() == ".pdf" and CV_PATH.exists():
        pdf_path = CV_PATH
    else:
        fallback = BASE_DIR / "cv.pdf"
        if fallback.exists():
            pdf_path = fallback

    if not pdf_path:
        return {"error": "PDF CV not found"}

    return FileResponse(path=pdf_path, filename=pdf_path.name)


@router.get("/recruiter/contact")
def contact(authorization: str | None = Header(default=None)):
    guard(authorization)
    cv = load_cv()
    if not cv:
        return {"error": "CV data not found", "contact": {}}

    return {"contact": build_contact_info(cv)}


@router.get("/recruiter/references")
def references(authorization: str | None = Header(default=None)):
    guard(authorization)
    cv = load_cv()
    if not cv:
        return {"error": "CV data not found", "references": []}

    return {"references": build_references(cv)}
