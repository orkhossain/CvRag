from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from langchain_core.documents import Document

from .dedupe import dedupe_documents
from .formatting import format_docs
from .intent import detect_query_intent, enhance_query
from .retriever import build_query_profile, get_retriever


@dataclass
class ContextItem:
    section: str
    content: str
    source: str | None = None
    chunk_id: str | None = None
    company: str | None = None
    dates: str | None = None
    score: float | None = None


def document_to_context_item(doc: Document, score: float | None = None) -> ContextItem:
    metadata = doc.metadata or {}
    return ContextItem(
        section=metadata.get("section", "unknown"),
        content=doc.page_content,
        source=metadata.get("source"),
        chunk_id=metadata.get("chunk_id"),
        company=metadata.get("company"),
        dates=metadata.get("dates"),
        score=score,
    )


@dataclass
class RetrievalBundle:
    docs: list[Document]
    confidence: float
    reasons: list[str]
    scores: list[float]


def compute_retrieval_confidence(docs: list[Document], profile) -> tuple[float, list[str]]:
    if not docs:
        return 0.0, ["no_documents"]

    reasons: list[str] = []
    score = 0.0

    top_docs = docs[:5]
    metadata_sections = [
        (doc.metadata or {}).get("section", "").lower()
        for doc in top_docs
    ]
    text = " ".join(doc.page_content.lower() for doc in top_docs)

    entity_matches = [
        entity for entity in getattr(profile, "entities", [])
        if entity.lower() in text
    ]
    section_matches = [
        section for section in getattr(profile, "sections", [])
        if section.lower() in metadata_sections or section.lower() in text
    ]

    if entity_matches:
        score += 0.35
        reasons.append("entity_match")

    if section_matches:
        score += 0.25
        reasons.append("section_match")

    if len(docs) >= 5:
        score += 0.15
        reasons.append("enough_documents")

    if getattr(profile, "wants_metrics", False) and any(ch.isdigit() for ch in text):
        score += 0.15
        reasons.append("metric_or_number_present")

    if getattr(profile, "wants_dates", False) and any(
        year in text for year in ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
    ):
        score += 0.10
        reasons.append("date_present")

    if not reasons:
        reasons.append("low_signal")

    return min(score, 1.0), reasons


def build_secondary_query(intent: str, profile) -> str:
    terms: list[str] = []

    if intent in {"cover_letter", "role_targeting", "recruiter_pitch", "interview_prep",
                  "role_fit_matcher", "quick_summary", "export_linkedin", "export_ats"}:
        terms += ["achievement", "impact", "leadership", "results", "metrics"]

    if intent in {"technical_deepdive", "project_deep_dives"}:
        terms += ["architecture", "implementation", "technical", "stack", "technologies"]

    if intent in {"star_examples", "star_bank"}:
        terms += ["led", "implemented", "improved", "reduced", "increased", "challenge", "result"]

    if intent == "skills_matrix":
        terms += ["skills", "technologies", "tools", "programming", "frameworks"]

    terms += list(getattr(profile, "entities", []) or [])
    terms += list(getattr(profile, "sections", []) or [])

    if getattr(profile, "wants_metrics", False):
        terms += ["quantified", "impact", "result", "metric"]

    if getattr(profile, "wants_dates", False):
        terms += ["date", "year", "recent", "current"]

    return " ".join(dict.fromkeys(terms))


def retrieve_context(query: str, intent: str | None = None) -> Dict[str, object]:
    retriever = get_retriever()
    resolved_intent = intent or detect_query_intent(query)
    enhanced_query = enhance_query(query, resolved_intent)
    profile = build_query_profile(query)
    secondary_query = build_secondary_query(resolved_intent, profile)

    if not retriever:
        return {
            "context": "No context available",
            "context_used": 0,
            "query_enhanced": enhanced_query != query,
            "secondary_query": secondary_query,
        }

    try:
        primary_docs = retriever.invoke(enhanced_query)

        secondary_docs = []
        if secondary_query:
            secondary_docs = retriever.invoke(secondary_query)

        unique_docs = dedupe_documents(primary_docs + secondary_docs)
        confidence, confidence_reasons = compute_retrieval_confidence(unique_docs, profile)
        top_docs = unique_docs[:15]
        context = format_docs(top_docs)
        structured = [document_to_context_item(doc) for doc in top_docs]
        return {
            "context": context,
            "context_used": len(context.split("\n")) if context else 0,
            "query_enhanced": enhanced_query != query,
            "secondary_query": secondary_query,
            "retrieval_confidence": confidence,
            "retrieval_confidence_reasons": confidence_reasons,
            "structured_context": structured,
        }
    except Exception as exc:
        print(f"Retriever error: {exc}")
        return {
            "context": "Error retrieving context",
            "context_used": 0,
            "query_enhanced": enhanced_query != query,
            "secondary_query": secondary_query,
            "retrieval_confidence": 0.0,
            "retrieval_confidence_reasons": ["error"],
            "error": True,
        }
