from __future__ import annotations

from typing import Dict

from .formatting import format_docs
from .intent import detect_query_intent, enhance_query
from .retriever import get_retriever


def retrieve_context(query: str, intent: str | None = None) -> Dict[str, object]:
    retriever = get_retriever()
    resolved_intent = intent or detect_query_intent(query)
    enhanced_query = enhance_query(query, resolved_intent)

    if not retriever:
        return {
            "context": "No context available",
            "context_used": 0,
            "query_enhanced": enhanced_query != query,
        }

    try:
        primary_docs = retriever.invoke(enhanced_query)

        secondary_docs = []
        if resolved_intent in [
            "role_targeting",
            "cover_letter",
            "interview_prep",
            "role_fit_matcher",
            "quick_summary",
            "export_linkedin",
            "export_ats",
        ]:
            secondary_docs = retriever.invoke(
                "achievements leadership impact results metrics"
            )
        elif resolved_intent in ["technical_deepdive", "project_deep_dives"]:
            secondary_docs = retriever.invoke(
                "technical implementation architecture technologies stack"
            )
        elif resolved_intent in ["star_examples", "star_bank"]:
            secondary_docs = retriever.invoke(
                "led architected implemented improved reduced increased"
            )
        elif resolved_intent == "skills_matrix":
            secondary_docs = retriever.invoke("skills technologies tools programming")

        all_docs = primary_docs + secondary_docs
        seen_content = set()
        unique_docs = []

        for doc in all_docs:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)

        context = format_docs(unique_docs[:15])
        return {
            "context": context,
            "context_used": len(context.split("\n")) if context else 0,
            "query_enhanced": enhanced_query != query,
        }
    except Exception as exc:
        print(f"Retriever error: {exc}")
        return {
            "context": "Error retrieving context",
            "context_used": 0,
            "query_enhanced": enhanced_query != query,
            "error": True,
        }
