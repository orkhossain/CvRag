from __future__ import annotations

from dataclasses import dataclass

ALLOWED_INTENTS = {
    "role_targeting",
    "cover_letter",
    "star_examples",
    "technical_deepdive",
    "interview_prep",
    "recruiter_pitch",
    "skills_matrix",
    "general_qa",
    "role_fit_matcher",
    "project_deep_dives",
    "star_bank",
    "quick_summary",
}


@dataclass
class IntentResult:
    intent: str
    confidence: float
    matched_rule: str | None = None


def detect_query_intent_with_confidence(query: str) -> IntentResult:
    q = query.lower()

    if any(phrase in q for phrase in ["cover letter", "application letter", "letter for"]):
        return IntentResult("cover_letter", 0.95, "cover_letter")

    if any(
        phrase in q
        for phrase in [
            "give me an example",
            "tell me about a time",
            "describe a situation",
            "star example",
            "star examples",
            "star format",
            "behavioral",
        ]
    ):
        return IntentResult("star_examples", 0.90, "star_examples")

    if any(
        phrase in q
        for phrase in [
            "how did you implement",
            "technical details",
            "architecture",
            "deep dive",
            "explain the technical",
        ]
    ):
        return IntentResult("technical_deepdive", 0.85, "technical_deepdive")

    if any(
        phrase in q
        for phrase in ["for the role", "position at", "applying to", "job at", "role of"]
    ):
        return IntentResult("role_targeting", 0.85, "role_targeting")

    if any(phrase in q for phrase in ["interview", "prepare for", "questions about", "what would you say"]):
        return IntentResult("interview_prep", 0.80, "interview")

    if any(
        phrase in q
        for phrase in [
            "why are you the best",
            "why should we hire",
            "why hire",
            "best candidate",
            "best option",
            "sell yourself",
            "pitch yourself",
        ]
    ):
        return IntentResult("recruiter_pitch", 0.80, "recruiter_pitch")

    if any(phrase in q for phrase in ["skill", "skills", "technologies", "tech stack", "stack",
                                       "programming language", "programming languages"]):
        return IntentResult("skills_matrix", 0.80, "skills")

    return IntentResult("general_qa", 0.55, None)


def llm_intent_router(query: str) -> IntentResult:
    """LLM fallback for low-confidence rule-based results."""
    try:
        from .llm import get_llm

        allowed = ", ".join(sorted(ALLOWED_INTENTS))
        prompt = (
            f"Classify the following query into exactly one of these intents: {allowed}\n\n"
            f"Query: {query}\n\n"
            "Reply with only the intent name, nothing else."
        )
        llm = get_llm()
        result = llm.invoke(prompt)
        raw = (result.content if hasattr(result, "content") else str(result)).strip().lower()
        raw = raw.strip("\"'`").split()[0] if raw.split() else "general_qa"
        intent = raw if raw in ALLOWED_INTENTS else "general_qa"
        return IntentResult(intent, 0.75, "llm_fallback")
    except Exception:
        return IntentResult("general_qa", 0.50, "llm_fallback_error")


def detect_query_intent(query: str) -> str:
    """Backward-compatible entry point. Uses LLM fallback for low-confidence cases."""
    result = detect_query_intent_with_confidence(query)
    if result.confidence < 0.70:
        result = llm_intent_router(query)
    return result.intent


def enhance_query(query: str, intent: str | None = None) -> str:
    if not intent:
        intent = detect_query_intent(query)

    query_lower = query.strip().lower()

    if intent == "role_targeting":
        return f"relevant experience skills achievements for role: {query}"
    if intent == "role_fit_matcher":
        return f"job description match experience skills achievements: {query}"
    if intent == "cover_letter":
        return f"achievements leadership impact quantified results: {query}"
    if intent == "star_examples":
        return f"specific examples achievements leadership problem solving: {query}"
    if intent == "star_bank":
        return f"behavioral examples achievements leadership impact: {query}"
    if intent == "technical_deepdive":
        return f"technical implementation architecture details: {query}"
    if intent == "project_deep_dives":
        return f"projects architecture impact challenges technologies: {query}"
    if intent == "interview_prep":
        return f"experience skills achievements examples: {query}"
    if intent == "quick_summary":
        return f"summary highlights impact experience: {query}"
    if intent == "skills_matrix":
        return f"skills technologies tools experience: {query}"
    if intent == "export_linkedin":
        return f"professional summary highlights impact: {query}"
    if intent == "export_ats":
        return f"resume summary skills experience projects: {query}"

    if any(word in query_lower for word in ["aws", "cloud", "kubernetes", "docker"]):
        return f"cloud infrastructure devops: {query}"
    if any(word in query_lower for word in ["experience", "worked", "job", "role"]):
        return f"professional experience work history: {query}"
    if any(word in query_lower for word in ["skill", "technology", "tech", "programming"]):
        return f"technical skills technologies: {query}"
    if any(word in query_lower for word in ["project", "built", "created", "developed"]):
        return f"projects development work: {query}"
    if any(word in query_lower for word in ["education", "degree", "university", "college"]):
        return f"education academic background: {query}"

    return query
