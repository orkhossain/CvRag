from __future__ import annotations

import re


RECRUITER_GUIDE_OPTIONS = {
    "role_fit": "Assess role fit against a job description",
    "summary": "Generate a 30-second recruiter summary",
    "skills": "Highlight core technical strengths",
    "star": "Show leadership or STAR examples",
    "deep_dive": "Explain a project or technical decision",
}

_RECRUITER_INTENTS = {
    "role_targeting",
    "cover_letter",
    "interview_prep",
    "recruiter_pitch",
}


# ──────────────────────────────────────────
# Specificity scoring
# ──────────────────────────────────────────

def query_specificity_score(query: str) -> float:
    score = 0.0
    q = query.lower()

    if len(q.split()) >= 8:
        score += 0.25

    if any(role in q for role in ["engineer", "developer", "architect", "manager", "analyst"]):
        score += 0.25

    if any(
        term in q
        for term in [
            "aws", "docker", "kubernetes", "python", "java", "react", "fastapi",
            "gcp", "azure", "terraform", "django", "node", "typescript",
            "leadership", "behavioral", "stakeholder", "backend", "frontend",
            "platform", "security", "devops", "fullstack", "cloud", "sre",
        ]
    ):
        score += 0.25

    if " at " in q or " for " in q:
        score += 0.15

    if len(q) > 180:
        score += 0.10

    return min(score, 1.0)


# ──────────────────────────────────────────
# Unified clarification decision
# ──────────────────────────────────────────

def should_clarify(
    query: str,
    intent: str,
    retrieval_confidence: float | None = None,
    intent_confidence: float | None = None,
) -> bool:
    if intent not in _RECRUITER_INTENTS and not _is_recruiter_like(query.lower(), intent):
        return False

    specificity = query_specificity_score(query)
    rc = retrieval_confidence if retrieval_confidence is not None else 0.0
    ic = intent_confidence if intent_confidence is not None else 1.0

    return specificity < 0.45 and rc < 0.60 and ic >= 0.50


# ──────────────────────────────────────────
# Response builder
# ──────────────────────────────────────────

def build_clarification_response(
    query: str,
    intent: str,
    retrieval_confidence: float | None = None,
    intent_confidence: float | None = None,
) -> dict[str, object] | None:
    cleaned = query.strip()
    if not cleaned:
        return None

    if not _is_recruiter_like(cleaned.lower(), intent):
        return None

    if not should_clarify(cleaned, intent, retrieval_confidence, intent_confidence):
        return None

    if intent == "cover_letter":
        question = "Which cover-letter direction do you want me to take for the recruiter?"
        options = [
            "Target a specific role and company",
            "Emphasize cloud or platform experience",
            "Emphasize backend or full-stack delivery",
            "Emphasize leadership and measurable impact",
        ]
        suggestion = (
            "Reply with the role title, company, and 2-4 priority skills or paste the job description."
        )
    elif intent == "interview_prep":
        question = "Which interview angle should I prepare for the recruiter?"
        options = [
            "Technical depth and architecture",
            "Leadership and stakeholder management",
            "Delivery impact and metrics",
            "Behavioral examples in STAR format",
        ]
        suggestion = (
            "Reply with the interview focus, seniority level, and any skills or projects you want prioritized."
        )
    else:
        question = "Which direction should I take for the recruiter?"
        options = list(RECRUITER_GUIDE_OPTIONS.values())
        suggestion = (
            "Reply with one option plus any of: role title, company, job description, must-have skills, or key concerns."
        )

    answer = "\n".join(
        [
            question,
            "",
            "Options:",
            *[f"- {option}" for option in options],
            "",
            suggestion,
        ]
    )

    return {
        "answer": answer,
        "needs_clarification": True,
        "clarifying_question": question,
        "follow_up_options": options,
    }


# ──────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────

def _is_recruiter_like(query: str, intent: str) -> bool:
    if intent in _RECRUITER_INTENTS:
        return True
    return any(
        phrase in query
        for phrase in [
            "candidate",
            "recruiter",
            "hire",
            "fit for",
            "summary",
            "summarize",
            "why should",
            "what should i ask",
            "screen",
        ]
    )
