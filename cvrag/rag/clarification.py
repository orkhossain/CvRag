from __future__ import annotations

import re


RECRUITER_GUIDE_OPTIONS = {
    "role_fit": "Assess role fit against a job description",
    "summary": "Generate a 30-second recruiter summary",
    "skills": "Highlight core technical strengths",
    "star": "Show leadership or STAR examples",
    "deep_dive": "Explain a project or technical decision",
}


def build_clarification_response(query: str, intent: str) -> dict[str, object] | None:
    cleaned = query.strip()
    if not cleaned:
        return None

    if intent not in {
        "role_targeting",
        "cover_letter",
        "interview_prep",
        "recruiter_pitch",
        "general_qa",
    }:
        return None

    lowered = cleaned.lower()
    recruiter_like = _is_recruiter_like(lowered, intent)
    if not recruiter_like:
        return None

    if not _needs_clarification(lowered, intent):
        return None

    if intent == "cover_letter":
        question = (
            "Which cover-letter direction do you want me to take for the recruiter?"
        )
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


def _is_recruiter_like(query: str, intent: str) -> bool:
    if intent in {"role_targeting", "cover_letter", "interview_prep", "recruiter_pitch"}:
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


def _needs_clarification(query: str, intent: str) -> bool:
    token_count = len(re.findall(r"\w+", query))
    has_job_description = len(query) > 180 or "job description" in query
    has_company = bool(re.search(r"\bat\s+[a-z0-9&.-]+\b", query))
    has_role = bool(
        re.search(
            r"\b(engineer|developer|architect|manager|lead|consultant|designer|sre|devops)\b",
            query,
        )
    )
    has_focus = bool(
        re.search(
            r"\b(aws|azure|gcp|kubernetes|docker|terraform|python|react|node|leadership|backend|frontend|platform|security)\b",
            query,
        )
    )

    if intent == "cover_letter":
        return not (has_role or has_company or has_focus)
    if intent == "interview_prep":
        return token_count < 7 or not has_focus
    if intent == "role_targeting":
        return not (has_role or has_company or has_job_description or has_focus)
    if intent == "recruiter_pitch":
        return not (has_role or has_company or has_focus)
    return token_count < 6 or (
        any(
            phrase in query
            for phrase in [
                "summarize the candidate",
                "tell me about the candidate",
                "is this candidate good",
                "why should we hire",
                "screen this profile",
            ]
        )
        and not (has_role or has_company or has_focus or has_job_description)
    )
