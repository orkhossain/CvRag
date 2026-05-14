from __future__ import annotations

import re

TECH_TERMS = {
    "aws", "azure", "gcp", "docker", "kubernetes", "python", "java",
    "react", "angular", "fastapi", "django", "postgresql", "mongodb",
    "terraform", "pulumi", "github actions", "gitlab", "ci/cd",
    "typescript", "javascript", "go", "rust", "node", "redis",
    "kafka", "rabbitmq", "elasticsearch", "prometheus", "grafana",
}


def extract_years(text: str) -> set[str]:
    return set(re.findall(r"\b20\d{2}\b", text))


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def extract_tech_terms(text: str) -> set[str]:
    lower = text.lower()
    return {term for term in TECH_TERMS if term in lower}


def verify_answer_grounding(answer: str, context: str) -> dict:
    answer_terms = extract_tech_terms(answer)
    context_terms = extract_tech_terms(context)

    answer_years = extract_years(answer)
    context_years = extract_years(context)

    unsupported_tech = sorted(answer_terms - context_terms)
    unsupported_years = sorted(answer_years - context_years)

    return {
        "is_grounded": not unsupported_tech and not unsupported_years,
        "unsupported_tech": unsupported_tech,
        "unsupported_years": unsupported_years,
    }
