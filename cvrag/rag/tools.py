from langchain_core.tools import tool

from ..core.cv_data import get_section, load_cv_data
from ..recruiter.features import (
    build_availability,
    build_certifications,
    build_contact_info,
    build_projects,
    build_references,
    build_skills_matrix,
)


def _load_cv() -> dict:
    data = load_cv_data()
    return data if isinstance(data, dict) else {}


@tool
def get_cv_sections() -> list[str]:
    """List top-level sections available in the CV JSON."""
    cv = _load_cv()
    return sorted(cv.keys())


@tool
def get_cv_section(section: str) -> str:
    """Return a JSON string for a specific CV section."""
    cv = _load_cv()
    if not cv:
        return "CV data not found."

    value = get_section(cv, section)
    if value is None:
        available = ", ".join(sorted(cv.keys()))
        return f"Section '{section}' not found. Available sections: {available}"

    return _serialize(value)


@tool
def get_cv_stats() -> dict[str, int]:
    """Return counts for common CV sections."""
    cv = _load_cv()
    if not cv:
        return {}

    def _count_list(key: str) -> int:
        value = cv.get(key, [])
        return len(value) if isinstance(value, list) else 0

    return {
        "skills": _count_list("skills"),
        "experience_roles": _count_list("experience") or _count_list("work"),
        "projects": _count_list("projects"),
        "education_entries": _count_list("education"),
    }


@tool
def get_skills_matrix() -> list[dict]:
    """Return a structured skills matrix when available."""
    cv = _load_cv()
    if not cv:
        return []
    return build_skills_matrix(cv)


@tool
def get_contact_info() -> dict:
    """Return contact details and scheduling link if available."""
    cv = _load_cv()
    if not cv:
        return {}
    return build_contact_info(cv)


@tool
def get_availability() -> dict:
    """Return availability and location preferences."""
    cv = _load_cv()
    if not cv:
        return {}
    return build_availability(cv)


@tool
def get_certifications() -> list[dict]:
    """Return certifications with verification links when available."""
    cv = _load_cv()
    if not cv:
        return []
    return build_certifications(cv)


@tool
def get_references() -> list[dict]:
    """Return references or endorsements."""
    cv = _load_cv()
    if not cv:
        return []
    return build_references(cv)


@tool
def get_projects() -> list[dict]:
    """Return project summaries."""
    cv = _load_cv()
    if not cv:
        return []
    return build_projects(cv)


def _serialize(value: object) -> str:
    try:
        import json

        return json.dumps(value, indent=2)
    except TypeError:
        return str(value)
