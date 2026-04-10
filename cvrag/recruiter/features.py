from __future__ import annotations

from typing import Any

from ..core.config import SOFT_SKILLS
from ..core.cv_data import get_section, load_cv_data


def load_cv() -> dict[str, Any]:
    data = load_cv_data()
    return data if isinstance(data, dict) else {}


def build_skills_matrix(cv: dict[str, Any]) -> list[dict[str, Any]]:
    skills = get_section(cv, "skills", "skill")
    matrix: list[dict[str, Any]] = []

    if isinstance(skills, dict):
        for category, entries in skills.items():
            for entry in _coerce_list(entries):
                normalized = _normalize_skill(entry, category)
                if normalized:
                    matrix.append(normalized)
        return matrix

    for entry in _coerce_list(skills):
        normalized = _normalize_skill(entry, None)
        if normalized:
            matrix.append(normalized)
    existing = {
        str(entry.get("skill", "")).strip().lower()
        for entry in matrix
        if isinstance(entry, dict)
    }
    for skill in SOFT_SKILLS:
        if skill.strip().lower() in existing:
            continue
        matrix.append(
            {
                "skill": skill,
                "category": "Soft Skills",
                "proficiency": None,
                "years": None,
                "last_used": None,
                "keywords": None,
            }
        )
    return matrix


def build_contact_info(cv: dict[str, Any]) -> dict[str, Any]:
    basics = get_section(cv, "basics", "contact") or {}
    if not isinstance(basics, dict):
        basics = {}

    profiles = _coerce_list(basics.get("profiles") or basics.get("links") or cv.get("profiles"))
    schedule_url = _find_schedule_link(profiles, cv)

    location = basics.get("location")
    if isinstance(location, dict):
        location = ", ".join(
            part
            for part in [
                location.get("city"),
                location.get("region"),
                location.get("country"),
                location.get("countryCode"),
            ]
            if part
        )

    return {
        "name": basics.get("name"),
        "title": basics.get("label") or basics.get("headline"),
        "email": basics.get("email"),
        "phone": basics.get("phone"),
        "website": basics.get("website") or basics.get("url"),
        "location": location,
        "profiles": profiles,
        "schedule_url": schedule_url,
    }


def build_availability(cv: dict[str, Any]) -> dict[str, Any]:
    availability = get_section(cv, "availability", "preferences") or {}
    if not isinstance(availability, dict):
        availability = {}

    basics = get_section(cv, "basics") or {}
    if not isinstance(basics, dict):
        basics = {}

    location = basics.get("location")
    if isinstance(location, dict):
        location = ", ".join(
            part
            for part in [
                location.get("city"),
                location.get("region"),
                location.get("country"),
                location.get("countryCode"),
            ]
            if part
        )

    return {
        "timezone": availability.get("timezone") or basics.get("timezone"),
        "notice_period": availability.get("notice_period") or availability.get("noticePeriod"),
        "work_authorization": availability.get("work_authorization")
        or availability.get("workAuthorization"),
        "remote": availability.get("remote"),
        "relocation": availability.get("relocation"),
        "start_date": availability.get("start_date") or availability.get("startDate"),
        "location": location,
    }


def build_certifications(cv: dict[str, Any]) -> list[dict[str, Any]]:
    certs = get_section(cv, "certifications", "certificates", "certification")
    items: list[dict[str, Any]] = []
    for entry in _coerce_list(certs):
        if isinstance(entry, str):
            items.append({"name": entry})
            continue
        if not isinstance(entry, dict):
            continue

        items.append(
            {
                "name": entry.get("name") or entry.get("title"),
                "issuer": entry.get("issuer") or entry.get("organization"),
                "date": entry.get("date") or entry.get("issued"),
                "credential_id": entry.get("credential_id") or entry.get("credentialId"),
                "verify_url": entry.get("url")
                or entry.get("verification_url")
                or entry.get("credential_url"),
            }
        )
    return items


def build_references(cv: dict[str, Any]) -> list[dict[str, Any]]:
    refs = get_section(cv, "references", "endorsements")
    items: list[dict[str, Any]] = []
    for entry in _coerce_list(refs):
        if isinstance(entry, str):
            items.append({"reference": entry})
            continue
        if not isinstance(entry, dict):
            continue

        items.append(
            {
                "name": entry.get("name"),
                "title": entry.get("title") or entry.get("role"),
                "company": entry.get("company"),
                "reference": entry.get("reference") or entry.get("quote"),
                "context": entry.get("context"),
            }
        )
    return items


def build_projects(cv: dict[str, Any]) -> list[dict[str, Any]]:
    projects = get_section(cv, "projects", "project")
    items: list[dict[str, Any]] = []
    for entry in _coerce_list(projects):
        if isinstance(entry, str):
            items.append({"name": entry})
            continue
        if not isinstance(entry, dict):
            continue

        items.append(
            {
                "name": entry.get("name") or entry.get("title"),
                "description": entry.get("description") or entry.get("summary"),
                "highlights": entry.get("highlights") or entry.get("impact"),
                "technologies": entry.get("keywords")
                or entry.get("technologies")
                or entry.get("stack"),
                "url": entry.get("url") or entry.get("website"),
                "roles": entry.get("roles"),
                "start_date": entry.get("startDate") or entry.get("start_date"),
                "end_date": entry.get("endDate") or entry.get("end_date"),
            }
        )
    return items


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_skill(entry: Any, category: str | None) -> dict[str, Any] | None:
    if isinstance(entry, str):
        return {
            "skill": entry,
            "category": category,
            "proficiency": None,
            "years": None,
            "last_used": None,
            "keywords": None,
        }

    if not isinstance(entry, dict):
        return None

    name = (
        entry.get("name")
        or entry.get("skill")
        or entry.get("title")
        or entry.get("technology")
    )
    if not name:
        return None

    return {
        "skill": name,
        "category": entry.get("category") or category,
        "proficiency": entry.get("level") or entry.get("proficiency"),
        "years": entry.get("years") or entry.get("years_experience"),
        "last_used": entry.get("last_used") or entry.get("lastUsed"),
        "keywords": entry.get("keywords"),
    }


def _find_schedule_link(profiles: list[Any], cv: dict[str, Any]) -> str | None:
    direct = cv.get("schedule_url") or cv.get("calendar") or cv.get("calendly")
    if isinstance(direct, str):
        return direct

    for profile in profiles:
        if isinstance(profile, dict):
            network = (profile.get("network") or "").lower()
            url = profile.get("url")
            if not url:
                continue
            if "calendly" in url or "calendar" in network or "calendly" in network:
                return url

    return None
