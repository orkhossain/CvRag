from __future__ import annotations

import json
from typing import Any

from langchain_community.document_loaders import PDFMinerLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import BASE_DIR, CV_PATH
from ..rag.llm import get_llm


def load_cv_data() -> dict[str, Any] | None:
    json_path = _get_json_path()
    raw = None

    if json_path.exists():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = None

    if raw is None:
        raw = _build_json_from_pdf(json_path)

    return _normalize_cv_data(raw) if raw is not None else None


def ensure_cv_json() -> dict[str, Any] | None:
    """Ensure a JSON representation exists for the CV, generating from PDF if needed."""
    return load_cv_data()


def rebuild_cv_json() -> dict[str, Any] | None:
    """Force rebuild JSON from PDF, overwriting any existing JSON file."""
    json_path = _get_json_path()
    return _build_json_from_pdf(json_path)


def _normalize_cv_data(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return _group_document_list(raw)
    return None


def _get_json_path():
    if CV_PATH.suffix.lower() == ".pdf":
        return CV_PATH.with_suffix(".json")
    return CV_PATH


def _group_document_list(items: list[Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        content = (item.get("page_content") or "").strip()
        meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        section = meta.get("section", "other")

        entry: dict[str, Any] = {}
        if content:
            entry["text"] = content
        for key, value in meta.items():
            if key != "section":
                entry[key] = value

        if entry:
            grouped.setdefault(section, []).append(entry)

    return grouped


def get_section(cv: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in cv:
            return cv[key]
    return None


def _build_json_from_pdf(json_path) -> dict[str, Any] | None:
    pdf_path = _find_pdf_path()
    if not pdf_path:
        return None

    text = _extract_pdf_text(pdf_path)
    if not text:
        return None

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract CV data into JSON. Output JSON only. "
                "If a field is missing, use null or an empty array. "
                "Schema: {{basics:{{name,label,email,phone,website,location,profiles:[{{network,url}}]}}, "
                "summary, skills:[{{name,category,level,years,last_used}}], "
                "experience:[{{company,position,location,startDate,endDate,highlights}}], "
                "projects:[{{name,description,highlights,technologies,url}}], "
                "education:[{{institution,area,studyType,startDate,endDate}}], "
                "certifications:[{{name,issuer,date,credential_id,url}}], "
                "languages:[{{language,fluency}}], "
                "references:[{{name,title,company,reference}}], "
                "availability:{{timezone,notice_period,remote,relocation}}}}",
            ),
            ("user", "CV TEXT:\n{cv_text}"),
        ]
    )

    chain = prompt | get_llm() | StrOutputParser()
    response = chain.invoke({"cv_text": _truncate_text(text)})
    parsed = _parse_json(response)
    if isinstance(parsed, dict):
        try:
            json_path.write_text(
                json.dumps(parsed, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return parsed
        return parsed

    return None


def _find_pdf_path():
    if CV_PATH.suffix.lower() == ".pdf" and CV_PATH.exists():
        return CV_PATH

    fallback = BASE_DIR / "cv.pdf"
    if fallback.exists():
        return fallback

    return None


def _extract_pdf_text(pdf_path) -> str:
    try:
        loader = PDFMinerLoader(str(pdf_path))
        docs = loader.load()
    except Exception as exc:
        print(f"PDF load error for {pdf_path.name}: {exc}")
        return ""

    parts = [(doc.page_content or "").strip() for doc in docs]
    return "\n\n".join(part for part in parts if part)


def _truncate_text(text: str, max_chars: int = 16000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:12000]
    tail = text[-3000:]
    return f"{head}\n...\n{tail}"


def _parse_json(text: str) -> Any | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None
