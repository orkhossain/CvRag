import json
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PDFMinerLoader
from langchain_core.documents import Document

from ..core.config import CV_PATH, SOFT_SKILLS


def load_cv_docs() -> List[Document]:
    """
    Loads CV documents from either:
      - a JSON file formatted as an array of objects:
        [
            {"page_content": "...", "metadata": {...}},
            {"page_content": "...", "metadata": {...}},
            ...
        ]
      - a PDF file (single or directory of PDFs)
    Returns a list of LangChain Document objects compatible with multilingual embeddings.
    """
    path = CV_PATH
    if path.is_dir():
        return _append_soft_skills(_load_pdf_dir(path))
    if path.suffix.lower() == ".pdf":
        return _append_soft_skills(_load_pdf_docs(path))
    return _append_soft_skills(_load_json_docs(path))


def _append_soft_skills(docs: List[Document]) -> List[Document]:
    if not SOFT_SKILLS:
        return docs
    text = "Soft skills: " + ", ".join(SOFT_SKILLS)
    docs.append(
        Document(
            page_content=text,
            metadata={"section": "skills", "category": "soft_skills"},
        )
    )
    return docs


def _load_json_docs(path: Path) -> List[Document]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            cv_data = json.load(handle)
    except FileNotFoundError:
        print(f"{path.name} not found - returning empty list.")
        return []
    except json.JSONDecodeError as exc:
        print(f"JSON decoding error: {exc}")
        return []

    if isinstance(cv_data, dict):
        docs = _load_structured_cv(cv_data)
        print(f"Loaded {len(docs)} documents from structured JSON in {path.name}.")
        return docs

    if not isinstance(cv_data, list):
        print(f"Unsupported JSON format in {path.name}.")
        return []

    docs: List[Document] = []
    for idx, item in enumerate(cv_data):
        if not isinstance(item, dict):
            print(f"Skipping non-dict item at index {idx}: {item}")
            continue

        text = item.get("page_content", "")
        meta = item.get("metadata", {})

        if text and isinstance(meta, dict):
            docs.append(Document(page_content=text.strip(), metadata=meta))
        else:
            print(f"Skipping invalid entry at index {idx}: {item}")

    print(f"Loaded {len(docs)} documents from {path.name}.")
    return docs


def _load_structured_cv(cv_data: dict) -> List[Document]:
    docs: List[Document] = []
    for section, entries in cv_data.items():
        for item in _coerce_list(entries):
            text, meta = _entry_to_text_and_meta(item)
            if not text:
                continue
            meta = dict(meta) if isinstance(meta, dict) else {}
            meta.setdefault("section", section)
            docs.append(Document(page_content=text.strip(), metadata=meta))
    return docs


def _coerce_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _entry_to_text_and_meta(item):
    if isinstance(item, str):
        return item, {}

    if not isinstance(item, dict):
        return "", {}

    text = item.get("text") or item.get("summary") or item.get("description")
    if not text:
        highlights = item.get("highlights") or item.get("impact")
        if isinstance(highlights, list):
            text = "; ".join(str(entry) for entry in highlights if entry)
        elif isinstance(highlights, str):
            text = highlights

    if not text:
        text = _flatten_dict(item)

    meta = {key: value for key, value in item.items() if key not in {"text", "summary", "description", "highlights", "impact"}}
    return text, meta


def _flatten_dict(item: dict) -> str:
    parts = []
    for key, value in item.items():
        if value is None or key in {"highlights", "impact"}:
            continue
        if isinstance(value, (list, dict)):
            continue
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


def _load_pdf_docs(path: Path) -> List[Document]:
    if not path.exists():
        print(f"{path.name} not found - returning empty list.")
        return []

    try:
        loader = PDFMinerLoader(str(path))
        raw_docs = loader.load()
    except Exception as exc:
        print(f"PDF load error for {path.name}: {exc}")
        return []

    docs: List[Document] = []
    for idx, doc in enumerate(raw_docs):
        content = (doc.page_content or "").strip()
        if not content:
            continue
        meta = dict(doc.metadata) if doc.metadata else {}
        meta.setdefault("section", "pdf")
        meta.setdefault("source", str(path))
        meta.setdefault("page", meta.get("page", idx))
        docs.append(Document(page_content=content, metadata=meta))

    print(f"Loaded {len(docs)} documents from {path.name}.")
    return docs


def _load_pdf_dir(path: Path) -> List[Document]:
    docs: List[Document] = []
    for pdf_path in sorted(path.iterdir()):
        if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
            docs.extend(_load_pdf_docs(pdf_path))

    if not docs:
        print(f"No PDF documents found in {path}.")
    return docs
