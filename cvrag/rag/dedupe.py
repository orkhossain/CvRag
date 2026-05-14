from __future__ import annotations

import re

from langchain_core.documents import Document


def normalize_doc_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-\.]", "", text)
    return text.strip()


def get_doc_identity(doc: Document) -> str:
    metadata = doc.metadata or {}
    source = metadata.get("source", "")
    chunk_id = metadata.get("chunk_id", "")
    section = metadata.get("section", "")

    if source and chunk_id:
        return f"{source}:{chunk_id}"

    normalized = normalize_doc_text(doc.page_content)
    return f"{section}:{normalized[:300]}"


def dedupe_documents(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        identity = get_doc_identity(doc)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(doc)
    return unique
