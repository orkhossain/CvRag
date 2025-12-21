import json
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PDFMinerLoader
from langchain_core.documents import Document

from ..core.config import CV_PATH


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
    Returns a list of LangChain Document objects compatible with all-mpnet-base-v2 embeddings.
    """
    path = CV_PATH
    if path.is_dir():
        return _load_pdf_dir(path)
    if path.suffix.lower() == ".pdf":
        return _load_pdf_docs(path)
    return _load_json_docs(path)


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
