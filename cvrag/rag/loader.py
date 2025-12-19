import json
from typing import List

from langchain_core.documents import Document

from ..core.config import CV_PATH


def load_cv_docs() -> List[Document]:
    """
    Loads CV documents from a JSON file formatted as an array of objects:
    [
        {"page_content": "...", "metadata": {...}},
        {"page_content": "...", "metadata": {...}},
        ...
    ]
    Returns a list of LangChain Document objects compatible with all-mpnet-base-v2 embeddings.
    """
    try:
        with CV_PATH.open("r", encoding="utf-8") as handle:
            cv_data = json.load(handle)
    except FileNotFoundError:
        print("cv.json not found - returning empty list.")
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

    print(f"Loaded {len(docs)} documents from cv.json.")
    return docs
