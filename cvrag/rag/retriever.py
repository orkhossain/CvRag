from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .embeddings import get_embeddings, get_splitter
from .loader import load_cv_docs

_retriever = None

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "do",
    "for",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
    "you",
    "your",
}

_SECTION_KEYWORDS = {
    "summary": {"summary", "profile", "overview", "about"},
    "experience": {"experience", "work", "career", "role", "roles", "employment"},
    "skills": {"skill", "skills", "technology", "technologies", "tool", "tools", "stack"},
    "projects": {"project", "projects", "portfolio", "build", "built"},
    "education": {"education", "degree", "university", "college", "school"},
    "certifications": {"certification", "certifications", "certificate", "certified"},
    "languages": {"language", "languages"},
    "availability": {"availability", "remote", "relocation", "notice", "timezone"},
    "basics": {"contact", "email", "phone", "website", "linkedin", "location"},
}

_METRIC_HINTS = {
    "increase",
    "increased",
    "improve",
    "improved",
    "improvement",
    "reduce",
    "reduced",
    "latency",
    "performance",
    "scale",
    "scaled",
    "throughput",
    "revenue",
    "cost",
    "saved",
    "percent",
    "%",
}

_ROLE_HINTS = {
    "engineer",
    "developer",
    "architect",
    "manager",
    "lead",
    "devops",
    "platform",
    "backend",
    "frontend",
    "fullstack",
    "full-stack",
    "sre",
}


@dataclass(frozen=True)
class QueryProfile:
    keywords: tuple[str, ...]
    sections: tuple[str, ...]
    entities: tuple[str, ...]
    wants_metrics: bool
    wants_dates: bool


class MetadataAwareRetriever:
    def __init__(
        self,
        vectorstore: FAISS,
        chunks: Sequence[Document],
        *,
        k: int = 10,
        fetch_k: int = 24,
        lexical_k: int = 8,
    ) -> None:
        self.vectorstore = vectorstore
        self.chunks = list(chunks)
        self.k = k
        self.fetch_k = fetch_k
        self.lexical_k = lexical_k

    def invoke(self, query: str) -> list[Document]:
        profile = build_query_profile(query)
        semantic_docs = self.vectorstore.max_marginal_relevance_search(
            query,
            k=self.fetch_k,
            fetch_k=max(self.fetch_k * 2, 20),
            lambda_mult=0.7,
        )
        lexical_docs = keyword_retrieve(self.chunks, profile, limit=self.lexical_k)
        ranked = rerank_documents(semantic_docs + lexical_docs, profile)
        return ranked[: self.k]


def build_retriever():
    docs = load_cv_docs()
    if not docs:
        return None

    splitter = get_splitter()
    chunks = splitter.split_documents(docs)
    vs = FAISS.from_documents(chunks, get_embeddings())
    return MetadataAwareRetriever(vs, chunks)


def init_retriever():
    global _retriever
    _retriever = build_retriever()
    return _retriever


def set_retriever(new_retriever) -> None:
    global _retriever
    _retriever = new_retriever


def get_retriever():
    return _retriever


def build_query_profile(query: str) -> QueryProfile:
    tokens = tuple(_tokenize(query))
    sections = tuple(
        section
        for section, cues in _SECTION_KEYWORDS.items()
        if any(cue in query.lower() for cue in cues)
    )
    entities = tuple(token for token in tokens if _looks_like_entity(token))
    wants_metrics = bool(re.search(r"\b(metric|metrics|impact|results?|achievement|achievements)\b", query.lower()))
    wants_metrics = wants_metrics or any(token in _METRIC_HINTS for token in tokens)
    wants_dates = bool(re.search(r"\b(20\d{2}|19\d{2}|recent|latest|current|when)\b", query.lower()))
    return QueryProfile(
        keywords=tokens,
        sections=sections,
        entities=entities,
        wants_metrics=wants_metrics,
        wants_dates=wants_dates,
    )


def rerank_documents(documents: Sequence[Document], profile: QueryProfile) -> list[Document]:
    seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    scored: list[tuple[float, int, Document]] = []

    for idx, doc in enumerate(documents):
        content = (doc.page_content or "").strip()
        if not content:
            continue

        meta_key = tuple(
            sorted((str(key), str(value)) for key, value in (doc.metadata or {}).items())
        )
        dedupe_key = (content, meta_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        score = score_document(doc, profile)
        scored.append((score, idx, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored]


def score_document(doc: Document, profile: QueryProfile) -> float:
    metadata = {str(key).lower(): str(value).lower() for key, value in (doc.metadata or {}).items()}
    haystack_parts = [doc.page_content.lower(), *metadata.values()]
    haystack = " ".join(part for part in haystack_parts if part)

    score = 0.0

    keyword_hits = sum(1 for keyword in profile.keywords if keyword in haystack)
    score += keyword_hits * 2.0

    if profile.sections:
        section = metadata.get("section", "")
        normalized_section = _normalize_section(section)
        if normalized_section in profile.sections:
            score += 3.5

    entity_hits = sum(1 for entity in profile.entities if entity in haystack)
    score += entity_hits * 2.5

    if profile.wants_metrics and (_contains_metric(doc.page_content) or any(_contains_metric(value) for value in metadata.values())):
        score += 2.0

    if profile.wants_dates and (_contains_date(doc.page_content) or any(_contains_date(value) for value in metadata.values())):
        score += 1.5

    if any(role_hint in haystack for role_hint in _ROLE_HINTS):
        score += 0.5

    if metadata.get("company") and any(metadata["company"] == entity for entity in profile.entities):
        score += 2.0

    if metadata.get("name") and any(metadata["name"] == entity for entity in profile.entities):
        score += 2.0

    return score


def keyword_retrieve(
    documents: Sequence[Document],
    profile: QueryProfile,
    *,
    limit: int = 8,
) -> list[Document]:
    scored: list[tuple[float, int, Document]] = []
    for idx, doc in enumerate(documents):
        score = score_document(doc, profile)
        if score <= 0:
            continue
        scored.append((score, idx, doc))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored[:limit]]


def _tokenize(text: str) -> Iterable[str]:
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9+#./-]*", text.lower()):
        token = raw.strip(".-/")
        if len(token) < 2 or token in _STOPWORDS:
            continue
        yield token


def _looks_like_entity(token: str) -> bool:
    if len(token) <= 2:
        return False
    if any(char.isdigit() for char in token):
        return True
    return token not in _STOPWORDS and (
        token.isupper()
        or "-" in token
        or token in {"aws", "gcp", "azure", "kubernetes", "terraform", "docker", "python", "react"}
    )


def _contains_metric(text: str) -> bool:
    lowered = text.lower()
    if "%" in lowered:
        return True
    return bool(re.search(r"\b\d+(\.\d+)?\s*(ms|s|x|k|m|%|percent|users|services|pipelines|hours|days)\b", lowered))


def _contains_date(text: str) -> bool:
    return bool(re.search(r"\b(19|20)\d{2}\b", text))


def _normalize_section(section: str) -> str:
    section = section.lower()
    if section in {"project", "projects"}:
        return "projects"
    if section in {"certification", "certifications", "certificate"}:
        return "certifications"
    return section
