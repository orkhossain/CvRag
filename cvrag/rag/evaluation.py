from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RetrievalEvalCase:
    name: str
    query: str
    expected_terms: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalEvalResult:
    name: str
    passed: bool
    matched_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]


DEFAULT_RETRIEVAL_EVAL_CASES: tuple[RetrievalEvalCase, ...] = (
    RetrievalEvalCase(
        name="technical_kubernetes",
        query="What Kubernetes migration experience do I have?",
        expected_terms=("kubernetes", "migration"),
    ),
    RetrievalEvalCase(
        name="metrics_impact",
        query="What quantified impact or performance improvements have I delivered?",
        expected_terms=("reduced", "improved"),
    ),
    RetrievalEvalCase(
        name="skills_lookup",
        query="Which cloud and infrastructure skills do I have?",
        expected_terms=("aws", "terraform"),
    ),
    RetrievalEvalCase(
        name="project_lookup",
        query="Tell me about projects involving streaming or APIs",
        expected_terms=("project", "api"),
    ),
)


def evaluate_retrieval(
    retriever,
    cases: Sequence[RetrievalEvalCase] = DEFAULT_RETRIEVAL_EVAL_CASES,
) -> list[RetrievalEvalResult]:
    results: list[RetrievalEvalResult] = []
    for case in cases:
        docs = retriever.invoke(case.query)
        haystack = " ".join(getattr(doc, "page_content", "").lower() for doc in docs)
        matched = tuple(term for term in case.expected_terms if term in haystack)
        missing = tuple(term for term in case.expected_terms if term not in haystack)
        results.append(
            RetrievalEvalResult(
                name=case.name,
                passed=not missing,
                matched_terms=matched,
                missing_terms=missing,
            )
        )
    return results


def summarize_retrieval_results(results: Iterable[RetrievalEvalResult]) -> dict[str, object]:
    items = list(results)
    passed = sum(1 for item in items if item.passed)
    failed_cases = [item.name for item in items if not item.passed]
    return {
        "total": len(items),
        "passed": passed,
        "failed": len(items) - passed,
        "failed_cases": failed_cases,
    }
