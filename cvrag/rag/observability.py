from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RagRunLog:
    timestamp: str
    query: str
    intent: str | None = None
    intent_confidence: float | None = None
    matched_intent_rule: str | None = None
    enhanced_query: str | None = None
    secondary_query: str | None = None
    retrieved_sections: list[str] | None = None
    retrieved_doc_count: int | None = None
    scores: list[float] | None = None
    retrieval_confidence: float | None = None
    retrieval_confidence_reasons: list[str] | None = None
    clarification_triggered: bool = False
    grounding_passed: bool | None = None
    unsupported_tech: list[str] | None = None
    unsupported_years: list[str] | None = None
    provider: str | None = None
    latency_ms: int | None = None
    session_id: str | None = None
    language: str | None = None
    extra: dict[str, Any] | None = None


def write_rag_log(log: RagRunLog, path: str = "logs/rag_runs.jsonl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(log), ensure_ascii=False) + "\n")


class RagTimer:
    """Context manager that records elapsed ms."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self.elapsed_ms: int = 0

    def __enter__(self) -> "RagTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)
