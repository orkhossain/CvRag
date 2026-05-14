"""Evaluation runner for CvRag RAG system.

Usage:
    python tests/evals/run_evals.py              # run intent + clarification + retrieval
    python tests/evals/run_evals.py --all        # include answer generation tests (requires LLM)
    python tests/evals/run_evals.py --suite intent
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Allow running from project root
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVALS_DIR = Path(__file__).parent

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def load_json(name: str) -> list[dict]:
    path = EVALS_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pass(test_id: str, msg: str = "") -> dict:
    print(f"  PASS  {test_id}" + (f"  — {msg}" if msg else ""))
    return {"id": test_id, "passed": True, "message": msg}


def _fail(test_id: str, msg: str) -> dict:
    print(f"  FAIL  {test_id}  — {msg}")
    return {"id": test_id, "passed": False, "message": msg}


def print_header(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")


def print_summary(suite: str, results: list[dict]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    print(f"\n  {suite}: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)", end="")
    print()


# ──────────────────────────────────────────
# Suite: Intent detection
# ──────────────────────────────────────────

def run_intent_suite() -> list[dict]:
    from cvrag.rag.intent import detect_query_intent

    print_header("Intent Detection")
    cases = load_json("intent_cases.json")
    results = []
    for case in cases:
        got = detect_query_intent(case["query"])
        expected = case["expected_intent"]
        if got == expected:
            results.append(_pass(case["id"], f"{got}"))
        else:
            results.append(_fail(case["id"], f"expected={expected}  got={got}  query={case['query']!r}"))
    print_summary("Intent", results)
    return results


# ──────────────────────────────────────────
# Suite: Clarification
# ──────────────────────────────────────────

def run_clarification_suite() -> list[dict]:
    from cvrag.rag.clarification import build_clarification_response

    print_header("Clarification Gate")
    cases = load_json("clarification_cases.json")
    results = []
    for case in cases:
        response = build_clarification_response(case["query"], case["intent"])
        triggered = response is not None
        expected = case["should_clarify"]
        if triggered == expected:
            label = "triggered" if triggered else "not triggered"
            results.append(_pass(case["id"], label))
        else:
            expected_label = "should trigger" if expected else "should not trigger"
            got_label = "triggered" if triggered else "not triggered"
            results.append(_fail(case["id"], f"{expected_label} but {got_label}  query={case['query']!r}"))
    print_summary("Clarification", results)
    return results


# ──────────────────────────────────────────
# Suite: Retrieval
# ──────────────────────────────────────────

def run_retrieval_suite() -> list[dict]:
    print_header("Retrieval")
    cases = load_json("retrieval_cases.json")
    results = []

    try:
        from cvrag.rag.retriever import get_retriever, init_retriever
        retriever = get_retriever()
        if retriever is None:
            retriever = init_retriever()
        if retriever is None:
            print("  SKIP  — no retriever available (CV data not loaded)")
            return []
    except Exception as exc:
        print(f"  SKIP  — retriever init failed: {exc}")
        return []

    for case in cases:
        try:
            docs = retriever.invoke(case["query"])
            retrieved_sections = [
                (doc.metadata or {}).get("section", "unknown").lower()
                for doc in docs
            ]
            retrieved_text = " ".join(doc.page_content.lower() for doc in docs)

            issues = []

            must_include_any = case.get("must_include_any", [])
            if must_include_any:
                if not any(term.lower() in retrieved_text for term in must_include_any):
                    issues.append(f"none of {must_include_any!r} found in retrieved text")

            if issues:
                results.append(_fail(case["id"], "; ".join(issues)))
            else:
                results.append(_pass(case["id"], f"sections={list(set(retrieved_sections))}"))
        except Exception as exc:
            results.append(_fail(case["id"], f"error: {exc}"))

    print_summary("Retrieval", results)
    return results


# ──────────────────────────────────────────
# Suite: Answer generation (requires LLM)
# ──────────────────────────────────────────

def run_answer_suite() -> list[dict]:
    print_header("Answer Generation (LLM required)")
    cases = load_json("answer_cases.json")
    results = []

    try:
        from cvrag.rag.context import retrieve_context
        from cvrag.rag.graph import get_agent
    except Exception as exc:
        print(f"  SKIP  — import failed: {exc}")
        return []

    agent = get_agent()

    for case in cases:
        try:
            start = time.time()
            result = agent.invoke(
                {"query": case["query"], "messages": []},
                config={"configurable": {"thread_id": f"eval_{case['id']}"}},
            )
            latency_ms = int((time.time() - start) * 1000)
            answer = result.get("answer", "")

            issues = []

            must_not_include = case.get("must_not_include", [])
            for phrase in must_not_include:
                if phrase.lower() in answer.lower():
                    issues.append(f"answer contains forbidden phrase: {phrase!r}")

            must_include_any = case.get("must_include_any", [])
            if must_include_any:
                if not any(phrase.lower() in answer.lower() for phrase in must_include_any):
                    issues.append(f"answer missing all of {must_include_any!r}")

            max_words = case.get("max_words")
            if max_words:
                word_count = len(answer.split())
                if word_count > max_words:
                    issues.append(f"answer too long: {word_count} words > {max_words}")

            if issues:
                results.append(_fail(case["id"], "; ".join(issues)))
            else:
                word_count = len(answer.split())
                results.append(_pass(case["id"], f"{word_count} words  {latency_ms}ms"))
        except Exception as exc:
            results.append(_fail(case["id"], f"error: {exc}"))

    print_summary("Answer", results)
    return results


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="CvRag eval runner")
    parser.add_argument(
        "--suite",
        choices=["intent", "clarification", "retrieval", "answer", "all"],
        default="all",
        help="Which eval suite to run (default: all except answer)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="include_answer",
        help="Include answer generation tests (requires LLM)",
    )
    args = parser.parse_args()

    all_results: list[dict] = []

    if args.suite in ("intent", "all"):
        all_results += run_intent_suite()

    if args.suite in ("clarification", "all"):
        all_results += run_clarification_suite()

    if args.suite in ("retrieval", "all"):
        all_results += run_retrieval_suite()

    if args.suite == "answer" or args.include_answer:
        all_results += run_answer_suite()

    if not all_results:
        print("\nNo tests ran.")
        return 0

    total = len(all_results)
    passed = sum(1 for r in all_results if r["passed"])
    failed = total - passed

    print(f"\n{'═' * 50}")
    print(f"  TOTAL: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — all green")
    print(f"{'═' * 50}\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
