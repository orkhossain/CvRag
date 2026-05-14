# CV RAG System Upgrade Plan — Implementation Spec

## Goal

Upgrade the current CV RAG system from a strong prototype/early production implementation to a robust, testable, observable, and safer production-grade system.

Target outcome:

- More reliable intent detection
- Better retrieval relevance
- Smarter clarification behavior
- Fewer hallucinated CV claims
- Clear logging and debugging
- Evaluable improvements instead of prompt guessing
- Cleaner LangGraph architecture

---

## Current Architecture Summary

The system currently contains:

```text
cvrag/rag/
  llm.py
  embeddings.py
  intent.py
  retriever.py
  context.py
  formatting.py
  graph.py
  prompts.py
  clarification.py
```

Current strengths:

- Lazy singleton initialization for LLM and embeddings
- Multiple LLM providers: Groq, Ollama, HuggingFace
- Multilingual embeddings
- Rule-based intent detection
- Query enhancement before vector search
- Hybrid retrieval using FAISS MMR + lexical search + metadata reranking
- LangGraph workflow with memory
- Deterministic skills matrix route
- Clarification gate before generation

Current weaknesses to fix:

- No evaluation framework
- No observability/logging of RAG decisions
- Fixed secondary retrieval strings
- ToolNode is wired but passive
- Clarification can misfire
- Intent detection is brittle
- Deduplication uses exact content matching
- No answer groundedness verification
- Context is mostly string-based instead of structured

---

# Phase 1 — Add Evaluation Framework

## Objective

Create a repeatable eval suite before making major changes.

## Add Files

```text
tests/evals/
  intent_cases.json
  retrieval_cases.json
  clarification_cases.json
  answer_cases.json
  run_evals.py
```

## Example: `tests/evals/intent_cases.json`

```json
[
  {
    "id": "intent_001",
    "query": "Write a cover letter for a backend developer role",
    "expected_intent": "cover_letter"
  },
  {
    "id": "intent_002",
    "query": "Tell me about a time I solved a difficult technical problem",
    "expected_intent": "star_examples"
  },
  {
    "id": "intent_003",
    "query": "What cloud technologies do I know?",
    "expected_intent": "skills_matrix"
  }
]
```

## Example: `tests/evals/clarification_cases.json`

```json
[
  {
    "id": "clarify_001",
    "query": "Write a cover letter",
    "intent": "cover_letter",
    "should_clarify": true
  },
  {
    "id": "clarify_002",
    "query": "Write a cover letter for a backend developer role focused on AWS and Docker",
    "intent": "cover_letter",
    "should_clarify": false
  }
]
```

## Example: `tests/evals/retrieval_cases.json`

```json
[
  {
    "id": "retrieval_001",
    "query": "Tell me about my AWS experience",
    "expected_sections": ["experience", "skills", "projects"],
    "must_include_any": ["AWS", "cloud", "infrastructure"]
  }
]
```

## Example: `tests/evals/answer_cases.json`

```json
[
  {
    "id": "answer_001",
    "query": "Create a recruiter pitch for a backend role",
    "must_not_include": ["fake company", "invented certification"],
    "max_words": 160
  }
]
```

## Implement `run_evals.py`

The eval runner should:

1. Load all JSON files.
2. Run intent detection tests.
3. Run clarification tests.
4. Run retrieval tests.
5. Optionally run answer generation tests.
6. Print a summary table.
7. Exit with non-zero status if critical tests fail.

## Acceptance Criteria

- `python tests/evals/run_evals.py` works.
- Eval files are easy to extend.
- Intent and clarification tests can run without calling an LLM.
- Retrieval tests can run against the existing vector store.
- The output clearly shows pass/fail by test ID.

---

# Phase 2 — Add Observability Logs

## Objective

Log every RAG request so failures can be debugged.

## Add File

```text
cvrag/rag/observability.py
```

## Required API

```python
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any


@dataclass
class RagRunLog:
    timestamp: str
    query: str
    intent: str | None = None
    enhanced_query: str | None = None
    secondary_query: str | None = None
    retrieved_doc_ids: list[str] | None = None
    retrieved_sections: list[str] | None = None
    scores: list[float] | None = None
    retrieval_confidence: float | None = None
    clarification_triggered: bool = False
    provider: str | None = None
    latency_ms: int | None = None
    extra: dict[str, Any] | None = None


def write_rag_log(log: RagRunLog, path: str = "logs/rag_runs.jsonl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(log), ensure_ascii=False) + "\n")
```

## Integrate Logging

Log at the end of each graph run or inside `generate_response_node`.

Capture:

- original query
- detected intent
- enhanced query
- secondary query
- retrieved document IDs
- retrieved sections
- scores
- retrieval confidence
- clarification status
- provider
- latency

## Acceptance Criteria

- Each request appends one JSON line to `logs/rag_runs.jsonl`.
- Logs do not include secrets or full private CV content unless explicitly needed.
- Logs include enough information to debug bad retrieval.

---

# Phase 3 — Fix or Remove ToolNode

## Objective

Avoid dead architecture.

## Current Issue

The graph includes a ToolNode, but the LLM is not bound to tools.

Current graph shape:

```text
detect_intent -> retrieve_context -> generate_response
                                      ↓
                                    tools?
```

But generation does not use:

```python
get_llm().bind_tools(TOOLS)
```

## Option A — Remove ToolNode

Use this if tools are not needed.

Expected graph:

```text
START
  ↓
detect_intent
  ↓
retrieve_context
  ↓
generate_response
  ↓
END
```

## Option B — Properly Bind Tools

Use this only if tools are actually needed.

```python
llm_with_tools = get_llm().bind_tools(TOOLS)
```

Then use the bound model in generation.

## Recommendation

Choose Option A unless there are real tools that improve the CV assistant.

## Acceptance Criteria

- No dead conditional edge remains.
- If ToolNode remains, the LLM can actually emit tool calls.
- Graph behavior is covered by at least one test.

---

# Phase 4 — Improve Deduplication

## Objective

Replace exact content matching with normalized and metadata-aware dedupe.

## Add to `context.py` or new `dedupe.py`

```python
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
    normalized = normalize_doc_text(doc.page_content)

    if source and chunk_id:
        return f"{source}:{chunk_id}"

    return f"{section}:{normalized[:300]}"


def dedupe_documents(docs: list[Document]) -> list[Document]:
    seen = set()
    unique = []

    for doc in docs:
        identity = get_doc_identity(doc)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(doc)

    return unique
```

## Acceptance Criteria

- Duplicate chunks from primary and secondary retrieval are removed.
- Same source/chunk ID is preferred when metadata exists.
- Formatting still receives a clean list of documents.

---

# Phase 5 — Dynamic Secondary Retrieval

## Objective

Replace fixed secondary retrieval strings with query-aware retrieval expansion.

## Add Function

Add to `context.py` or `intent.py`:

```python
def build_secondary_query(intent: str, profile) -> str:
    terms: list[str] = []

    if intent in {"cover_letter", "role_targeting", "recruiter_pitch", "interview_prep"}:
        terms += ["achievement", "impact", "leadership", "results", "metrics"]

    if intent in {"technical_deepdive", "project_deep_dives"}:
        terms += ["architecture", "implementation", "technical", "stack", "technologies"]

    if intent in {"star_examples", "star_bank"}:
        terms += ["led", "implemented", "improved", "reduced", "increased", "challenge", "result"]

    if intent == "skills_matrix":
        terms += ["skills", "technologies", "tools", "programming", "frameworks"]

    terms += list(getattr(profile, "entities", []) or [])
    terms += list(getattr(profile, "sections", []) or [])

    if getattr(profile, "wants_metrics", False):
        terms += ["quantified", "impact", "result", "metric"]

    if getattr(profile, "wants_dates", False):
        terms += ["date", "year", "recent", "current"]

    return " ".join(dict.fromkeys(terms))
```

## Update `retrieve_context`

Current:

```python
secondary_query = FIXED_AUGMENTATION[intent]
```

Replace with:

```python
profile = build_query_profile(query)
secondary_query = build_secondary_query(intent, profile)
```

## Acceptance Criteria

- Secondary query includes user-specific entities.
- Technical queries pull technical context.
- Role/cover queries still pull impact/achievement context.
- Log secondary query in observability.

---

# Phase 6 — Retrieval Confidence Scoring

## Objective

Calculate whether retrieved context is strong enough before generation.

## Add Dataclass

```python
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class RetrievalBundle:
    docs: list[Document]
    confidence: float
    reasons: list[str]
    scores: list[float]
```

## Add Function

```python
def compute_retrieval_confidence(docs: list[Document], profile) -> tuple[float, list[str]]:
    if not docs:
        return 0.0, ["no_documents"]

    reasons = []
    score = 0.0

    top_docs = docs[:5]
    metadata_sections = [
        (doc.metadata or {}).get("section", "").lower()
        for doc in top_docs
    ]

    text = " ".join(doc.page_content.lower() for doc in top_docs)

    entity_matches = [
        entity for entity in getattr(profile, "entities", [])
        if entity.lower() in text
    ]

    section_matches = [
        section for section in getattr(profile, "sections", [])
        if section.lower() in metadata_sections or section.lower() in text
    ]

    if entity_matches:
        score += 0.35
        reasons.append("entity_match")

    if section_matches:
        score += 0.25
        reasons.append("section_match")

    if len(docs) >= 5:
        score += 0.15
        reasons.append("enough_documents")

    if getattr(profile, "wants_metrics", False) and any(char.isdigit() for char in text):
        score += 0.15
        reasons.append("metric_or_number_present")

    if getattr(profile, "wants_dates", False) and any(year in text for year in ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]):
        score += 0.10
        reasons.append("date_present")

    return min(score, 1.0), reasons
```

## Use Confidence

In `retrieve_context`, return confidence in state:

```python
state["retrieval_confidence"] = confidence
state["retrieval_confidence_reasons"] = reasons
```

Update `AgentState`.

## Acceptance Criteria

- Retrieval confidence is available in graph state.
- Confidence is logged.
- Low-confidence retrieval can influence clarification.

---

# Phase 7 — Smarter Clarification Gate

## Objective

Clarification should use intent confidence, query specificity, and retrieval confidence.

## Update Intent Result

Change intent detection to optionally return confidence.

```python
from dataclasses import dataclass


@dataclass
class IntentResult:
    intent: str
    confidence: float
    matched_rule: str | None = None
```

Keep backward compatibility:

```python
def detect_query_intent(query: str) -> str:
    return detect_query_intent_with_confidence(query).intent
```

## Add Specificity Score

In `clarification.py`:

```python
def query_specificity_score(query: str) -> float:
    score = 0.0
    q = query.lower()

    if len(q.split()) >= 8:
        score += 0.25

    if any(role in q for role in ["engineer", "developer", "architect", "manager", "analyst"]):
        score += 0.25

    if any(tech in q for tech in ["aws", "docker", "kubernetes", "python", "java", "react", "fastapi"]):
        score += 0.25

    if " at " in q or " for " in q:
        score += 0.15

    if len(q) > 180:
        score += 0.10

    return min(score, 1.0)
```

## New Clarification Decision

```python
def should_clarify(
    query: str,
    intent: str,
    retrieval_confidence: float | None = None,
    intent_confidence: float | None = None,
) -> bool:
    recruiter_like_intents = {
        "role_targeting",
        "cover_letter",
        "interview_prep",
        "recruiter_pitch",
    }

    if intent not in recruiter_like_intents:
        return False

    specificity = query_specificity_score(query)
    retrieval_confidence = retrieval_confidence or 0.0
    intent_confidence = intent_confidence or 1.0

    return (
        specificity < 0.45
        and retrieval_confidence < 0.60
        and intent_confidence >= 0.50
    )
```

## Acceptance Criteria

- Vague recruiter-like requests still trigger clarification.
- Specific role/focus requests do not trigger unnecessary clarification.
- Existing clarification response format remains compatible.

---

# Phase 8 — Hybrid Intent Router

## Objective

Keep fast rules, add an LLM fallback only when rules are uncertain.

## Rule Router

Update `intent.py`:

```python
def detect_query_intent_with_confidence(query: str) -> IntentResult:
    q = query.lower()

    if "cover letter" in q:
        return IntentResult("cover_letter", 0.95, "cover_letter")

    if "tell me about a time" in q or "behavioral" in q:
        return IntentResult("star_examples", 0.90, "star_examples")

    if "architecture" in q or "deep dive" in q:
        return IntentResult("technical_deepdive", 0.85, "technical_deepdive")

    if "interview" in q:
        return IntentResult("interview_prep", 0.80, "interview")

    if "skill" in q or "stack" in q:
        return IntentResult("skills_matrix", 0.80, "skills")

    return IntentResult("general_qa", 0.55, None)
```

## LLM Fallback

Add:

```python
def llm_intent_router(query: str) -> IntentResult:
    # Use get_llm() with a small structured prompt.
    # Return one of the allowed intents only.
    # If parsing fails, return general_qa with low confidence.
    ...
```

Use only when:

```python
result = detect_query_intent_with_confidence(query)

if result.confidence < 0.70:
    result = llm_intent_router(query)
```

## Allowed Intents

```python
ALLOWED_INTENTS = {
    "role_targeting",
    "cover_letter",
    "star_examples",
    "technical_deepdive",
    "interview_prep",
    "recruiter_pitch",
    "skills_matrix",
    "general_qa",
    "role_fit_matcher",
    "project_deep_dives",
    "star_bank",
    "quick_summary",
}
```

## Acceptance Criteria

- Rule detection still handles obvious cases.
- Ambiguous cases can use LLM fallback.
- LLM fallback cannot produce arbitrary intent names.
- Existing API-only intents remain supported.

---

# Phase 9 — Groundedness Verification

## Objective

Reduce hallucinated CV claims.

## Add File

```text
cvrag/rag/grounding.py
```

## First Version

Do a lightweight deterministic check.

```python
import re


TECH_TERMS = {
    "aws", "azure", "gcp", "docker", "kubernetes", "python", "java",
    "react", "angular", "fastapi", "django", "postgresql", "mongodb",
    "terraform", "pulumi", "github actions", "gitlab", "ci/cd"
}


def extract_years(text: str) -> set[str]:
    return set(re.findall(r"\b20\d{2}\b", text))


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def extract_tech_terms(text: str) -> set[str]:
    lower = text.lower()
    return {term for term in TECH_TERMS if term in lower}


def verify_answer_grounding(answer: str, context: str) -> dict:
    answer_terms = extract_tech_terms(answer)
    context_terms = extract_tech_terms(context)

    answer_years = extract_years(answer)
    context_years = extract_years(context)

    unsupported_tech = sorted(answer_terms - context_terms)
    unsupported_years = sorted(answer_years - context_years)

    return {
        "is_grounded": not unsupported_tech and not unsupported_years,
        "unsupported_tech": unsupported_tech,
        "unsupported_years": unsupported_years,
    }
```

## Use in Generation

After generating an answer:

```python
grounding = verify_answer_grounding(answer, context)

if not grounding["is_grounded"]:
    # Regenerate once with stricter prompt or remove unsupported facts.
```

## Prompt Addition

Add this to the system prompt:

```text
Only use facts present in the provided CV context.
Do not invent companies, roles, dates, metrics, technologies, certifications, or project details.
If information is missing, say that it is not available in the CV context.
```

## Acceptance Criteria

- Unsupported technologies and years are detected.
- System regenerates once or flags unsafe answer.
- No infinite regeneration loops.
- Grounding result is logged.

---

# Phase 10 — Structured Context Layer

## Objective

Keep both structured context and formatted text context.

## Add Dataclass

```python
from dataclasses import dataclass


@dataclass
class ContextItem:
    section: str
    content: str
    source: str | None = None
    chunk_id: str | None = None
    company: str | None = None
    dates: str | None = None
    score: float | None = None
```

## Add Converter

```python
def document_to_context_item(doc, score: float | None = None) -> ContextItem:
    metadata = doc.metadata or {}

    return ContextItem(
        section=metadata.get("section", "unknown"),
        content=doc.page_content,
        source=metadata.get("source"),
        chunk_id=metadata.get("chunk_id"),
        company=metadata.get("company"),
        dates=metadata.get("dates"),
        score=score,
    )
```

## Keep Existing Format

Existing prompt input should still receive:

```python
context = format_docs(docs)
```

But graph state should also include:

```python
structured_context: list[ContextItem]
```

## Acceptance Criteria

- Existing prompts do not break.
- Deterministic endpoints can use structured context.
- Logs can include chunk IDs and sections without full content.

---

# Phase 11 — Prompt Hardening

## Objective

Make prompts stricter without overengineering them.

## Add to Shared System Prompt

```text
You are a CV assistant.
Use only the provided CV context.
Do not invent facts.
Do not invent company names, job titles, dates, metrics, certifications, education, or technologies.
If the user asks for something not present in the context, say what is missing and provide the best available answer.
Keep the answer in the requested language: {language}.
```

## Add Intent-Specific Constraints

### Cover Letter

```text
Do not invent the recipient company unless the user provided one.
Do not claim exact years, numbers, or certifications unless present in context.
```

### Recruiter Pitch

```text
Prioritize verified experience, measurable impact, and relevant skills from context.
Do not oversell unsupported claims.
```

### STAR Examples

```text
Use only events supported by the context.
If the context lacks a complete STAR story, build the closest supported version and mention that details are limited.
```

## Acceptance Criteria

- Prompt still produces natural answers.
- Hallucinated specifics are reduced.
- JSON-only intents still return valid JSON.

---

# Phase 12 — Final Integration Checklist

## Update `AgentState`

Add:

```python
intent_confidence: float
matched_intent_rule: str | None
retrieval_confidence: float
retrieval_confidence_reasons: list[str]
secondary_query: str
structured_context: list
grounding: dict
```

## Update Flow

Final desired flow:

```text
START
  ↓
detect_intent_with_confidence
  ↓
build_query_profile
  ↓
enhance_query
  ↓
primary retrieval
  ↓
dynamic secondary retrieval
  ↓
dedupe
  ↓
rerank
  ↓
compute retrieval confidence
  ↓
clarification gate
  ↓
generate answer or deterministic response
  ↓
groundedness check
  ↓
log run
  ↓
END
```

## Backward Compatibility

Keep these public function names working:

```python
detect_query_intent(query) -> str
enhance_query(query, intent) -> str
retrieve_context(query, intent) -> str
get_llm()
get_embeddings()
```

If richer versions are added, use new names:

```python
detect_query_intent_with_confidence()
retrieve_context_bundle()
```

---

# Suggested Implementation Order

Implement in this order:

1. Add eval files and runner.
2. Add observability logging.
3. Fix or remove ToolNode.
4. Improve deduplication.
5. Add dynamic secondary retrieval.
6. Add retrieval confidence.
7. Update clarification gate.
8. Add hybrid intent router.
9. Add groundedness verification.
10. Add structured context.
11. Harden prompts.
12. Run evals and tune.

---

# Testing Requirements

## Unit Tests

Add tests for:

```text
intent.py
clarification.py
dedupe.py
context.py
grounding.py
```

## Integration Tests

Add tests for:

- role targeting query
- cover letter query
- technical deep dive query
- skills matrix query
- vague recruiter query
- multilingual query

## Regression Tests

Every bug found through logs should become an eval case.

---

# Done Criteria

The upgrade is complete when:

- Eval runner exists and passes.
- RAG logs are generated.
- No dead ToolNode remains.
- Secondary retrieval is dynamic.
- Retrieval confidence exists.
- Clarification uses confidence and specificity.
- Intent routing has confidence.
- Grounding verifier catches unsupported facts.
- Existing API behavior is not broken.
- At least 30 eval cases exist.
- Prompt outputs remain formatted correctly.
- The system can explain why it retrieved the context it used.

---

# Notes for Claude

Please implement this incrementally.

For each phase:

1. Modify only the necessary files.
2. Keep backward compatibility.
3. Add or update tests.
4. Run the relevant tests.
5. Avoid broad rewrites unless required.
6. Prefer small, reviewable commits.
7. Do not change public API behavior unless the spec explicitly says so.
8. If existing code differs from this spec, adapt the implementation while preserving the intent.

The most important priorities are:

1. Evaluation
2. Observability
3. Retrieval quality
4. Clarification accuracy
5. Hallucination reduction
