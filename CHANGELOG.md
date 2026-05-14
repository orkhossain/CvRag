# Changelog

## [Unreleased] — 2026-05-14

### Added
- **Observability** (`cvrag/rag/observability.py`): `RagRunLog` dataclass and `write_rag_log` appending structured JSONL to `logs/rag_runs.jsonl` on every request; `RagTimer` context manager for latency tracking.
- **Deduplication** (`cvrag/rag/dedupe.py`): identity-based document deduplication using `source:chunk_id` metadata with normalized-text fallback, replacing fragile exact-string comparison.
- **Grounding verifier** (`cvrag/rag/grounding.py`): `verify_answer_grounding` detects tech terms and years in the answer that are absent from the retrieved context, returning `{is_grounded, unsupported_tech, unsupported_years}`.
- **Eval framework** (`tests/evals/`): 42 test cases across four suites (intent ×20, clarification ×14, retrieval ×8, answer ×4) plus `run_evals.py` runner with `--suite` and `--all` flags; exits non-zero on failures.

### Changed
- **Intent detection** (`cvrag/rag/intent.py`): rewritten with `IntentResult` dataclass carrying confidence and matched rule; rule-based confidence tiers (0.55–0.95) with LLM fallback only when confidence < 0.70; added `programming language(s)` → `skills_matrix` rule.
- **Retrieval** (`cvrag/rag/context.py`): added `ContextItem` and `RetrievalBundle` dataclasses; `compute_retrieval_confidence` scoring (entity/section/metric/date signals, 0–1); dynamic `build_secondary_query` incorporating user entities from `QueryProfile`; `retrieve_context` now returns confidence scores and structured context.
- **Clarification gate** (`cvrag/rag/clarification.py`): replaced ad-hoc regex checks with principled `query_specificity_score` (word count, role keywords, tech/domain terms, prepositions) and `should_clarify` gating on three signals: specificity < 0.45, retrieval confidence < 0.60, intent confidence ≥ 0.50; extended domain term list to include behavioral/soft-skill terms.
- **Graph** (`cvrag/rag/graph.py`): removed dead `ToolNode` (LLM was never bound with `.bind_tools()`); `AgentState` extended with `intent_confidence`, `matched_intent_rule`, `retrieval_confidence`, `retrieval_confidence_reasons`, `grounding`, `structured_context`, `session_id`, `_start_time`; `_log_run` called on every exit path; clean linear flow: `detect_intent → retrieve_context → generate_response → END`.
- **Prompts** (`cvrag/rag/prompts.py`): stricter system prompt prohibiting invention of company names, job titles, dates, metrics, certifications, and technologies; `cover_letter`, `star_examples`, and `recruiter_pitch` templates hardened with explicit no-hallucination constraints.
- **Spec** (`cv_rag_upgrade_plan_spec.md`): full 12-phase upgrade plan documented and committed for reference.
