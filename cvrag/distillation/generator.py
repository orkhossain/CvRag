"""
Teacher-student data generation for LLM distillation.

Flow per example:
  seed query → retrieve_context → format prompt (same template as prod) →
  teacher LLM → save as ChatML JSONL
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Generator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from ..core.config import DISTILL_OUTPUT_DIR, GROQ_API_KEY, TEACHER_MODEL
from ..rag.context import retrieve_context
from ..rag.language import resolve_language
from ..rag.prompts import PROMPTS

# Intents handled by the standard generate_response_node (LLM-driven, query-only).
# Skills_matrix is excluded — it bypasses the LLM entirely.
DISTILLABLE_INTENTS: list[str] = [
    "general_qa",
    "role_targeting",
    "cover_letter",
    "star_examples",
    "technical_deepdive",
    "interview_prep",
    "recruiter_pitch",
]

SEED_QUERIES: dict[str, list[str]] = {
    "general_qa": [
        "What is my work experience?",
        "What is my educational background?",
        "What is my experience with Python?",
        "Tell me about yourself.",
        "What are your main responsibilities in your current role?",
        "How many years of experience do you have?",
        "What certifications do you hold?",
        "What languages do you speak?",
        "Describe your career so far.",
        "What is your most recent job?",
    ],
    "role_targeting": [
        "Tailor my CV for the role of Senior Engineer at Stripe",
        "I am applying to a job at Google for the role of SRE",
        "Create a targeted profile for a Machine Learning Engineer position",
        "Highlight my experience relevant to a Data Engineering role",
        "What makes me a good fit for a Platform Engineering position?",
        "Tailor my background for a Senior Backend Developer at a fintech startup",
        "How does my experience match a Cloud Architect role?",
    ],
    "cover_letter": [
        "Write a cover letter for a backend developer role",
        "Write an application letter for a DevOps position",
        "Create a cover letter for a Senior Software Engineer at a fintech company",
        "Write a compelling cover letter for a data engineering position",
        "Draft a cover letter for a Staff Engineer role at a product company",
        "Write a cover letter for a cloud infrastructure role",
    ],
    "star_examples": [
        "Tell me about a time I solved a difficult technical problem",
        "Give me a STAR example of a time I led a team",
        "Describe a situation where you improved system performance",
        "Tell me about a behavioral example of leadership",
        "Give me an example of when you had to learn something quickly under pressure",
        "Describe a challenging project and how you delivered it",
        "Tell me about a time you dealt with conflicting priorities",
        "Give me a STAR example of cross-team collaboration",
    ],
    "technical_deepdive": [
        "How did you implement the CI/CD pipeline?",
        "Explain the technical architecture of your main project",
        "Deep dive into the Kubernetes deployment you set up",
        "What was the database design for your most complex project?",
        "Explain your approach to system design and scalability",
        "Walk me through the most technically challenging thing you have built",
        "How did you handle observability and monitoring in production?",
    ],
    "interview_prep": [
        "Prepare me for a technical interview",
        "What questions should I prepare for a senior engineering interview?",
        "How should I answer questions about my greatest weakness?",
        "What behavioral interview questions should I be ready for?",
        "Help me prepare for system design interview questions",
        "What should I say when asked about my proudest achievement?",
    ],
    "recruiter_pitch": [
        "Why should we hire you?",
        "Sell yourself as the best candidate for this position",
        "Create a recruiter pitch for a backend role",
        "What makes you stand out from other candidates?",
        "Give me your 30-second elevator pitch",
        "Summarize my profile for a recruiter in 4 sentences",
    ],
}


def _get_teacher() -> ChatGroq:
    return ChatGroq(model=TEACHER_MODEL, temperature=0.3, api_key=GROQ_API_KEY)


def _lc_role(msg) -> str:
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    return getattr(msg, "type", "user")


def _format_example(intent: str, query: str, teacher) -> dict | None:
    ctx = retrieve_context(query, intent)
    context_text = (ctx.get("context") or "").strip()
    if not context_text or context_text in {
        "No context available",
        "Error retrieving context",
    }:
        return None

    prompt = PROMPTS.get(intent, PROMPTS["general_qa"])
    try:
        lc_messages = prompt.format_messages(
            context=context_text,
            question=query,
            chat_history=[],
            language=resolve_language(None, [query, context_text]),
        )
    except Exception:
        return None

    try:
        response = teacher.invoke(lc_messages)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception:
        return None

    messages = [
        {"role": _lc_role(m), "content": m.content} for m in lc_messages
    ] + [{"role": "assistant", "content": answer}]

    return {
        "messages": messages,
        "intent": intent,
        "query": query,
        "teacher_model": TEACHER_MODEL,
    }


def generate_examples(
    intents: list[str] | None = None,
    examples_per_intent: int = 5,
    augment: bool = False,
    *,
    progress_cb: "((int, int) -> None) | None" = None,
) -> Generator[dict, None, None]:
    """
    Yield training examples one by one.

    progress_cb(current, total) is called after each example attempt.
    """
    from ..rag.retriever import get_retriever, init_retriever

    if get_retriever() is None:
        init_retriever()

    selected = [i for i in (intents or DISTILLABLE_INTENTS) if i in SEED_QUERIES]
    teacher = _get_teacher()

    all_queries: list[tuple[str, str]] = []
    for intent in selected:
        seeds = SEED_QUERIES[intent]
        if augment and len(seeds) < examples_per_intent:
            seeds = seeds + _augment(seeds, examples_per_intent - len(seeds), teacher)
        sampled = random.sample(seeds, min(examples_per_intent, len(seeds)))
        all_queries.extend((intent, q) for q in sampled)

    total = len(all_queries)
    for current, (intent, query) in enumerate(all_queries, start=1):
        example = _format_example(intent, query, teacher)
        if progress_cb:
            progress_cb(current, total)
        if example:
            yield example


def _augment(seeds: list[str], n: int, teacher) -> list[str]:
    """Ask the teacher to paraphrase existing seeds to get more query variants."""
    prompt = [
        SystemMessage(
            content=(
                f"Generate {n} distinct paraphrases of the questions below. "
                "Output only the paraphrases, one per line, no numbering or extra text."
            )
        ),
        HumanMessage(content="\n".join(seeds[:5])),
    ]
    try:
        response = teacher.invoke(prompt)
        lines = [l.strip() for l in response.content.splitlines() if l.strip()]
        return lines[:n]
    except Exception:
        return []


def save_to_jsonl(examples: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return path


def output_path(task_id: str) -> Path:
    return DISTILL_OUTPUT_DIR / f"{task_id}.jsonl"
