import re
import time
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .clarification import build_clarification_response
from .context import retrieve_context
from ..core.config import LLM_PROVIDER
from ..core.cv_data import get_section, load_cv_data
from .grounding import verify_answer_grounding
from .intent import detect_query_intent_with_confidence
from .language import resolve_language
from .llm import get_llm
from .observability import RagRunLog, RagTimer, write_rag_log
from .prompts import PROMPTS


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    intent: str
    intent_confidence: float
    matched_intent_rule: str | None
    context: str
    answer: str
    context_used: int
    query_enhanced: bool
    secondary_query: str
    retrieval_confidence: float
    retrieval_confidence_reasons: list[str]
    error: bool
    language: str
    needs_clarification: bool
    clarifying_question: str
    follow_up_options: list[str]
    grounding: dict
    structured_context: list
    session_id: str
    _start_time: float


def detect_intent_node(state: AgentState) -> AgentState:
    result = detect_query_intent_with_confidence(state["query"])
    return {
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "matched_intent_rule": result.matched_rule,
        "_start_time": time.perf_counter(),
    }


def retrieve_context_node(state: AgentState) -> AgentState:
    intent = state.get("intent") or detect_query_intent_with_confidence(state["query"]).intent
    return retrieve_context(state["query"], intent)


def generate_response_node(state: AgentState) -> AgentState:
    retrieval_confidence = state.get("retrieval_confidence")
    intent_confidence = state.get("intent_confidence")

    clarification = build_clarification_response(
        state.get("query", ""),
        state.get("intent", "general_qa"),
        retrieval_confidence=retrieval_confidence,
        intent_confidence=intent_confidence,
    )
    if clarification:
        answer = str(clarification["answer"])
        _log_run(state, answer=answer, clarification_triggered=True)
        return {
            "answer": answer,
            "needs_clarification": True,
            "clarifying_question": str(clarification["clarifying_question"]),
            "follow_up_options": list(clarification["follow_up_options"]),
            "messages": [AIMessage(content=answer)],
        }

    context = (state.get("context") or "").strip()
    if not context or context in {"No context available", "Error retrieving context"}:
        answer = (
            "I can only answer questions about the CV. "
            "Please upload your CV using /set-cv or set CV_PATH to a cv.json/cv.pdf and restart."
        )
        _log_run(state, answer=answer)
        return {"answer": answer, "error": True}

    if state.get("intent") == "skills_matrix":
        cv = load_cv_data() or {}
        skills = get_section(cv, "skills", "skill") if isinstance(cv, dict) else None
        if not skills:
            skills = _extract_skills_from_context(context)
        if not skills:
            answer = "I don't have that information in the context."
            _log_run(state, answer=answer)
            return {"answer": answer, "messages": [AIMessage(content=answer)]}

        if isinstance(skills, list):
            rendered = ", ".join(
                str(item.get("name", item)) if isinstance(item, dict) else str(item)
                for item in skills
            )
        else:
            rendered = str(skills)

        answer = f"Technical skills: {rendered}"
        _log_run(state, answer=answer)
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    prompt = PROMPTS.get(state.get("intent"), PROMPTS["general_qa"])
    history = state.get("messages", [])
    chat_history = history[:-1] if history and isinstance(history[-1], HumanMessage) else history
    chain = (
        {
            "context": lambda _: state.get("context", ""),
            "question": lambda _: state["query"],
            "chat_history": lambda _: chat_history,
            "language": lambda _: resolve_language(
                state.get("language"),
                [state.get("query", ""), state.get("context", "")],
            ),
        }
        | prompt
        | get_llm()
    )

    try:
        response = chain.invoke({})
        answer = response.content if isinstance(response, AIMessage) else str(response)
        ai_msg = response if isinstance(response, AIMessage) else AIMessage(content=answer)

        grounding = verify_answer_grounding(answer, context)
        _log_run(state, answer=answer, grounding=grounding)

        return {
            "answer": answer,
            "grounding": grounding,
            "messages": [ai_msg],
        }
    except Exception as exc:
        answer = f"I apologize, but I encountered an error processing your request: {exc}"
        _log_run(state, answer=answer)
        return {"answer": answer, "error": True}


def _log_run(
    state: AgentState,
    *,
    answer: str = "",
    clarification_triggered: bool = False,
    grounding: dict | None = None,
) -> None:
    start = state.get("_start_time") or time.perf_counter()
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    structured = state.get("structured_context") or []
    sections = list({item.section for item in structured if hasattr(item, "section")}) if structured else None

    log = RagRunLog(
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=state.get("query", ""),
        intent=state.get("intent"),
        intent_confidence=state.get("intent_confidence"),
        matched_intent_rule=state.get("matched_intent_rule"),
        enhanced_query=None,
        secondary_query=state.get("secondary_query"),
        retrieved_sections=sections,
        retrieved_doc_count=state.get("context_used"),
        retrieval_confidence=state.get("retrieval_confidence"),
        retrieval_confidence_reasons=state.get("retrieval_confidence_reasons"),
        clarification_triggered=clarification_triggered,
        grounding_passed=grounding.get("is_grounded") if grounding else None,
        unsupported_tech=grounding.get("unsupported_tech") if grounding else None,
        unsupported_years=grounding.get("unsupported_years") if grounding else None,
        provider=LLM_PROVIDER,
        latency_ms=elapsed_ms,
        session_id=state.get("session_id"),
        language=state.get("language"),
    )
    try:
        write_rag_log(log)
    except Exception:
        pass


def build_agent():
    memory = MemorySaver()
    graph = StateGraph(AgentState)
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_response", generate_response_node)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile(checkpointer=memory)


def _extract_skills_from_context(context: str) -> list[str]:
    if not context:
        return []

    results: list[str] = []
    for line in context.splitlines():
        line_clean = line.strip()
        if not line_clean:
            continue
        lower = line_clean.lower()
        if not any(
            key in lower
            for key in ["skills", "technologies", "tech stack", "stack", "tooling"]
        ):
            continue

        text = re.sub(r"^-\s*\[[^\]]+\]\s*", "", line_clean)
        if ":" in text:
            text = text.split(":", 1)[1]
        parts = re.split(r"[;,/|•]", text)
        for part in parts:
            item = part.strip()
            if not item:
                continue
            if any(
                item.lower().startswith(prefix)
                for prefix in ["skills", "technologies", "tech stack", "stack"]
            ):
                item = item.split(" ", 1)[-1].strip()
            if item:
                results.append(item)

    seen: set[str] = set()
    deduped: list[str] = []
    for item in results:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent
