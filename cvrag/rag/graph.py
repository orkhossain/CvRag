import json
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .context import retrieve_context
from ..core.cv_data import get_section, load_cv_data
from .intent import detect_query_intent
from .language import resolve_language
from .llm import get_llm
from .prompts import PROMPTS
from .tools import (
    get_availability,
    get_certifications,
    get_contact_info,
    get_cv_section,
    get_cv_sections,
    get_cv_stats,
    get_cv_skills,
    get_projects,
    get_references,
    get_skills_matrix,
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    intent: str
    context: str
    answer: str
    context_used: int
    query_enhanced: bool
    error: bool
    language: str


def detect_intent_node(state: AgentState) -> AgentState:
    return {"intent": detect_query_intent(state["query"])}


def retrieve_context_node(state: AgentState) -> AgentState:
    intent = state.get("intent") or detect_query_intent(state["query"])
    return retrieve_context(state["query"], intent)


TOOLS = [
    get_cv_sections,
    get_cv_section,
    get_cv_stats,
    get_cv_skills,
    get_skills_matrix,
    get_contact_info,
    get_availability,
    get_certifications,
    get_references,
    get_projects,
]


def generate_response_node(state: AgentState) -> AgentState:
    context = (state.get("context") or "").strip()
    if not context or context in {"No context available", "Error retrieving context"}:
        return {
            "answer": (
                "I can only answer questions about the CV. "
                "Please upload your CV using /set-cv or set CV_PATH to a cv.json/cv.pdf and restart."
            ),
            "error": True,
        }

    if state.get("intent") == "skills_matrix":
        cv = load_cv_data() or {}
        skills = get_section(cv, "skills", "skill") if isinstance(cv, dict) else None
        if not skills:
            skills = _extract_skills_from_context(context)
        if not skills:
            answer = "I don't have that information in the context."
            return {"answer": answer, "messages": [AIMessage(content=answer)]}

        if isinstance(skills, list):
            rendered = ", ".join(
                str(item.get("name", item)) if isinstance(item, dict) else str(item)
                for item in skills
            )
        else:
            rendered = str(skills)

        answer = f"Technical skills: {rendered}"
        return {"answer": answer, "messages": [AIMessage(content=answer)]}

    prompt = PROMPTS.get(state.get("intent"), PROMPTS["general_qa"])
    history = state.get("messages", [])
    if history and isinstance(history[-1], HumanMessage):
        chat_history = history[:-1]
    else:
        chat_history = history
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
        if isinstance(response, AIMessage):
            return {"answer": response.content, "messages": [response]}
        return {"answer": str(response), "messages": [AIMessage(content=str(response))]}
    except Exception as exc:
        return {
            "answer": (
                "I apologize, but I encountered an error processing your request: "
                f"{exc}"
            ),
            "error": True,
        }


def build_agent():
    memory = MemorySaver()
    graph = StateGraph(AgentState)
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")

    def route_after_agent(state: AgentState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return END
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph.add_conditional_edges("generate_response", route_after_agent)
    graph.add_edge("tools", "generate_response")

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

    # De-dup while preserving order
    seen = set()
    deduped = []
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
