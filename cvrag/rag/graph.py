from typing import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from .formatting import format_docs
from .intent import detect_query_intent, enhance_query
from .llm import get_llm
from .prompts import PROMPTS
from .retriever import get_retriever


class AgentState(TypedDict, total=False):
    query: str
    intent: str
    context: str
    answer: str
    context_used: int
    query_enhanced: bool
    error: bool


def detect_intent_node(state: AgentState) -> AgentState:
    return {"intent": detect_query_intent(state["query"])}


def retrieve_context_node(state: AgentState) -> AgentState:
    retriever = get_retriever()
    intent = state.get("intent") or detect_query_intent(state["query"])
    enhanced_query = enhance_query(state["query"], intent)

    if not retriever:
        return {
            "context": "No context available",
            "context_used": 0,
            "query_enhanced": enhanced_query != state["query"],
        }

    try:
        primary_docs = retriever.invoke(enhanced_query)

        secondary_docs = []
        if intent in ["role_targeting", "cover_letter", "interview_prep"]:
            secondary_docs = retriever.invoke("achievements leadership impact results metrics")
        elif intent == "technical_deepdive":
            secondary_docs = retriever.invoke(
                "technical implementation architecture technologies stack"
            )
        elif intent == "star_examples":
            secondary_docs = retriever.invoke(
                "led architected implemented improved reduced increased"
            )

        all_docs = primary_docs + secondary_docs
        seen_content = set()
        unique_docs = []

        for doc in all_docs:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                unique_docs.append(doc)

        context = format_docs(unique_docs[:15])
        return {
            "context": context,
            "context_used": len(context.split("\n")) if context else 0,
            "query_enhanced": enhanced_query != state["query"],
        }
    except Exception as exc:
        print(f"Retriever error: {exc}")
        return {
            "context": "Error retrieving context",
            "context_used": 0,
            "query_enhanced": enhanced_query != state["query"],
            "error": True,
        }


def generate_response_node(state: AgentState) -> AgentState:
    prompt = PROMPTS.get(state.get("intent"), PROMPTS["general_qa"])
    chain = (
        {"context": lambda _: state.get("context", ""), "question": lambda _: state["query"]}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    try:
        response = chain.invoke({})
        return {"answer": response}
    except Exception as exc:
        return {
            "answer": (
                "I apologize, but I encountered an error processing your request: "
                f"{exc}"
            ),
            "error": True,
        }


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("detect_intent", detect_intent_node)
    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_response", generate_response_node)

    graph.set_entry_point("detect_intent")
    graph.add_edge("detect_intent", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent
