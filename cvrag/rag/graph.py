from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .context import retrieve_context
from .intent import detect_query_intent
from .llm import get_llm
from .prompts import PROMPTS
from .tools import (
    get_availability,
    get_certifications,
    get_contact_info,
    get_cv_section,
    get_cv_sections,
    get_cv_stats,
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


def detect_intent_node(state: AgentState) -> AgentState:
    return {"intent": detect_query_intent(state["query"])}


def retrieve_context_node(state: AgentState) -> AgentState:
    intent = state.get("intent") or detect_query_intent(state["query"])
    return retrieve_context(state["query"], intent)


TOOLS = [
    get_cv_sections,
    get_cv_section,
    get_cv_stats,
    get_skills_matrix,
    get_contact_info,
    get_availability,
    get_certifications,
    get_references,
    get_projects,
]


def generate_response_node(state: AgentState) -> AgentState:
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
        }
        | prompt
        | get_llm().bind_tools(TOOLS)
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


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent
