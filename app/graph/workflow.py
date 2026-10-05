import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from app.graph.state import AgentState
from app.agents.agents import (
    supervisor_node,
    rag_agent_node,
    research_agent_node,
    synthesis_agent_node
)

logger = logging.getLogger(__name__)

def supervisor_routing_decision(state: AgentState) -> str:
    """
    Evaluates supervisor state to direct the workflow:
    - 'rag' or 'both' -> proceeds to 'rag_agent'
    - 'research' -> proceeds to 'research_agent'
    """
    route = state.get("route", "rag").lower()
    if route in ("rag", "both"):
        return "rag_agent"
    elif route == "research":
        return "research_agent"
    return "rag_agent"

def post_rag_routing_decision(state: AgentState) -> str:
    """
    After RAG completes:
    - If route is 'both', chain directly to 'research_agent'
    - If route is 'rag', jump straight to 'synthesis_agent'
    """
    route = state.get("route", "rag").lower()
    if route == "both":
        return "research_agent"
    return "synthesis_agent"

def build_graph() -> Any:
    """
    Constructs the LangGraph state machine orchestrating the Multi-Agent architecture:
    1. Supervisor -> RAG Agent -> Synthesis
    2. Supervisor -> Research Agent -> Synthesis
    3. Supervisor -> RAG Agent -> Research Agent -> Synthesis (Collaborative 'both' route)
    """
    workflow = StateGraph(AgentState)

    # Register Agent Nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("synthesis_agent", synthesis_agent_node)

    # Set Entry Point
    workflow.set_entry_point("supervisor")

    # Conditional Routing from Supervisor
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_routing_decision,
        {
            "rag_agent": "rag_agent",
            "research_agent": "research_agent"
        }
    )

    # Conditional Routing from RAG Agent
    workflow.add_conditional_edges(
        "rag_agent",
        post_rag_routing_decision,
        {
            "research_agent": "research_agent",
            "synthesis_agent": "synthesis_agent"
        }
    )

    # Direct Edge from Research Agent to Synthesis
    workflow.add_edge("research_agent", "synthesis_agent")

    # Synthesis Agent terminates workflow
    workflow.add_edge("synthesis_agent", END)

    app = workflow.compile()
    return app

# Compiled graph instance
research_graph = build_graph()

def run_research_workflow(
    query: str,
    selected_document_ids: Optional[List[str]] = None
) -> AgentState:
    """
    Initializes state and runs the compiled LangGraph workflow to completion.
    """
    initial_state: AgentState = {
        "query": query,
        "selected_document_ids": selected_document_ids,
        "route": "rag",
        "supervisor_reasoning": "",
        "internal_retrieved_chunks": [],
        "internal_answer": "",
        "external_research_results": [],
        "external_answer": "",
        "final_answer": "",
        "sources": {"internal": [], "external": []},
        "errors": []
    }

    logger.info(f"Invoking LangGraph Research Assistant Workflow for query: '{query}'")
    final_output = research_graph.invoke(initial_state)
    return final_output
