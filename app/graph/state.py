from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    Shared typed state for the multi-agent LangGraph workflow.
    Passed between Supervisor, RAG Agent, Research Agent, and Synthesis Agent.
    """
    query: str
    selected_document_ids: Optional[List[str]]
    route: str  # "rag", "research", or "both"
    supervisor_reasoning: str

    # Internal RAG Artifacts
    internal_retrieved_chunks: List[Dict[str, Any]]
    internal_answer: str

    # External Web Research Artifacts
    external_research_results: List[Dict[str, Any]]
    external_answer: str

    # Synthesized Final Deliverable
    final_answer: str
    sources: Dict[str, Any]

    # Diagnostics & Traceability
    errors: List[str]
