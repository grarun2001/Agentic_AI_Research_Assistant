import json
import logging
from typing import Dict, Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.graph.state import AgentState
from app.tools.rag_tool import query_internal_rag
from app.tools.web_tool import perform_web_search
from app.reliability.resilient import resilient_call

logger = logging.getLogger(__name__)

def get_llm():
    """Returns a ChatGoogleGenerativeAI instance using configured Gemini settings."""
    api_key = settings.get_gemini_api_key()
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set. Gemini calls will trigger resilience fallback.")
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=api_key or "DUMMY_KEY_FOR_FALLBACK",
        temperature=settings.GEMINI_TEMPERATURE,
        convert_system_message_to_human=True
    )

def extract_content_text(content: Any) -> str:
    """Helper to cleanly extract text from either str or list of block dicts from Gemini."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(item.text)
        return "\n".join(parts).strip()
    return str(content).strip()

def invoke_llm_resilient(messages: List[Any], fallback_text: str = "") -> str:
    """Invokes Google Gemini with circuit breaker, exponential backoff, and graceful fallback."""
    def _call():
        api_key = settings.get_gemini_api_key()
        if not api_key or api_key == "DUMMY_KEY_FOR_FALLBACK":
            raise ValueError("GEMINI_API_KEY is not configured.")
        llm = get_llm()
        response = llm.invoke(messages)
        return extract_content_text(response.content)

    def _fallback():
        logger.warning("Gemini LLM fallback activated: returning deterministic fallback response.")
        if fallback_text:
            return fallback_text
        return (
            "The AI reasoning service (Gemini) is temporarily unavailable or running in offline mode. "
            "Please verify your GEMINI_API_KEY and model configuration in .env."
        )

    return resilient_call(
        service_name="gemini",
        func=_call,
        fallback=_fallback,
        max_retries=1,
        backoff_factor=1.0
    )

# ----------------- Supervisor Agent -----------------

SUPERVISOR_PROMPT = """You are the Supervisor of an enterprise AI Research Assistant.
Your task is to analyze the user's query and decide the optimal routing:
- 'rag': The query specifically asks about internal company documentation, policies, HR, leave, attendance, employee handbook, benefits, code of conduct, or internal company guidelines.
- 'research': The query is a general knowledge question, external internet search, current world news, mathematical calculation (e.g. 12+12), science, programming/code question, or general query not specific to internal company policies.
- 'both': The query asks to compare internal company policy with external tech/industry standards or market trends.

Respond ONLY with a JSON object in this exact format:
{
  "route": "rag" | "research" | "both",
  "reasoning": "Brief 1-sentence rationale for the routing decision"
}
"""

def supervisor_node(state: AgentState) -> Dict[str, Any]:
    """
    Supervisor Agent evaluates user query to dynamically route to RAG, Research, or Both.
    """
    query = state.get("query", "").strip()
    logger.info(f"Supervisor evaluating routing for query: '{query}'")

    lower_query = query.lower()

    # Math or calculation detection (e.g., 12+12=?, calculate, math)
    is_math_or_calc = (
        any(op in query for op in ["+", "*", "/", "="])
        or any(w in lower_query for w in ["calculate", "solve", "math", "sum", "multiply", "divide", "minus", "plus", "root"])
    )

    # Company specific keywords
    has_company = any(
        w in lower_query for w in [
            "policy", "handbook", "leave", "attendance", "casual leave", "sick leave",
            "earned leave", "maternity", "paternity", "employee", "internal", "we", "our",
            "lyfngo", "acme", "hr", "benefits", "stipend", "reimbursement", "rules", "dress code",
            "conduct", "notice period", "resignation", "working hours", "probation"
        ]
    )

    has_comparison = any(
        w in lower_query for w in ["compare", "versus", "vs", "industry standard", "benchmark", "market trend"]
    )
    has_external = any(
        w in lower_query for w in ["current", "latest", "today", "2024", "2025", "2026", "global", "market", "competitor", "news"]
    )

    # Default heuristic logic
    if is_math_or_calc:
        default_route = "research"
        default_reasoning = "Query is a mathematical calculation or formula."
    elif (has_company and has_external) or has_comparison:
        default_route = "both"
        default_reasoning = "Query requires cross-referencing internal company documentation with external market trends."
    elif has_company:
        default_route = "rag"
        default_reasoning = "Routed to internal RAG based on enterprise documentation inquiry."
    else:
        default_route = "research"
        default_reasoning = "Query focuses on external research, general knowledge, or web inquiry."

    try:
        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"User Query: {query}")
        ]
        raw_output = invoke_llm_resilient(
            messages,
            fallback_text=json.dumps({"route": default_route, "reasoning": default_reasoning})
        )

        clean_json = raw_output.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        route = data.get("route", default_route).lower()
        if route not in ("rag", "research", "both"):
            route = default_route
        reasoning = data.get("reasoning", default_reasoning)
    except Exception as e:
        logger.warning(f"Supervisor LLM parse error: {e}. Using deterministic routing: {default_route}")
        route = default_route
        reasoning = default_reasoning

    logger.info(f"Supervisor Decision: Route='{route}', Reasoning='{reasoning}'")
    return {
        "route": route,
        "supervisor_reasoning": reasoning
    }

# ----------------- RAG Agent -----------------

RAG_PROMPT = """You are an expert Internal Knowledge Assistant.
Your task is to answer the user's question clearly, accurately, and conversationally based STRICTLY on the retrieved company document excerpts below.

CRITICAL INSTRUCTIONS:
1. Provide a direct, well-synthesized answer that immediately answers the user's specific question.
2. DO NOT simply dump or copy-paste raw document excerpts or repetitive snippets.
3. Summarize the rules, leave types, numbers, or policies in clear, readable bullet points or short paragraphs.
4. Ground your answer completely in the facts from the excerpts.
5. If the excerpts do not contain the answer, politely state: 'The provided internal documents do not contain information regarding this inquiry.'
"""

def rag_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    RAG Agent retrieves internal documents using hybrid search and generates a grounded response.
    """
    query = state.get("query", "")
    selected_docs = state.get("selected_document_ids")
    logger.info(f"RAG Agent executing hybrid retrieval for: '{query}'")

    chunks = query_internal_rag(query, selected_document_ids=selected_docs, top_n=5)

    if not chunks:
        logger.info("RAG Agent: 0 chunks retrieved.")
        return {
            "internal_retrieved_chunks": [],
            "internal_answer": "No relevant internal company documents were found matching this inquiry."
        }

    # Format context for LLM
    context_blocks = []
    for idx, c in enumerate(chunks, 1):
        doc_name = c.get("document_name", "Unknown Document")
        section = c.get("page_or_section", "N/A")
        text = c.get("text", "")
        context_blocks.append(f"--- Document: {doc_name} ({section}) ---\n{text}")

    context_str = "\n\n".join(context_blocks)
    prompt = f"Retrieved Internal Documents:\n{context_str}\n\nUser Question: {query}\n\nDirect, clear synthesized answer:"

    messages = [
        SystemMessage(content=RAG_PROMPT),
        HumanMessage(content=prompt)
    ]

    fallback_lines = ["**Summary of relevant internal document excerpts:**\n"]
    for c in chunks[:3]:
        fallback_lines.append(f"- **{c.get('document_name')} ({c.get('page_or_section')}):** {c.get('text')[:220]}...")
    fallback_text = "\n".join(fallback_lines)

    answer = invoke_llm_resilient(messages, fallback_text=fallback_text)

    return {
        "internal_retrieved_chunks": chunks,
        "internal_answer": answer
    }

# ----------------- Research Agent -----------------

RESEARCH_PROMPT = """You are an expert External Research and Intelligence Agent.
Your job is to provide a factual, accurate, and direct response to the user's inquiry:
1. If the user query is a mathematical calculation, arithmetic, or logical question (e.g. '12+12=?'), calculate and provide the exact mathematical answer immediately and clearly.
2. If the inquiry is about current events, people, places, technologies, or general facts, synthesize the answer based on the provided live web search results.
3. Be concise, objective, and accurate. Attribute key external facts to the web sources when applicable.
"""

def research_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Research Agent queries Tavily web search and generates a current factual summary.
    """
    query = state.get("query", "")
    logger.info(f"Research Agent querying external web for: '{query}'")

    # Fast check for simple math queries (e.g. 12+12=?)
    lower_query = query.lower().replace(" ", "").replace("=?", "").replace("=", "")
    if any(op in lower_query for op in ["+", "*", "/"]) and all(c in "0123456789+*/.-" for c in lower_query):
        try:
            # Safe evaluation of basic arithmetic
            math_val = eval(lower_query, {"__builtins__": None}, {})
            direct_ans = f"**Calculation Result:** `{query.strip()}` = **{math_val}**"
            return {
                "external_research_results": [],
                "external_answer": direct_ans
            }
        except Exception:
            pass

    web_results = perform_web_search(query, max_results=5)

    context_blocks = []
    for idx, r in enumerate(web_results, 1):
        title = r.get("title", "Web Result")
        url = r.get("url", "")
        content = r.get("content", "")
        context_blocks.append(f"--- Source {idx}: {title} ({url}) ---\n{content}")

    context_str = "\n\n".join(context_blocks)
    prompt = f"Live Web Search Context:\n{context_str}\n\nUser Question: {query}\n\nClear, factual research answer:"

    messages = [
        SystemMessage(content=RESEARCH_PROMPT),
        HumanMessage(content=prompt)
    ]

    fallback_lines = ["**External Research Summary (Offline Mode):**\n"]
    for r in web_results[:3]:
        fallback_lines.append(f"- **{r.get('title')}:** {r.get('content')[:180]}... [Link]({r.get('url')})")
    fallback_text = "\n".join(fallback_lines)

    answer = invoke_llm_resilient(messages, fallback_text=fallback_text)

    return {
        "external_research_results": web_results,
        "external_answer": answer
    }

# ----------------- Synthesis Agent -----------------

SYNTHESIS_PROMPT = """You are the Senior Synthesis Agent for an enterprise intelligence platform.
Your objective is to produce the final, definitive response by integrating the contributions of specialized agents:

Rules:
1. Deliver a clean, natural, and helpful answer to the user's question immediately.
2. DO NOT simply dump raw document excerpts. Formulate a cohesive, human-readable response.
3. If the query was a calculation or simple question, state the direct result clearly.
4. When both internal documents and external research are provided:
   - For company-specific policies and benefits, prioritize internal documents.
   - For industry trends or benchmarks, use external research.
   - Explicitly highlight any differences or conflicts between company policy and industry practice.
5. Use clean formatting with bold headings and bullet points where helpful.
"""

def synthesis_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesis Agent combines internal RAG and external research results into the final deliverable.
    """
    query = state.get("query", "")
    route = state.get("route", "rag")
    internal_answer = state.get("internal_answer", "")
    external_answer = state.get("external_answer", "")
    internal_chunks = state.get("internal_retrieved_chunks", [])
    external_results = state.get("external_research_results", [])

    logger.info(f"Synthesis Agent synthesizing results for route: '{route}'")

    # If it's a direct calculation or quick answer from research, return cleanly
    if route == "research" and not external_results and external_answer:
        final_answer = external_answer
    else:
        prompt = f"""User Inquiry: {query}
Routing Decision: {route}

=== Internal Agent Findings ===
{internal_answer if internal_answer else "None (RAG agent was not called for this query)."}

=== External Research Findings ===
{external_answer if external_answer else "None (Research agent was not called for this query)."}

Produce a cohesive, well-formatted final response:"""

        messages = [
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(content=prompt)
        ]

        fallback_text = ""
        if route == "rag":
            fallback_text = internal_answer or "No internal findings were generated."
        elif route == "research":
            fallback_text = external_answer or "No external findings were generated."
        else:
            fallback_text = f"### Internal Company Perspective\n{internal_answer}\n\n### External Industry Perspective\n{external_answer}"

        final_answer = invoke_llm_resilient(messages, fallback_text=fallback_text)

    # Compile structured citations
    internal_sources = []
    for c in internal_chunks:
        internal_sources.append({
            "document_name": c.get("document_name", ""),
            "page_or_section": c.get("page_or_section", ""),
            "document_id": c.get("document_id", ""),
            "snippet": c.get("text", "")[:280] + ("..." if len(c.get("text", "")) > 280 else ""),
            "score": c.get("rerank_score", c.get("rrf_score"))
        })

    external_sources = []
    for r in external_results:
        external_sources.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:280] + ("..." if len(r.get("content", "")) > 280 else ""),
            "score": r.get("score")
        })

    return {
        "final_answer": final_answer,
        "sources": {
            "internal": internal_sources,
            "external": external_sources
        }
    }
