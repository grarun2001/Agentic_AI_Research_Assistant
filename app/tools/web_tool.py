import logging
import os
from typing import List, Dict, Any, Optional

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

from app.config import settings
from app.reliability.resilient import resilient_call

logger = logging.getLogger(__name__)

def perform_web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Executes live external web search using Tavily.
    Protected with independent circuit breaker and graceful degradation fallback.
    """
    api_key = settings.TAVILY_API_KEY or os.getenv("TAVILY_API_KEY", "")

    def _execute_search() -> List[Dict[str, Any]]:
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not configured in environment.")

        if TavilyClient is None:
            raise ImportError("tavily-python library is not installed.")

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=False
        )

        raw_results = response.get("results", [])
        formatted_results = []
        for r in raw_results:
            formatted_results.append({
                "title": r.get("title", "Untitled Source"),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": float(r.get("score", 0.0))
            })

        logger.info(f"Tavily web search returned {len(formatted_results)} results for query: '{query[:40]}'")
        return formatted_results

    def _search_fallback() -> List[Dict[str, Any]]:
        logger.warning(f"Tavily web search fallback triggered for query '{query[:40]}'.")
        return [
            {
                "title": "External Web Search Degraded / Unavailable",
                "url": "https://tavily.com",
                "content": (
                    "Live external web search is currently unreachable or TAVILY_API_KEY is unconfigured. "
                    "Operating in degraded mode with available local sources."
                ),
                "score": 0.0
            }
        ]

    return resilient_call(
        service_name="tavily",
        func=_execute_search,
        fallback=_search_fallback,
        max_retries=1,
        backoff_factor=1.0
    )
