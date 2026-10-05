import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional, Tuple

from app.cache.redis_cache import cache_service
from app.concurrency.limiter import concurrency_controller
from app.config import settings

logger = logging.getLogger(__name__)

# Resilient Worker ThreadPool for asynchronous queue processing & batch I/O
_WORKER_POOL = ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_REQUESTS, thread_name_prefix="ResearchWorker")

def _try_celery_dispatch(query: str, selected_document_ids: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    """
    Attempts to dispatch the research task to a Celery distributed worker if reachable.
    If Redis broker or Celery worker is unavailable, returns None to fall back to the local worker queue.
    """
    # Fast path: Only attempt Celery if Redis broker is actively connected
    if not cache_service.is_connected():
        return None

    try:
        from app.tasks.tasks import execute_research_workflow_task
        from app.tasks.celery_app import celery_app
        insp = celery_app.control.inspect(timeout=0.3)
        active_workers = insp.ping() if insp else None

        if active_workers:
            logger.info("📡 Dispatching research task to Celery Distributed Worker Queue...")
            async_result = execute_research_workflow_task.delay(query, selected_document_ids)
            # Wait for Celery worker result with timeout
            return async_result.get(timeout=120)
    except Exception as e:
        logger.debug(f"Celery distributed queue bypass ({e}). Using local asynchronous worker queue.")

    return None

def execute_scalable_research_pipeline(
    query: str,
    selected_document_ids: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    Implements the Scalable Architecture Flow:
    
    Incoming Request
           ↓
    Request / Concurrency Control
           ↓
       Cache Check
           ↓
     ┌─────┴─────┐
    YES         NO
     ↓           ↓
    Return     Queue
    Cache        ↓
              Background Worker
                   ↓
            Batch Processing
                   ↓
              Async I/O
                   ↓
            Processed Result
                   ↓
              Save to Cache
              
    Returns (workflow_output, is_cached)
    """
    # 1. Request / Concurrency Control
    with concurrency_controller.acquire(timeout=15.0):
        # 2. Cache Check (Redis / In-memory fallback)
        cache_key = cache_service.generate_cache_key(query, selected_document_ids)
        cached_result = cache_service.get(cache_key)

        # Cache HIT Branch
        if cached_result is not None:
            logger.info(f"⚡ Returning cached result for query: '{query[:40]}'")
            return cached_result, True

        # Cache MISS Branch -> Queue & Background Worker
        logger.info(f"Cache miss for query: '{query[:40]}'. Queuing to Background Worker...")

        # 3. Queue -> Celery Worker or Resilient Async Worker Pool
        celery_result = _try_celery_dispatch(query, selected_document_ids)
        if celery_result is not None:
            processed_result = celery_result
        else:
            # Dispatch to Background Worker Pool (Async I/O & multi-agent execution)
            future = _WORKER_POOL.submit(
                _execute_in_worker,
                query,
                selected_document_ids
            )
            processed_result = future.result(timeout=120)

        # 4. Save Processed Result to Cache
        cache_service.set(cache_key, processed_result, ttl=settings.CACHE_TTL_SECONDS)
        logger.info(f"Successfully processed and cached result for key: {cache_key[:25]}...")

        return processed_result, False

def _execute_in_worker(query: str, selected_document_ids: Optional[List[str]]) -> Dict[str, Any]:
    """Worker job executing the multi-agent graph with Async I/O and hybrid retrieval."""
    from app.graph.workflow import run_research_workflow
    return run_research_workflow(
        query=query,
        selected_document_ids=selected_document_ids
    )
