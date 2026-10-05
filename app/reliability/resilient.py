import time
import functools
import logging
from typing import Callable, Any, Optional, Dict

from app.reliability.circuit_breaker import (
    get_circuit_breaker,
    CircuitBreakerOpenException,
    CircuitState
)
from app.config import settings

logger = logging.getLogger(__name__)

def resilient_call(
    service_name: str,
    func: Callable,
    *args,
    fallback: Optional[Callable] = None,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    **kwargs
) -> Any:
    """
    Executes a callable with Circuit Breaker protection, Exponential Backoff Retries,
    and automatic Fallback degradation for external dependencies (Milvus, Gemini, Reranker, Tavily).
    """
    cb = get_circuit_breaker(
        service_name,
        failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
    )

    retries = max_retries if max_retries is not None else settings.MAX_RETRIES
    backoff = backoff_factor if backoff_factor is not None else settings.INITIAL_BACKOFF_SECONDS

    # 1. Check Circuit Breaker State
    if not cb.can_execute():
        logger.warning(
            f"ResilientCall: Circuit breaker for '{service_name}' is OPEN. "
            f"Failing fast without remote call."
        )
        if fallback is not None:
            logger.info(f"ResilientCall: Invoking fallback handler for '{service_name}'.")
            return fallback(*args, **kwargs)
        raise CircuitBreakerOpenException(service_name)

    # 2. Execute with Retry and Exponential Backoff
    last_exception: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            result = func(*args, **kwargs)
            cb.record_success()
            return result
        except Exception as exc:
            last_exception = exc
            cb.record_failure(exc)
            logger.warning(
                f"ResilientCall: Attempt {attempt + 1}/{retries + 1} for '{service_name}' failed. "
                f"Error: {exc}"
            )

            # If more attempts remain and circuit didn't just trip to OPEN
            if attempt < retries and cb.state != CircuitState.OPEN:
                sleep_duration = backoff * (2 ** attempt)
                logger.info(f"ResilientCall: Retrying '{service_name}' in {sleep_duration:.2f} seconds...")
                time.sleep(sleep_duration)
            else:
                break

    # 3. All retries exhausted or Circuit tripped
    logger.error(f"ResilientCall: All attempts exhausted for service '{service_name}'.")
    if fallback is not None:
        logger.info(f"ResilientCall: Triggering fallback execution for '{service_name}'.")
        try:
            return fallback(*args, **kwargs)
        except Exception as fb_exc:
            logger.error(f"ResilientCall: Fallback for '{service_name}' also failed: {fb_exc}")
            raise fb_exc

    if last_exception:
        raise last_exception
    raise RuntimeError(f"Service '{service_name}' failed without explicit exception.")


def resilient(
    service_name: str,
    fallback: Optional[Callable] = None,
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None
):
    """
    Decorator for wrapping functions with resilient_call.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return resilient_call(
                service_name=service_name,
                func=func,
                *args,
                fallback=fallback,
                max_retries=max_retries,
                backoff_factor=backoff_factor,
                **kwargs
            )
        return wrapper
    return decorator
