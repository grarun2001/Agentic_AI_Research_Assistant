import time
import threading
import logging
from contextlib import contextmanager
from typing import Generator
from fastapi import HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)

class ConcurrencyController:
    """
    Request & Concurrency Control Layer.
    Controls the rate of concurrent workflow executions to prevent resource
    exhaustion, GPU memory spikes, or API rate limit triggers.
    """
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self._semaphore = threading.BoundedSemaphore(value=max_concurrent)
        self._active_count = 0
        self._lock = threading.Lock()
        self._total_processed = 0

    @property
    def active_requests(self) -> int:
        with self._lock:
            return self._active_count

    @property
    def active_count(self) -> int:
        with self._lock:
            return self._active_count

    @property
    def total_processed(self) -> int:
        with self._lock:
            return self._total_processed

    @contextmanager
    def acquire(self, timeout: float = 15.0) -> Generator[None, None, None]:
        """
        Acquires concurrency slot. If saturated, queues request up to timeout.
        Raises HTTP 429 if the concurrency queue times out.
        """
        acquired = self._semaphore.acquire(timeout=timeout)
        if not acquired:
            logger.warning(
                f"Concurrency limit ({self.max_concurrent}) exceeded! Request timed out waiting in queue."
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Server is currently handling peak concurrency ({self.max_concurrent} active tasks). "
                    "Please retry your request in a moment."
                )
            )

        with self._lock:
            self._active_count += 1
            current_active = self._active_count

        logger.info(f"Concurrency slot acquired. Active tasks: {current_active}/{self.max_concurrent}")
        try:
            yield
        finally:
            with self._lock:
                self._active_count -= 1
                self._total_processed += 1
            self._semaphore.release()
            logger.debug(f"Concurrency slot released. Active tasks remaining: {self.active_requests}")

# Singleton concurrency controller
concurrency_controller = ConcurrencyController(max_concurrent=settings.MAX_CONCURRENT_REQUESTS)
