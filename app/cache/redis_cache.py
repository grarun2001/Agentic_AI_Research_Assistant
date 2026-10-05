import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, List

try:
    import redis
except ImportError:
    redis = None

from app.config import settings

logger = logging.getLogger(__name__)

class CacheService:
    """
    High-performance caching layer with Redis backend and in-memory TTL fallback.
    Ensures zero-downtime scalability: If Redis is available, stores in Redis;
    otherwise gracefully uses thread-safe in-memory cache.
    """
    def __init__(self):
        self._redis_client: Optional[Any] = None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._redis_available: Optional[bool] = None
        self._last_redis_check: float = 0.0

    def _get_redis(self) -> Optional[Any]:
        """Lazy-initializes Redis client with connection test and quick timeout."""
        if redis is None:
            return None

        now = time.time()
        # Avoid hammering Redis connection if recently determined unavailable (retry every 30s)
        if self._redis_available is False and (now - self._last_redis_check) < 30:
            return None

        if self._redis_client is None or self._redis_available is False:
            try:
                self._last_redis_check = now
                client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    password=settings.REDIS_PASSWORD or None,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                    decode_responses=True
                )
                client.ping()
                self._redis_client = client
                self._redis_available = True
                logger.info(f"Connected to Redis cache at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            except Exception as e:
                self._redis_available = False
                self._redis_client = None
                logger.debug(f"Redis unavailable ({e}). Using in-memory TTL cache fallback.")

        return self._redis_client

    def generate_cache_key(self, query: str, selected_document_ids: Optional[List[str]] = None) -> str:
        """Computes a deterministic hash cache key for the query and document scope."""
        normalized_q = query.strip().lower()
        doc_str = ""
        if selected_document_ids:
            doc_str = ",".join(sorted(str(d) for d in selected_document_ids if d))
        raw_key = f"rag_q:{normalized_q}|docs:{doc_str}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        return f"cache:research:{digest}"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached response from Redis or in-memory fallback."""
        r = self._get_redis()
        if r is not None:
            try:
                val = r.get(key)
                if val:
                    logger.info(f"⚡ Cache HIT [Redis] for key: {key[:20]}...")
                    return json.loads(val)
            except Exception as e:
                logger.warning(f"Redis get error: {e}. Checking memory cache.")

        # In-memory fallback check
        entry = self._memory_cache.get(key)
        if entry:
            if time.time() < entry.get("expires_at", 0):
                logger.info(f"⚡ Cache HIT [In-Memory] for key: {key[:20]}...")
                return entry.get("data")
            else:
                del self._memory_cache[key]

        return None

    def set(self, key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Saves response to Redis and in-memory cache with TTL."""
        ttl_seconds = ttl or settings.CACHE_TTL_SECONDS
        json_str = json.dumps(data)

        # 1. Save to Redis
        r = self._get_redis()
        if r is not None:
            try:
                r.setex(name=key, time=ttl_seconds, value=json_str)
                logger.info(f"Saved response to Redis cache (TTL: {ttl_seconds}s)")
            except Exception as e:
                logger.warning(f"Redis set error: {e}. Storing in memory cache.")

        # 2. Save to In-Memory Cache
        self._memory_cache[key] = {
            "data": data,
            "expires_at": time.time() + ttl_seconds
        }

        # Keep memory cache size bounded (max 500 entries)
        if len(self._memory_cache) > 500:
            oldest_key = min(self._memory_cache.keys(), key=lambda k: self._memory_cache[k]["expires_at"])
            del self._memory_cache[oldest_key]

    def is_connected(self) -> bool:
        """Returns True if Redis is active."""
        return self._redis_available is True

# Singleton cache instance
cache_service = CacheService()
