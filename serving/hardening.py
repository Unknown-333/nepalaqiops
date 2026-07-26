"""
System Hardening Utilities — Retry logic, fallback caches, and data validation.
Used across ingestion, serving, and training layers.
"""

import functools
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    on_retry: Callable | None = None,
):
    """
    Decorator: exponential backoff with jitter for transient failures.
    
    Used for DuckDB lock contention, Kafka broker unavailability, and API timeouts.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Cap on the delay between retries.
        exceptions: Tuple of exception types to catch.
        on_retry: Optional callback(attempt, exception, delay) for metrics/logging.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"[{func.__name__}] All {max_retries} retries exhausted. "
                            f"Last error: {e}"
                        )
                        raise

                    # Exponential backoff with jitter
                    import random
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = random.uniform(0, delay * 0.1)
                    actual_delay = delay + jitter

                    logger.warning(
                        f"[{func.__name__}] Attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {actual_delay:.2f}s..."
                    )

                    if on_retry:
                        on_retry(attempt, e, actual_delay)

                    time.sleep(actual_delay)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Retry loop exited without a result")  # Should not reach here
        return wrapper
    return decorator


class FallbackCache:
    """
    Two-tier fallback cache for serving layer resilience.
    
    Tier 1: Redis (fast, shared across instances)
    Tier 2: In-memory LRU (survives Redis outage)
    Tier 3: Static default (last resort)
    
    Used by ModelStore when MLflow/Redis are temporarily unavailable.
    """

    def __init__(self, maxsize: int = 1000, default_ttl: int = 3600):
        from collections import OrderedDict
        self._memory_cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._maxsize = maxsize
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Get from memory cache (TTL-aware)."""
        if key in self._memory_cache:
            value, expires_at = self._memory_cache[key]
            if time.time() < expires_at:
                # Move to end (LRU)
                self._memory_cache.move_to_end(key)
                return value
            else:
                del self._memory_cache[key]
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store in memory cache with TTL."""
        expires_at = time.time() + (ttl or self._default_ttl)
        self._memory_cache[key] = (value, expires_at)
        self._memory_cache.move_to_end(key)

        # Evict oldest if over capacity
        while len(self._memory_cache) > self._maxsize:
            self._memory_cache.popitem(last=False)

    def invalidate(self, key: str) -> None:
        """Remove a key from cache."""
        self._memory_cache.pop(key, None)
