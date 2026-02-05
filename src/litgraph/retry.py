"""Unified retry decorator and sliding-window rate limiter."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)


def with_retry(func):
    """Decorator: exponential backoff retry. Handles both sync and async functions.

    Reads max_retries and backoff_base from Settings at call time.
    """
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            from litgraph.settings import get_settings
            settings = get_settings()
            max_retries = settings.retry.max_retries
            backoff_base = settings.retry.backoff_base

            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        wait = backoff_base ** attempt
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs: %s",
                            attempt + 1, max_retries, func.__name__, wait, exc,
                        )
                        await asyncio.sleep(wait)
            raise last_exc
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            from litgraph.settings import get_settings
            settings = get_settings()
            max_retries = settings.retry.max_retries
            backoff_base = settings.retry.backoff_base

            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        wait = backoff_base ** attempt
                        logger.warning(
                            "Retry %d/%d for %s after %.1fs: %s",
                            attempt + 1, max_retries, func.__name__, wait, exc,
                        )
                        time.sleep(wait)
            raise last_exc
        return sync_wrapper


class RateLimiter:
    """Sliding-window rate limiter.

    Args:
        max_calls: Maximum number of calls per window.
        period: Window size in seconds.
    """

    def __init__(self, max_calls: int | None = None, period: int | None = None):
        from litgraph.settings import get_settings
        settings = get_settings()
        self.max_calls = max_calls or settings.rate_limit.search_max_calls
        self.period = period or settings.rate_limit.search_period
        self._timestamps: deque[float] = deque()

    def _prune(self, now: float) -> None:
        """Remove timestamps outside the current window."""
        cutoff = now - self.period
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def acquire_sync(self) -> None:
        """Block (sync) until a call slot is available."""
        while True:
            now = time.monotonic()
            self._prune(now)
            if len(self._timestamps) < self.max_calls:
                self._timestamps.append(now)
                return
            # Wait until the oldest entry expires
            wait = self._timestamps[0] + self.period - now
            if wait > 0:
                logger.debug("Rate limiter: waiting %.1fs", wait)
                time.sleep(wait)

    async def acquire(self) -> None:
        """Block (async) until a call slot is available."""
        while True:
            now = time.monotonic()
            self._prune(now)
            if len(self._timestamps) < self.max_calls:
                self._timestamps.append(now)
                return
            wait = self._timestamps[0] + self.period - now
            if wait > 0:
                logger.debug("Rate limiter: waiting %.1fs", wait)
                await asyncio.sleep(wait)
