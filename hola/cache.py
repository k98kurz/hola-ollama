"""
Simple async LRU cache implementation.
"""

from collections import OrderedDict
from typing import Any
import asyncio


class AsyncLRUCache:
    """Thread-safe async LRU cache with bounded size."""

    def __init__(self, maxsize: int | None = None) -> None:
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get value by key, updating access order. Returns None if not found."""
        async with self._lock:
            if key not in self._cache:
                return None
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

    async def set(self, key: str, value: Any) -> None:
        """Set value for key, evicting LRU if over capacity."""
        async with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            self._cache[key] = value
            if self.maxsize is not None and len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
