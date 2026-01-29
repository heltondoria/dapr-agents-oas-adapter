"""Caching support for dapr-agents-oas-adapter.

Provides optional caching to avoid re-processing repeated conversions.
"""

from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dapr_agents_oas_adapter.logging import get_logger
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    ToolRegistry,
    WorkflowDefinition,
)


@dataclass
class CacheEntry[T]:
    """A cached value with metadata."""

    value: T
    created_at: float
    expires_at: float | None

    def is_expired(self) -> bool:
        """Check if this entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class CacheBackend[T](ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    def get(self, key: str) -> T | None:
        """Get a value from the cache.

        Args:
            key: The cache key

        Returns:
            The cached value or None if not found/expired
        """
        ...

    @abstractmethod
    def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from the cache.

        Args:
            key: The cache key

        Returns:
            True if the key was deleted, False if not found
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        ...

    @abstractmethod
    def size(self) -> int:
        """Get the number of entries in the cache."""
        ...


class InMemoryCache[T](CacheBackend[T]):
    """Thread-safe in-memory cache with TTL support."""

    def __init__(self, default_ttl: float | None = None, max_size: int | None = None) -> None:
        """Initialize the cache.

        Args:
            default_ttl: Default time-to-live in seconds (None = no expiration)
            max_size: Maximum number of entries (None = unlimited)
        """
        self._cache: dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._logger = get_logger("InMemoryCache")

    def get(self, key: str) -> T | None:
        """Get a value from the cache."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._logger.debug("cache_miss", key=key)
                return None

            if entry.is_expired():
                self._logger.debug("cache_expired", key=key)
                del self._cache[key]
                return None

            self._logger.debug("cache_hit", key=key)
            return entry.value

    def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """Set a value in the cache."""
        with self._lock:
            # Use provided TTL or default
            actual_ttl = ttl if ttl is not None else self._default_ttl

            # Calculate expiration time
            now = time.time()
            expires_at = now + actual_ttl if actual_ttl is not None else None

            # Enforce max size by removing the oldest entries
            if (
                self._max_size is not None
                and len(self._cache) >= self._max_size
                and key not in self._cache
            ):
                self._evict_oldest()

            self._cache[key] = CacheEntry(value=value, created_at=now, expires_at=expires_at)
            self._logger.debug("cache_set", key=key, ttl=actual_ttl)

    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._logger.debug("cache_deleted", key=key)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._logger.debug("cache_cleared", entries_removed=count)

    def size(self) -> int:
        """Get the number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def _evict_oldest(self) -> None:
        """Evict the oldest entry from the cache."""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
        self._logger.debug("cache_evicted", key=oldest_key)

    def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
            for key in expired_keys:
                del self._cache[key]
            if expired_keys:
                self._logger.debug("cache_cleanup", entries_removed=len(expired_keys))
            return len(expired_keys)


def _compute_cache_key(content: str | dict[str, Any], prefix: str = "") -> str:
    """Compute a cache key from content.

    Args:
        content: String or dict content to hash
        prefix: Optional prefix for the key

    Returns:
        A unique cache key
    """
    if isinstance(content, dict):
        # Sort keys for consistent hashing
        import json

        content_str = json.dumps(content, sort_keys=True, default=str)
    else:
        content_str = content

    content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]
    return f"{prefix}:{content_hash}" if prefix else content_hash


class CachedLoader:
    """A caching wrapper around DaprAgentSpecLoader.

    Provides transparent caching for loaded specifications to avoid
    reparsing the same content multiple times.

    Example:
        ```python
        from dapr_agents_oas_adapter import DaprAgentSpecLoader
        from dapr_agents_oas_adapter.cache import CachedLoader, InMemoryCache

        # Create a cached loader with 5-minute TTL
        cache = InMemoryCache(default_ttl=300)
        loader = CachedLoader(DaprAgentSpecLoader(), cache)

        # First call parses the JSON
        config1 = loader.load_json(json_string)

        # Second call returns a cached result
        config2 = loader.load_json(json_string)
        ```
    """

    def __init__(
        self,
        loader: Any,  # DaprAgentSpecLoader, but avoid circular import
        cache: CacheBackend[DaprAgentConfig | WorkflowDefinition] | None = None,
        *,
        default_ttl: float | None = 300.0,  # 5 minutes default
    ) -> None:
        """Initialize the cached loader.

        Args:
            loader: The underlying DaprAgentSpecLoader instance
            cache: Cache backend to use (creates InMemoryCache if None)
            default_ttl: Default TTL for cached entries in seconds
        """
        self._loader = loader
        self._cache: CacheBackend[DaprAgentConfig | WorkflowDefinition] = (
            cache if cache is not None else InMemoryCache(default_ttl=default_ttl)
        )
        self._default_ttl = default_ttl
        self._logger = get_logger("CachedLoader")
        self._stats = CacheStats()

    @property
    def loader(self) -> Any:
        """Get the underlying loader."""
        return self._loader

    @property
    def cache(self) -> CacheBackend[DaprAgentConfig | WorkflowDefinition]:
        """Get the cache backend."""
        return self._cache

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def tool_registry(self) -> ToolRegistry:
        """Get the tool registry from the underlying loader."""
        return self._loader.tool_registry

    @tool_registry.setter
    def tool_registry(self, registry: ToolRegistry) -> None:
        """Set the tool registry on the underlying loader."""
        self._loader.tool_registry = registry

    def register_tool(self, name: str, implementation: Callable[..., Any]) -> None:
        """Register a tool implementation."""
        self._loader.register_tool(name, implementation)

    def load_json(
        self, json_content: str, *, use_cache: bool = True
    ) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a JSON string with caching.

        Args:
            json_content: JSON string containing the OAS specification
            use_cache: Whether to use caching (default: True)

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows
        """
        if not use_cache:
            return self._loader.load_json(json_content)

        cache_key = _compute_cache_key(json_content, prefix="json")
        return self._get_or_load(cache_key, lambda: self._loader.load_json(json_content))

    def load_yaml(
        self, yaml_content: str, *, use_cache: bool = True
    ) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from an YAML string with caching.

        Args:
            yaml_content: YAML string containing the OAS specification
            use_cache: Whether to use caching (default: True)

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows
        """
        if not use_cache:
            return self._loader.load_yaml(yaml_content)

        cache_key = _compute_cache_key(yaml_content, prefix="yaml")
        return self._get_or_load(cache_key, lambda: self._loader.load_yaml(yaml_content))

    def load_json_file(
        self, file_path: str | Path, *, use_cache: bool = True
    ) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a JSON file with caching.

        Args:
            file_path: Path to the JSON file
            use_cache: Whether to use caching (default: True)

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows
        """
        if not use_cache:
            return self._loader.load_json_file(file_path)

        # Use file path and mtime for cache key
        path = Path(file_path)
        mtime = path.stat().st_mtime if path.exists() else 0
        cache_key = _compute_cache_key(f"{path}:{mtime}", prefix="json_file")
        return self._get_or_load(cache_key, lambda: self._loader.load_json_file(file_path))

    def load_yaml_file(
        self, file_path: str | Path, *, use_cache: bool = True
    ) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a YAML file with caching.

        Args:
            file_path: Path to the YAML file
            use_cache: Whether to use caching (default: True)

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows
        """
        if not use_cache:
            return self._loader.load_yaml_file(file_path)

        # Use file path and mtime for cache key
        path = Path(file_path)
        mtime = path.stat().st_mtime if path.exists() else 0
        cache_key = _compute_cache_key(f"{path}:{mtime}", prefix="yaml_file")
        return self._get_or_load(cache_key, lambda: self._loader.load_yaml_file(file_path))

    def load_dict(
        self, spec_dict: dict[str, Any], *, use_cache: bool = True
    ) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a dictionary with caching.

        Args:
            spec_dict: Dictionary containing the OAS specification
            use_cache: Whether to use caching (default: True)

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows
        """
        if not use_cache:
            return self._loader.load_dict(spec_dict)

        cache_key = _compute_cache_key(spec_dict, prefix="dict")
        return self._get_or_load(cache_key, lambda: self._loader.load_dict(spec_dict))

    def load_component(self, component: Any) -> DaprAgentConfig | WorkflowDefinition:
        """Load a PyAgentSpec Component (not cached - components are unique objects)."""
        return self._loader.load_component(component)

    def create_agent(
        self,
        config: DaprAgentConfig,
        additional_tools: dict[str, Callable[..., Any]] | None = None,
    ) -> Any:
        """Create an executable Dapr Agent (not cached - creates new instance)."""
        return self._loader.create_agent(config, additional_tools)

    def create_workflow(
        self,
        workflow_def: WorkflowDefinition,
        task_implementations: dict[str, Callable[..., Any]] | None = None,
    ) -> Any:
        """Create an executable Dapr workflow (not cached - creates new instance)."""
        return self._loader.create_workflow(workflow_def, task_implementations)

    def invalidate(self, content: str | dict[str, Any], prefix: str = "") -> bool:
        """Invalidate a specific cache entry.

        Args:
            content: The original content used for caching
            prefix: The prefix used when caching (json, YAML, dict, etc.)

        Returns:
            True if the entry was invalidated, False if not found
        """
        cache_key = _compute_cache_key(content, prefix=prefix)
        return self._cache.delete(cache_key)

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._stats.reset()

    def _get_or_load(
        self,
        cache_key: str,
        loader_fn: Callable[[], DaprAgentConfig | WorkflowDefinition],
    ) -> DaprAgentConfig | WorkflowDefinition:
        """Get from cache or load and cache the result."""
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._stats.record_hit()
            self._logger.debug("cache_hit", key=cache_key)
            return cached

        self._stats.record_miss()
        result = loader_fn()
        self._cache.set(cache_key, result, self._default_ttl)
        self._logger.debug("cache_miss_loaded", key=cache_key)
        return result


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    hits: int = 0
    misses: int = 0

    @property
    def total_requests(self) -> int:
        """Total number of cache requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
