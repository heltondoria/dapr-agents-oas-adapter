"""Tests for the caching module."""

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dapr_agents_oas_adapter.cache import (
    CachedLoader,
    CacheEntry,
    CacheStats,
    InMemoryCache,
    _compute_cache_key,
)
from dapr_agents_oas_adapter.types import DaprAgentConfig, WorkflowDefinition


class TestCacheEntry:
    """Tests for CacheEntry class."""

    def test_entry_not_expired_without_ttl(self) -> None:
        """Entry without expiration is never expired."""
        entry = CacheEntry(value="test", created_at=time.time(), expires_at=None)
        assert not entry.is_expired()

    def test_entry_not_expired_within_ttl(self) -> None:
        """Entry within TTL is not expired."""
        entry = CacheEntry(value="test", created_at=time.time(), expires_at=time.time() + 100)
        assert not entry.is_expired()

    def test_entry_expired_after_ttl(self) -> None:
        """Entry past TTL is expired."""
        entry = CacheEntry(value="test", created_at=time.time() - 100, expires_at=time.time() - 1)
        assert entry.is_expired()


class TestInMemoryCache:
    """Tests for InMemoryCache class."""

    def test_get_nonexistent_key(self) -> None:
        """Get returns None for nonexistent key."""
        cache: InMemoryCache[str] = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self) -> None:
        """Set and get a value."""
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_set_with_ttl(self) -> None:
        """Set a value with TTL."""
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "value", ttl=1.0)
        assert cache.get("key") == "value"

    def test_expired_entry_returns_none(self) -> None:
        """Expired entry returns None and is removed."""
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "value", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("key") is None
        assert cache.size() == 0

    def test_default_ttl(self) -> None:
        """Default TTL is applied when not specified."""
        cache: InMemoryCache[str] = InMemoryCache(default_ttl=0.01)
        cache.set("key", "value")
        time.sleep(0.02)
        assert cache.get("key") is None

    def test_delete_existing_key(self) -> None:
        """Delete removes an existing key."""
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None

    def test_delete_nonexistent_key(self) -> None:
        """Delete returns False for nonexistent key."""
        cache: InMemoryCache[str] = InMemoryCache()
        assert cache.delete("nonexistent") is False

    def test_clear(self) -> None:
        """Clear removes all entries."""
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_size(self) -> None:
        """Size returns correct count."""
        cache: InMemoryCache[str] = InMemoryCache()
        assert cache.size() == 0
        cache.set("key1", "value1")
        assert cache.size() == 1
        cache.set("key2", "value2")
        assert cache.size() == 2
        cache.delete("key1")
        assert cache.size() == 1

    def test_max_size_eviction(self) -> None:
        """Max size triggers eviction of oldest entry."""
        cache: InMemoryCache[str] = InMemoryCache(max_size=2)
        cache.set("key1", "value1")
        time.sleep(0.001)  # Ensure different timestamps
        cache.set("key2", "value2")
        time.sleep(0.001)
        cache.set("key3", "value3")  # Should evict key1
        assert cache.size() == 2
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

    def test_max_size_update_existing(self) -> None:
        """Updating existing key doesn't evict."""
        cache: InMemoryCache[str] = InMemoryCache(max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key1", "updated")  # Update, not insert
        assert cache.size() == 2
        assert cache.get("key1") == "updated"

    def test_cleanup_expired(self) -> None:
        """Cleanup removes all expired entries."""
        cache: InMemoryCache[str] = InMemoryCache()
        cache.set("key1", "value1", ttl=0.01)
        cache.set("key2", "value2", ttl=100)
        cache.set("key3", "value3", ttl=0.01)
        time.sleep(0.02)
        removed = cache.cleanup_expired()
        assert removed == 2
        assert cache.size() == 1
        assert cache.get("key2") == "value2"

    def test_thread_safety(self) -> None:
        """Cache is thread-safe."""
        cache: InMemoryCache[int] = InMemoryCache()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(100):
                    cache.set(f"key{i}", i)
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                for i in range(100):
                    cache.get(f"key{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)] + [
            threading.Thread(target=reader) for _ in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestComputeCacheKey:
    """Tests for _compute_cache_key function."""

    def test_string_content(self) -> None:
        """Compute key from string content."""
        key = _compute_cache_key("test content")
        assert len(key) == 16  # SHA256 truncated to 16 chars

    def test_dict_content(self) -> None:
        """Compute key from dict content."""
        key = _compute_cache_key({"key": "value"})
        assert len(key) == 16

    def test_with_prefix(self) -> None:
        """Compute key with prefix."""
        key = _compute_cache_key("content", prefix="json")
        assert key.startswith("json:")

    def test_same_content_same_key(self) -> None:
        """Same content produces same key."""
        key1 = _compute_cache_key("test")
        key2 = _compute_cache_key("test")
        assert key1 == key2

    def test_different_content_different_key(self) -> None:
        """Different content produces different key."""
        key1 = _compute_cache_key("test1")
        key2 = _compute_cache_key("test2")
        assert key1 != key2

    def test_dict_order_independent(self) -> None:
        """Dict key order doesn't affect hash."""
        key1 = _compute_cache_key({"a": 1, "b": 2})
        key2 = _compute_cache_key({"b": 2, "a": 1})
        assert key1 == key2


class TestCacheStats:
    """Tests for CacheStats class."""

    def test_initial_values(self) -> None:
        """Initial values are zero."""
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.total_requests == 0

    def test_record_hit(self) -> None:
        """Record hit increments hits."""
        stats = CacheStats()
        stats.record_hit()
        assert stats.hits == 1
        assert stats.total_requests == 1

    def test_record_miss(self) -> None:
        """Record miss increments misses."""
        stats = CacheStats()
        stats.record_miss()
        assert stats.misses == 1
        assert stats.total_requests == 1

    def test_hit_rate_empty(self) -> None:
        """Hit rate is 0 when no requests."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self) -> None:
        """Hit rate is calculated correctly."""
        stats = CacheStats()
        stats.record_hit()
        stats.record_hit()
        stats.record_miss()
        assert stats.hit_rate == pytest.approx(2 / 3)

    def test_reset(self) -> None:
        """Reset clears all stats."""
        stats = CacheStats()
        stats.record_hit()
        stats.record_miss()
        stats.reset()
        assert stats.hits == 0
        assert stats.misses == 0


class TestCachedLoader:
    """Tests for CachedLoader class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_loader = MagicMock()
        self.mock_loader.tool_registry = {}

    def test_load_json_caches_result(self) -> None:
        """Load JSON caches the result."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_json.return_value = agent_config

        cached_loader = CachedLoader(self.mock_loader)
        json_content = '{"name": "test_agent", "component_type": "Agent"}'

        result1 = cached_loader.load_json(json_content)
        result2 = cached_loader.load_json(json_content)

        assert result1 == agent_config
        assert result2 == agent_config
        assert self.mock_loader.load_json.call_count == 1  # Only called once

    def test_load_json_cache_disabled(self) -> None:
        """Load JSON without cache calls loader each time."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_json.return_value = agent_config

        cached_loader = CachedLoader(self.mock_loader)
        json_content = '{"name": "test_agent"}'

        cached_loader.load_json(json_content, use_cache=False)
        cached_loader.load_json(json_content, use_cache=False)

        assert self.mock_loader.load_json.call_count == 2

    def test_load_yaml_caches_result(self) -> None:
        """Load YAML caches the result."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_yaml.return_value = agent_config

        cached_loader = CachedLoader(self.mock_loader)
        yaml_content = "name: test_agent"

        result1 = cached_loader.load_yaml(yaml_content)
        result2 = cached_loader.load_yaml(yaml_content)

        assert result1 == agent_config
        assert result2 == agent_config
        assert self.mock_loader.load_yaml.call_count == 1

    def test_load_dict_caches_result(self) -> None:
        """Load dict caches the result."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_dict.return_value = agent_config

        cached_loader = CachedLoader(self.mock_loader)
        spec_dict = {"name": "test_agent", "component_type": "Agent"}

        result1 = cached_loader.load_dict(spec_dict)
        result2 = cached_loader.load_dict(spec_dict)

        assert result1 == agent_config
        assert result2 == agent_config
        assert self.mock_loader.load_dict.call_count == 1

    def test_load_json_file_caches_result(self) -> None:
        """Load JSON file caches the result."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_json_file.return_value = agent_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"name": "test_agent"}, f)
            f.flush()
            file_path = f.name

        try:
            cached_loader = CachedLoader(self.mock_loader)

            result1 = cached_loader.load_json_file(file_path)
            result2 = cached_loader.load_json_file(file_path)

            assert result1 == agent_config
            assert result2 == agent_config
            assert self.mock_loader.load_json_file.call_count == 1
        finally:
            Path(file_path).unlink()

    def test_load_yaml_file_caches_result(self) -> None:
        """Load YAML file caches the result."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_yaml_file.return_value = agent_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent")
            f.flush()
            file_path = f.name

        try:
            cached_loader = CachedLoader(self.mock_loader)

            result1 = cached_loader.load_yaml_file(file_path)
            result2 = cached_loader.load_yaml_file(file_path)

            assert result1 == agent_config
            assert result2 == agent_config
            assert self.mock_loader.load_yaml_file.call_count == 1
        finally:
            Path(file_path).unlink()

    def test_load_component_not_cached(self) -> None:
        """Load component is not cached."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_component.return_value = agent_config

        cached_loader = CachedLoader(self.mock_loader)
        component = MagicMock()

        cached_loader.load_component(component)
        cached_loader.load_component(component)

        assert self.mock_loader.load_component.call_count == 2

    def test_create_agent_not_cached(self) -> None:
        """Create agent is not cached."""
        self.mock_loader.create_agent.return_value = MagicMock()

        cached_loader = CachedLoader(self.mock_loader)
        config = DaprAgentConfig(name="test")

        cached_loader.create_agent(config)
        cached_loader.create_agent(config)

        assert self.mock_loader.create_agent.call_count == 2

    def test_create_workflow_not_cached(self) -> None:
        """Create workflow is not cached."""
        self.mock_loader.create_workflow.return_value = MagicMock()

        cached_loader = CachedLoader(self.mock_loader)
        workflow = WorkflowDefinition(
            name="test",
            tasks=[],
            edges=[],
            start_node="start",
            end_nodes=["end"],
        )

        cached_loader.create_workflow(workflow)
        cached_loader.create_workflow(workflow)

        assert self.mock_loader.create_workflow.call_count == 2

    def test_stats_tracking(self) -> None:
        """Stats are tracked correctly."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_json.return_value = agent_config

        cached_loader = CachedLoader(self.mock_loader)

        cached_loader.load_json('{"name": "test1"}')  # Miss
        cached_loader.load_json('{"name": "test1"}')  # Hit
        cached_loader.load_json('{"name": "test2"}')  # Miss

        assert cached_loader.stats.hits == 1
        assert cached_loader.stats.misses == 2
        assert cached_loader.stats.hit_rate == pytest.approx(1 / 3)

    def test_invalidate(self) -> None:
        """Invalidate removes cached entry."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_json.return_value = agent_config

        cached_loader = CachedLoader(self.mock_loader)
        json_content = '{"name": "test"}'

        cached_loader.load_json(json_content)
        assert cached_loader.cache.size() == 1

        result = cached_loader.invalidate(json_content, prefix="json")
        assert result is True
        assert cached_loader.cache.size() == 0

    def test_clear_cache(self) -> None:
        """Clear cache removes all entries and resets stats."""
        agent_config = DaprAgentConfig(name="test_agent")
        self.mock_loader.load_json.return_value = agent_config

        cached_loader = CachedLoader(self.mock_loader)

        cached_loader.load_json('{"name": "test1"}')
        cached_loader.load_json('{"name": "test2"}')

        cached_loader.clear_cache()

        assert cached_loader.cache.size() == 0
        assert cached_loader.stats.total_requests == 0

    def test_tool_registry_property(self) -> None:
        """Tool registry property delegates to loader."""
        cached_loader = CachedLoader(self.mock_loader)
        assert cached_loader.tool_registry == {}

        new_registry = {"tool": lambda: None}
        cached_loader.tool_registry = new_registry
        self.mock_loader.tool_registry = new_registry

    def test_register_tool(self) -> None:
        """Register tool delegates to loader."""
        cached_loader = CachedLoader(self.mock_loader)
        tool_fn = lambda: None  # noqa: E731

        cached_loader.register_tool("my_tool", tool_fn)
        self.mock_loader.register_tool.assert_called_once_with("my_tool", tool_fn)

    def test_custom_cache_backend(self) -> None:
        """Custom cache backend is used."""
        custom_cache: InMemoryCache[DaprAgentConfig | WorkflowDefinition] = InMemoryCache(
            default_ttl=60
        )
        cached_loader = CachedLoader(self.mock_loader, cache=custom_cache)

        assert cached_loader.cache is custom_cache

    def test_loader_property(self) -> None:
        """Loader property returns underlying loader."""
        cached_loader = CachedLoader(self.mock_loader)
        assert cached_loader.loader is self.mock_loader


class TestCachedLoaderIntegration:
    """Integration tests for CachedLoader with real loader."""

    def test_with_real_loader(self) -> None:
        """CachedLoader works with real DaprAgentSpecLoader."""
        from dapr_agents_oas_adapter import CachedLoader, DaprAgentSpecLoader

        loader = DaprAgentSpecLoader()
        cached_loader = CachedLoader(loader)

        # Test that it initializes without error
        assert cached_loader.loader is loader
        assert cached_loader.cache is not None
        assert cached_loader.stats.total_requests == 0
