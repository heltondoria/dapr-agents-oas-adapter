# Caching

Classes for caching loaded configurations.

## CachedLoader

Loader wrapper that caches results.

::: dapr_agents_oas_adapter.cache.CachedLoader
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - load_yaml
        - load_json
        - load_dict
        - stats

## InMemoryCache

Thread-safe in-memory cache with TTL support.

::: dapr_agents_oas_adapter.cache.InMemoryCache
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - get
        - set
        - delete
        - clear

## CacheBackend

Abstract base class for cache implementations.

::: dapr_agents_oas_adapter.cache.CacheBackend
    options:
      show_root_heading: true
      show_source: true

## CacheStats

Cache statistics tracking.

::: dapr_agents_oas_adapter.cache.CacheStats
    options:
      show_root_heading: true
      show_source: true
      members:
        - hits
        - misses
        - hit_ratio

## Usage Examples

### Basic Caching

```python
from dapr_agents_oas_adapter import (
    CachedLoader,
    DaprAgentSpecLoader,
    InMemoryCache
)

# Create cache
cache = InMemoryCache(
    max_size=100,
    ttl_seconds=300
)

# Create cached loader
loader = CachedLoader(
    loader=DaprAgentSpecLoader(),
    cache=cache
)

# Load (cache miss)
config = loader.load_yaml(yaml_content)

# Load again (cache hit)
config = loader.load_yaml(yaml_content)

# Check stats
print(f"Hits: {loader.stats.hits}")
print(f"Misses: {loader.stats.misses}")
```

### Custom Cache Backend

```python
from dapr_agents_oas_adapter import CacheBackend
from typing import TypeVar

T = TypeVar("T")

class MyCache(CacheBackend[T]):
    def get(self, key: str) -> T | None:
        # Custom get implementation
        pass

    def set(self, key: str, value: T) -> None:
        # Custom set implementation
        pass

    def delete(self, key: str) -> bool:
        # Custom delete implementation
        pass

    def clear(self) -> None:
        # Custom clear implementation
        pass
```

## Related

- [DaprAgentSpecLoader](loader.md) - Base loader
- [Loading Guide](../guide/caching.md) - Usage guide
