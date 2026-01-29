# Caching

The `CachedLoader` provides optional caching to improve performance when loading the same specifications multiple times.

## Basic Usage

```python
from dapr_agents_oas_adapter import CachedLoader, DaprAgentSpecLoader, InMemoryCache

# Create cache with TTL and size limits
cache = InMemoryCache(
    max_size=100,        # Maximum entries
    ttl_seconds=300      # 5 minute TTL
)

# Wrap a loader with caching
loader = CachedLoader(
    loader=DaprAgentSpecLoader(),
    cache=cache
)

# First load - cache miss
config1 = loader.load_yaml(yaml_content)

# Second load - cache hit
config2 = loader.load_yaml(yaml_content)
```

## Cache Statistics

Track cache performance:

```python
stats = loader.stats

print(f"Hits: {stats.hits}")
print(f"Misses: {stats.misses}")
print(f"Hit ratio: {stats.hit_ratio:.2%}")
```

## Cache Configuration

### InMemoryCache Options

```python
cache = InMemoryCache(
    max_size=100,        # Max entries (default: 100)
    ttl_seconds=300,     # Time-to-live in seconds (default: 300)
    cleanup_interval=60  # Cleanup expired entries interval
)
```

### Custom Cache Backend

Implement `CacheBackend` for custom storage:

```python
from dapr_agents_oas_adapter import CacheBackend
from typing import TypeVar

T = TypeVar("T")

class RedisCache(CacheBackend[T]):
    def get(self, key: str) -> T | None:
        # Implement Redis get
        pass

    def set(self, key: str, value: T) -> None:
        # Implement Redis set
        pass

    def delete(self, key: str) -> bool:
        # Implement Redis delete
        pass

    def clear(self) -> None:
        # Implement Redis clear
        pass
```

## Cache Invalidation

```python
# Clear all cached entries
cache.clear()

# Delete specific entry (by content hash)
cache.delete(content_hash)
```

## Thread Safety

`InMemoryCache` is thread-safe and can be shared across threads:

```python
from concurrent.futures import ThreadPoolExecutor

cache = InMemoryCache()
loader = CachedLoader(loader=DaprAgentSpecLoader(), cache=cache)

def load_spec(yaml_content):
    return loader.load_yaml(yaml_content)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(load_spec, yaml_contents))
```

## When to Use Caching

**Good use cases:**

- Loading the same spec multiple times
- API endpoints that serve OAS conversions
- Development/testing with repeated operations

**Consider alternatives when:**

- Specs change frequently
- Memory is constrained
- Each request is unique
