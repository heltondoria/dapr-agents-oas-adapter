# Async Support

The `AsyncDaprAgentSpecLoader` provides asynchronous operations for non-blocking I/O scenarios.

## Basic Usage

```python
from dapr_agents_oas_adapter import AsyncDaprAgentSpecLoader

loader = AsyncDaprAgentSpecLoader()

# Async loading
config = await loader.load_dict(spec_dict)
```

## Context Manager

Use the async context manager for proper resource cleanup:

```python
async with AsyncDaprAgentSpecLoader() as loader:
    config = await loader.load_yaml(yaml_content)
    # Resources automatically cleaned up
```

## Loading Methods

All synchronous methods have async equivalents:

```python
# From dictionary
config = await loader.load_dict(spec_dict)

# From YAML string
config = await loader.load_yaml(yaml_string)

# From JSON string
config = await loader.load_json(json_string)
```

## Concurrent Loading

Load multiple specs concurrently:

```python
import asyncio
from dapr_agents_oas_adapter import AsyncDaprAgentSpecLoader

async def load_all_specs(yaml_contents: list[str]):
    async with AsyncDaprAgentSpecLoader() as loader:
        tasks = [loader.load_yaml(content) for content in yaml_contents]
        return await asyncio.gather(*tasks)

# Load 10 specs concurrently
configs = await load_all_specs(yaml_list)
```

## Synchronous Wrapper

For synchronous code that needs to call async methods:

```python
from dapr_agents_oas_adapter import AsyncDaprAgentSpecLoader, run_sync

loader = AsyncDaprAgentSpecLoader()

# Run async method synchronously
config = run_sync(loader.load_dict(spec_dict))
```

## Tool Registry

The async loader supports the same tool registry:

```python
async def async_search(query: str) -> list[str]:
    # Async search implementation
    await asyncio.sleep(0.1)
    return [f"Result: {query}"]

loader = AsyncDaprAgentSpecLoader(
    tool_registry={"search": async_search}
)
```

## Error Handling

Errors work the same as synchronous loading:

```python
from dapr_agents_oas_adapter.converters.base import ConversionError

try:
    config = await loader.load_yaml(invalid_yaml)
except ConversionError as e:
    print(f"Error: {e}")
```

## Performance Considerations

The async loader uses a thread pool executor for CPU-bound conversion operations:

- I/O operations are truly async
- Conversion logic runs in thread pool
- Best for I/O-bound workloads

For CPU-bound batch processing, consider using `ProcessPoolExecutor` instead:

```python
from concurrent.futures import ProcessPoolExecutor
from dapr_agents_oas_adapter import DaprAgentSpecLoader

def load_sync(yaml_content):
    return DaprAgentSpecLoader().load_yaml(yaml_content)

with ProcessPoolExecutor() as pool:
    configs = list(pool.map(load_sync, yaml_contents))
```
