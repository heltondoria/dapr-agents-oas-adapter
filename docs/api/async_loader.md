# AsyncDaprAgentSpecLoader

Asynchronous version of the loader for non-blocking operations.

## Class Reference

::: dapr_agents_oas_adapter.async_loader.AsyncDaprAgentSpecLoader
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - __aenter__
        - __aexit__
        - load_yaml
        - load_json
        - load_dict

## run_sync

Utility function to run async code synchronously.

::: dapr_agents_oas_adapter.async_loader.run_sync
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Basic Async Usage

```python
from dapr_agents_oas_adapter import AsyncDaprAgentSpecLoader

loader = AsyncDaprAgentSpecLoader()

# Async load
config = await loader.load_yaml(yaml_content)
```

### Context Manager

```python
async with AsyncDaprAgentSpecLoader() as loader:
    config = await loader.load_dict(spec)
    # Resources cleaned up automatically
```

### Concurrent Loading

```python
import asyncio

async def load_all(yaml_contents: list[str]):
    async with AsyncDaprAgentSpecLoader() as loader:
        tasks = [loader.load_yaml(c) for c in yaml_contents]
        return await asyncio.gather(*tasks)

configs = await load_all(yaml_list)
```

### Synchronous Wrapper

```python
from dapr_agents_oas_adapter import AsyncDaprAgentSpecLoader, run_sync

loader = AsyncDaprAgentSpecLoader()

# Run async code in sync context
config = run_sync(loader.load_dict(spec))
```

### With Tool Registry

```python
async def async_tool(query: str) -> str:
    await asyncio.sleep(0.1)
    return f"Result: {query}"

loader = AsyncDaprAgentSpecLoader(
    tool_registry={"my_tool": async_tool}
)
```

## Related

- [DaprAgentSpecLoader](loader.md) - Synchronous loader
- [Async Guide](../guide/async.md) - Usage guide
