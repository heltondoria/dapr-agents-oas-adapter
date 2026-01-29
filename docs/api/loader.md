# DaprAgentSpecLoader

The main class for loading Open Agent Spec (OAS) specifications and converting them to Dapr Agents configurations.

## Class Reference

::: dapr_agents_oas_adapter.loader.DaprAgentSpecLoader
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - load_yaml
        - load_json
        - load_dict
        - create_agent
        - create_workflow
        - generate_workflow_code
        - register_tool
        - tool_registry

## Usage Examples

### Basic Loading

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

loader = DaprAgentSpecLoader()

# Load from YAML
config = loader.load_yaml("""
component_type: Agent
name: assistant
system_prompt: You are helpful.
""")
```

### With Tool Registry

```python
def my_tool(query: str) -> str:
    return f"Result: {query}"

loader = DaprAgentSpecLoader(
    tool_registry={"my_tool": my_tool}
)
```

### Creating Agents

```python
from dapr_agents_oas_adapter.types import DaprAgentConfig

config = loader.load_yaml(yaml_content)
if isinstance(config, DaprAgentConfig):
    agent = loader.create_agent(config)
```

### Creating Workflows

```python
from dapr_agents_oas_adapter.types import WorkflowDefinition

workflow = loader.load_yaml(workflow_yaml)
if isinstance(workflow, WorkflowDefinition):
    workflow_fn = loader.create_workflow(workflow)
```

## Related

- [StrictLoader](validation.md#strictloader) - Loader with validation
- [CachedLoader](caching.md) - Loader with caching
- [AsyncDaprAgentSpecLoader](async_loader.md) - Async version
