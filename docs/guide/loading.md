# Loading OAS Specifications

The `DaprAgentSpecLoader` is the primary class for converting Open Agent Spec (OAS) specifications into Dapr Agents configurations.

## Basic Usage

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

loader = DaprAgentSpecLoader()
```

## Loading Methods

### From YAML String

```python
yaml_content = """
component_type: Agent
name: assistant
system_prompt: You are helpful.
"""

config = loader.load_yaml(yaml_content)
```

### From JSON String

```python
json_content = '{"component_type": "Agent", "name": "assistant"}'
config = loader.load_json(json_content)
```

### From Dictionary

```python
spec_dict = {
    "component_type": "Agent",
    "name": "assistant"
}
config = loader.load_dict(spec_dict)
```

### From File

```python
# Load from YAML file
with open("agent.yaml") as f:
    config = loader.load_yaml(f.read())
```

## Tool Registry

Register custom tools that agents can use:

```python
def web_search(query: str) -> list[str]:
    """Search the web for information."""
    return [f"Result for: {query}"]

def calculate(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

loader = DaprAgentSpecLoader(
    tool_registry={
        "web_search": web_search,
        "calculate": calculate
    }
)

# Or register after creation
loader.register_tool("summarize", summarize_function)
```

## Creating Agents

After loading a configuration, create an executable agent:

```python
from dapr_agents_oas_adapter.types import DaprAgentConfig

config = loader.load_yaml(yaml_content)

if isinstance(config, DaprAgentConfig):
    # Pass additional tools
    agent = loader.create_agent(config, additional_tools={
        "custom_tool": my_custom_tool
    })
```

## Creating Workflows

For Flow specifications, create executable workflow functions:

```python
from dapr_agents_oas_adapter.types import WorkflowDefinition

workflow_def = loader.load_yaml(workflow_yaml)

if isinstance(workflow_def, WorkflowDefinition):
    # Optional task implementations
    task_impls = {
        "process_data": my_process_function
    }

    workflow_fn = loader.create_workflow(
        workflow_def,
        task_implementations=task_impls
    )
```

## Generated Code

Get the Python source code for a workflow:

```python
code = loader.generate_workflow_code(workflow_def)
print(code)  # Python source code
```

## Error Handling

```python
from dapr_agents_oas_adapter.converters.base import ConversionError

try:
    config = loader.load_yaml(yaml_content)
except ConversionError as e:
    print(f"Conversion failed: {e}")
    print(f"Suggestion: {e.suggestion}")
    if e.caused_by:
        print(f"Caused by: {e.caused_by}")
```

## Return Types

The loader returns different types based on the OAS component:

| component_type | Return Type |
|----------------|-------------|
| Agent | `DaprAgentConfig` |
| Flow | `WorkflowDefinition` |

Use `isinstance()` checks for type-safe handling:

```python
result = loader.load_dict(spec)

if isinstance(result, DaprAgentConfig):
    agent = loader.create_agent(result)
elif isinstance(result, WorkflowDefinition):
    workflow_fn = loader.create_workflow(result)
```
