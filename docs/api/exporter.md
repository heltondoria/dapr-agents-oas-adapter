# DaprAgentSpecExporter

Class for exporting Dapr Agents configurations to Open Agent Spec (OAS) format.

## Class Reference

::: dapr_agents_oas_adapter.exporter.DaprAgentSpecExporter
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - to_dict
        - to_yaml
        - to_json

## Usage Examples

### Export to Dictionary

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import DaprAgentConfig

exporter = DaprAgentSpecExporter()

config = DaprAgentConfig(
    name="my_agent",
    role="Assistant",
    goal="Help users"
)

oas_dict = exporter.to_dict(config)
# {"component_type": "Agent", "name": "my_agent", ...}
```

### Export to YAML

```python
yaml_str = exporter.to_yaml(config)
print(yaml_str)
```

### Export to JSON

```python
json_str = exporter.to_json(config)
print(json_str)
```

### Export Workflow

```python
from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowTaskDefinition,
    WorkflowEdgeDefinition
)

workflow = WorkflowDefinition(
    name="my_workflow",
    tasks=[
        WorkflowTaskDefinition(name="start", task_type="start"),
        WorkflowTaskDefinition(name="end", task_type="end")
    ],
    edges=[
        WorkflowEdgeDefinition(from_node="start", to_node="end")
    ],
    start_node="start",
    end_nodes=["end"]
)

oas = exporter.to_dict(workflow)
```

## Related

- [DaprAgentSpecLoader](loader.md) - Load OAS specs
- [Types](types.md) - Data models
