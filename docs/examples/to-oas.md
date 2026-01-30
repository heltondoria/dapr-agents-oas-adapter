# To OAS Examples

Examples of exporting Dapr agents/workflows to Open Agent Spec format.

## Export Agent Configuration

Create an agent programmatically and export to OAS:

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import DaprAgentConfig

# Create configuration
config = DaprAgentConfig(
    name="customer_support",
    role="Support Agent",
    goal="Help customers resolve issues quickly",
    instructions=[
        "Greet the customer professionally",
        "Listen carefully to their issue",
        "Provide clear step-by-step solutions",
        "Escalate complex issues appropriately"
    ],
    tools=["lookup_order", "check_inventory", "create_ticket"],
    system_prompt="You are a helpful customer support agent."
)

# Export
exporter = DaprAgentSpecExporter()

# To YAML
yaml_output = exporter.to_yaml(config)
print(yaml_output)

# To JSON
json_output = exporter.to_json(config)

# To dictionary
dict_output = exporter.to_dict(config)
```

Output YAML:

```yaml
agentspec_version: "1.0"
component_type: Agent
name: customer_support
description: null
system_prompt: You are a helpful customer support agent.
tools:
  - lookup_order
  - check_inventory
  - create_ticket
```

## Export Workflow Definition

Create and export a workflow:

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowTaskDefinition,
    WorkflowEdgeDefinition
)

# Create workflow
workflow = WorkflowDefinition(
    name="document_processor",
    description="Process and analyze documents",
    tasks=[
        WorkflowTaskDefinition(
            name="start",
            task_type="start"
        ),
        WorkflowTaskDefinition(
            name="extract_text",
            task_type="tool",
            config={"tool_name": "ocr_extract"}
        ),
        WorkflowTaskDefinition(
            name="analyze",
            task_type="llm",
            config={
                "prompt_template": "Analyze this document:\n{{ text }}"
            }
        ),
        WorkflowTaskDefinition(
            name="summarize",
            task_type="llm",
            config={
                "prompt_template": "Summarize the analysis:\n{{ analysis }}"
            }
        ),
        WorkflowTaskDefinition(
            name="end",
            task_type="end"
        )
    ],
    edges=[
        WorkflowEdgeDefinition(from_node="start", to_node="extract_text"),
        WorkflowEdgeDefinition(from_node="extract_text", to_node="analyze"),
        WorkflowEdgeDefinition(from_node="analyze", to_node="summarize"),
        WorkflowEdgeDefinition(from_node="summarize", to_node="end")
    ],
    start_node="start",
    end_nodes=["end"]
)

# Export
exporter = DaprAgentSpecExporter()
yaml_output = exporter.to_yaml(workflow)
print(yaml_output)
```

## Export Branching Workflow

Workflow with conditional branches:

```python
workflow = WorkflowDefinition(
    name="approval_workflow",
    description="Route requests based on amount",
    tasks=[
        WorkflowTaskDefinition(name="start", task_type="start"),
        WorkflowTaskDefinition(
            name="classify",
            task_type="llm",
            config={
                "prompt_template": "Classify amount {{ amount }}: high or low?",
                "branch_output_key": "classification"
            }
        ),
        WorkflowTaskDefinition(name="manager_approval", task_type="agent"),
        WorkflowTaskDefinition(name="auto_approve", task_type="tool"),
        WorkflowTaskDefinition(name="end", task_type="end")
    ],
    edges=[
        WorkflowEdgeDefinition(from_node="start", to_node="classify"),
        WorkflowEdgeDefinition(
            from_node="classify",
            to_node="manager_approval",
            from_branch="high"
        ),
        WorkflowEdgeDefinition(
            from_node="classify",
            to_node="auto_approve",
            from_branch="low"
        ),
        WorkflowEdgeDefinition(from_node="manager_approval", to_node="end"),
        WorkflowEdgeDefinition(from_node="auto_approve", to_node="end")
    ],
    start_node="start",
    end_nodes=["end"]
)

exporter = DaprAgentSpecExporter()
print(exporter.to_yaml(workflow))
```

## Roundtrip Conversion

Load, modify, and export back:

```python
from dapr_agents_oas_adapter import (
    DaprAgentSpecLoader,
    DaprAgentSpecExporter
)
from dapr_agents_oas_adapter.types import DaprAgentConfig

# Load original
loader = DaprAgentSpecLoader()
config = loader.load_yaml(original_yaml)

# Modify
if isinstance(config, DaprAgentConfig):
    config.tools.append("new_capability")
    config.instructions.append("Use new_capability when needed")

# Export updated version
exporter = DaprAgentSpecExporter()
updated_yaml = exporter.to_yaml(config)

# Verify roundtrip
reloaded = loader.load_yaml(updated_yaml)
assert reloaded.name == config.name
```

## Save to Files

Export to files:

```python
from pathlib import Path
from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import DaprAgentConfig

config = DaprAgentConfig(
    name="file_agent",
    role="Helper",
    goal="Assist with files"
)

exporter = DaprAgentSpecExporter()

# Save as YAML
Path("agent.yaml").write_text(exporter.to_yaml(config))

# Save as JSON
Path("agent.json").write_text(exporter.to_json(config))

# Save as Python dict (for programmatic use)
import json
Path("agent_dict.json").write_text(
    json.dumps(exporter.to_dict(config), indent=2)
)
```

## Bulk Export

Export multiple configurations:

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import DaprAgentConfig

agents = [
    DaprAgentConfig(name="agent1", role="Role1", goal="Goal1"),
    DaprAgentConfig(name="agent2", role="Role2", goal="Goal2"),
    DaprAgentConfig(name="agent3", role="Role3", goal="Goal3"),
]

exporter = DaprAgentSpecExporter()

for agent in agents:
    filename = f"{agent.name}.yaml"
    with open(filename, "w") as f:
        f.write(exporter.to_yaml(agent))
    print(f"Exported: {filename}")
```
