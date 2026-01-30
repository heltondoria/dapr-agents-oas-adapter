# Exporting to OAS Format

The `DaprAgentSpecExporter` converts Dapr Agents configurations back to Open Agent Spec format, enabling portability and interoperability.

## Basic Usage

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter

exporter = DaprAgentSpecExporter()
```

## Export Methods

### To Dictionary

```python
from dapr_agents_oas_adapter.types import DaprAgentConfig

config = DaprAgentConfig(
    name="my_agent",
    role="Assistant",
    goal="Help users with tasks",
    tools=["search", "calculate"]
)

oas_dict = exporter.to_dict(config)
# Returns: {"component_type": "Agent", "name": "my_agent", ...}
```

### To YAML String

```python
oas_yaml = exporter.to_yaml(config)
print(oas_yaml)
# component_type: Agent
# name: my_agent
# ...
```

### To JSON String

```python
oas_json = exporter.to_json(config)
# {"component_type": "Agent", "name": "my_agent", ...}
```

## Exporting Agents

```python
from dapr_agents_oas_adapter.types import DaprAgentConfig

config = DaprAgentConfig(
    name="research_assistant",
    role="Researcher",
    goal="Find and summarize information",
    instructions=[
        "Search for relevant sources",
        "Summarize findings clearly"
    ],
    tools=["web_search", "summarize"],
    system_prompt="You are a helpful research assistant."
)

oas = exporter.to_dict(config)
```

## Exporting Workflows

```python
from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowTaskDefinition,
    WorkflowEdgeDefinition
)

workflow = WorkflowDefinition(
    name="content_pipeline",
    description="Generate and review content",
    tasks=[
        WorkflowTaskDefinition(
            name="start",
            task_type="start"
        ),
        WorkflowTaskDefinition(
            name="generate",
            task_type="llm",
            config={"prompt_template": "Write about: {{ topic }}"}
        ),
        WorkflowTaskDefinition(
            name="end",
            task_type="end"
        )
    ],
    edges=[
        WorkflowEdgeDefinition(from_node="start", to_node="generate"),
        WorkflowEdgeDefinition(from_node="generate", to_node="end")
    ],
    start_node="start",
    end_nodes=["end"]
)

oas = exporter.to_dict(workflow)
```

## Roundtrip Conversion

Load an OAS spec, modify it, and export back:

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader, DaprAgentSpecExporter

loader = DaprAgentSpecLoader()
exporter = DaprAgentSpecExporter()

# Load
config = loader.load_yaml(original_yaml)

# Modify (if needed)
config.tools.append("new_tool")

# Export back
updated_yaml = exporter.to_yaml(config)
```

## OAS Version

Exported specs include the agentspec version:

```python
oas = exporter.to_dict(config)
print(oas["agentspec_version"])  # e.g., "1.0"
```

## Writing to File

```python
# YAML file
with open("agent.yaml", "w") as f:
    f.write(exporter.to_yaml(config))

# JSON file
import json
with open("agent.json", "w") as f:
    f.write(exporter.to_json(config))
```
