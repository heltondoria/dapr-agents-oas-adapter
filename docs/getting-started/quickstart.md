# Quick Start

This guide walks you through the basic usage of `dapr-agents-oas-adapter`.

## Loading an OAS Specification

### From YAML

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

yaml_spec = """
component_type: Agent
name: research_assistant
description: An AI assistant for research tasks
system_prompt: You are a helpful research assistant.
tools:
  - web_search
  - summarize
"""

loader = DaprAgentSpecLoader()
config = loader.load_yaml(yaml_spec)

print(f"Agent: {config.name}")
print(f"Tools: {config.tools}")
```

### From JSON

```python
import json
from dapr_agents_oas_adapter import DaprAgentSpecLoader

json_spec = json.dumps({
    "component_type": "Agent",
    "name": "assistant",
    "system_prompt": "You are helpful."
})

loader = DaprAgentSpecLoader()
config = loader.load_json(json_spec)
```

### From Dictionary

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

spec_dict = {
    "component_type": "Agent",
    "name": "assistant",
    "system_prompt": "You are helpful."
}

loader = DaprAgentSpecLoader()
config = loader.load_dict(spec_dict)
```

## Creating a Dapr Agent

After loading the configuration, create an executable agent:

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

loader = DaprAgentSpecLoader()
config = loader.load_yaml(yaml_spec)

# Create the Dapr agent
agent = loader.create_agent(config)

# Use the agent
response = await agent.run("What is machine learning?")
```

## Loading a Workflow

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

workflow_spec = """
component_type: Flow
name: content_pipeline
nodes:
  - component_type: StartNode
    id: start
    name: start
  - component_type: LlmNode
    id: generate
    name: generate_content
    prompt_template: "Write about: {{ topic }}"
  - component_type: EndNode
    id: end
    name: end
control_flow_connections:
  - from_node: {$component_ref: start}
    to_node: {$component_ref: generate}
  - from_node: {$component_ref: generate}
    to_node: {$component_ref: end}
start_node: {$component_ref: start}
"""

loader = DaprAgentSpecLoader()
workflow_def = loader.load_yaml(workflow_spec)

# Create executable workflow function
workflow_fn = loader.create_workflow(workflow_def)
```

## Exporting to OAS

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import DaprAgentConfig

# Create a configuration programmatically
config = DaprAgentConfig(
    name="my_agent",
    role="Assistant",
    goal="Help users",
    instructions=["Be helpful", "Be concise"],
    tools=["search", "calculate"]
)

# Export to OAS format
exporter = DaprAgentSpecExporter()

# As dictionary
oas_dict = exporter.to_dict(config)

# As YAML string
oas_yaml = exporter.to_yaml(config)

# As JSON string
oas_json = exporter.to_json(config)
```

## Validation

Use `StrictLoader` for pre-conversion validation:

```python
from dapr_agents_oas_adapter import StrictLoader
from dapr_agents_oas_adapter.validation import OASSchemaValidationError

loader = StrictLoader()

try:
    config = loader.load_dict({
        "component_type": "Agent",
        "name": "valid_agent",
        "system_prompt": "Hello"
    })
except OASSchemaValidationError as e:
    print(f"Validation errors: {e.issues}")
```

## Next Steps

- [Loading Guide](../guide/loading.md) - Detailed loading options
- [Exporting Guide](../guide/exporting.md) - Export configurations
- [Caching Guide](../guide/caching.md) - Performance optimization
- [API Reference](../api/index.md) - Complete API documentation
