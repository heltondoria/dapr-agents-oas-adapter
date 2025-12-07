# Dapr Agents OAS Adapter

Adapter library enabling bidirectional interoperability between [Open Agent Spec (OAS)](https://oracle.github.io/agent-spec/) and [Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/).

## Features

- **OAS → Dapr Agents Import**: Loads OAS specifications (JSON/YAML) and creates executable Dapr Agents and workflows
- **Dapr Agents → OAS Export**: Exports Dapr agents and workflows to OAS format
- **Component Converters**: Support for Agent, Flow, LlmConfig, Tool, Node and Edge
- **Code Generation**: Generates Python code for Dapr workflows from OAS specifications
- **Compatibility**: Follows existing adapter patterns (LangGraph, CrewAI)

## Installation

```bash
pip install dapr-agents-oas-adapter
```

Or with development dependencies:

```bash
pip install dapr-agents-oas-adapter[dev]
```

## Quick Start

### Load an OAS specification

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

# Register tool implementations
def search_tool(query: str) -> list[str]:
    """Web search."""
    return ["result1", "result2"]

loader = DaprAgentSpecLoader(
    tool_registry={"search": search_tool}
)

# Load from JSON
with open("agent_spec.json") as f:
    config = loader.load_json(f.read())

# Create executable Dapr agent
agent = loader.create_agent(config)

# Start the agent
await agent.start()
```

### Export to OAS

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import DaprAgentConfig

exporter = DaprAgentSpecExporter()

# Configure agent
config = DaprAgentConfig(
    name="my_assistant",
    role="Assistant",
    goal="Help users",
    instructions=["Be helpful", "Be concise"],
    tools=["search", "calculator"],
)

# Export to JSON
json_spec = exporter.to_json(config)

# Export to YAML file
exporter.to_yaml_file(config, "agent_spec.yaml")
```

### Working with Workflows

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader, DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowTaskDefinition,
    WorkflowEdgeDefinition,
)

# Create workflow definition
workflow = WorkflowDefinition(
    name="my_workflow",
    description="Processes user data",
    tasks=[
        WorkflowTaskDefinition(name="start", task_type="start"),
        WorkflowTaskDefinition(
            name="analyze",
            task_type="llm",
            config={"prompt_template": "Analyze: {{input}}"},
        ),
        WorkflowTaskDefinition(name="end", task_type="end"),
    ],
    edges=[
        WorkflowEdgeDefinition(from_node="start", to_node="analyze"),
        WorkflowEdgeDefinition(from_node="analyze", to_node="end"),
    ],
    start_node="start",
    end_nodes=["end"],
)

# Export to OAS
exporter = DaprAgentSpecExporter()
oas_spec = exporter.to_json(workflow)

# Generate workflow Python code
loader = DaprAgentSpecLoader()
code = loader.generate_workflow_code(workflow)
print(code)
```

## Component Mapping

| OAS Component | Dapr Agents Equivalent |
|---------------|------------------------|
| `Agent` | `AssistantAgent` / `ReActAgent` |
| `Flow` | `@workflow` decorated function |
| `LlmNode` | `@task` with LLM call |
| `ToolNode` | `@task` with tool call |
| `StartNode` | Workflow entry point |
| `EndNode` | Workflow return |
| `ServerTool` | `@tool` decorated function |
| `MCPTool` | MCP integration via Dapr |
| `LlmConfig` | `DaprChatClient` config |
| `ControlFlowEdge` | Sequence of `yield ctx.call_activity()` |
| `DataFlowEdge` | Parameter passing between tasks |

## API Reference

### DaprAgentSpecLoader

```python
class DaprAgentSpecLoader:
    def __init__(self, tool_registry: dict[str, Callable] | None = None)
    def load_json(self, json_content: str) -> DaprAgentConfig | WorkflowDefinition
    def load_yaml(self, yaml_content: str) -> DaprAgentConfig | WorkflowDefinition
    def load_json_file(self, file_path: str | Path) -> DaprAgentConfig | WorkflowDefinition
    def load_yaml_file(self, file_path: str | Path) -> DaprAgentConfig | WorkflowDefinition
    def load_component(self, component: Component) -> DaprAgentConfig | WorkflowDefinition
    def load_dict(self, spec_dict: dict) -> DaprAgentConfig | WorkflowDefinition
    def create_agent(self, config: DaprAgentConfig, additional_tools: dict | None = None) -> Any
    def create_workflow(self, workflow_def: WorkflowDefinition, task_implementations: dict | None = None) -> Callable
    def generate_workflow_code(self, workflow_def: WorkflowDefinition) -> str
    def register_tool(self, name: str, implementation: Callable) -> None
```

### DaprAgentSpecExporter

```python
class DaprAgentSpecExporter:
    def to_json(self, component: DaprAgentConfig | WorkflowDefinition, indent: int = 2) -> str
    def to_yaml(self, component: DaprAgentConfig | WorkflowDefinition) -> str
    def to_dict(self, component: DaprAgentConfig | WorkflowDefinition) -> dict
    def to_component(self, component: DaprAgentConfig | WorkflowDefinition) -> Component
    def to_json_file(self, component: DaprAgentConfig | WorkflowDefinition, file_path: str | Path) -> None
    def to_yaml_file(self, component: DaprAgentConfig | WorkflowDefinition, file_path: str | Path) -> None
    def from_dapr_agent(self, agent: Any) -> DaprAgentConfig
    def from_dapr_workflow(self, workflow_func: Callable, task_funcs: list[Callable] | None = None) -> WorkflowDefinition
    def export_agent_to_json(self, agent: Any) -> str
    def export_agent_to_yaml(self, agent: Any) -> str
```

## Development

### Setup

```bash
git clone https://github.com/your-username/dapr-agents-oas-adapter.git
cd dapr-agents-oas-adapter
uv sync --all-extras
```

### Run Tests

```bash
pytest
```

### Linting and Type Checking

```bash
ruff check src/
mypy src/
```

## Requirements

- Python >= 3.12
- pyagentspec >= 25.4.1
- dapr-agents >= 0.10.4
- dapr >= 1.16.0
- pydantic >= 2.0.0

## License

MIT License

## Links

- [Open Agent Spec](https://oracle.github.io/agent-spec/)
- [Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/)
- [PyAgentSpec](https://github.com/oracle/agent-spec)
- [Dapr](https://dapr.io/)
