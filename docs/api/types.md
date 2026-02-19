# Types

Data models for agent and workflow configurations.

## DaprAgentConfig

Configuration model for Dapr Agent creation.

::: dapr_agents_oas_adapter.types.DaprAgentConfig
    options:
      show_root_heading: true
      show_source: true
      members:
        - name
        - role
        - goal
        - instructions
        - system_prompt
        - tools
        - llm_config
        - agent_type

## WorkflowDefinition

Definition for a converted workflow.

::: dapr_agents_oas_adapter.types.WorkflowDefinition
    options:
      show_root_heading: true
      show_source: true
      members:
        - name
        - description
        - tasks
        - edges
        - start_node
        - end_nodes
        - inputs
        - outputs
        - subflows

## WorkflowTaskDefinition

Definition for a workflow task.

::: dapr_agents_oas_adapter.types.WorkflowTaskDefinition
    options:
      show_root_heading: true
      show_source: true
      members:
        - name
        - task_type
        - config
        - inputs
        - outputs

## WorkflowEdgeDefinition

Definition for workflow edges (control and data flow).

::: dapr_agents_oas_adapter.types.WorkflowEdgeDefinition
    options:
      show_root_heading: true
      show_source: true
      members:
        - from_node
        - to_node
        - from_branch
        - condition
        - data_mapping

## ToolDefinition

Definition for a converted tool.

::: dapr_agents_oas_adapter.types.ToolDefinition
    options:
      show_root_heading: true
      show_source: true

## LlmProviderConfig

Configuration for LLM provider.

::: dapr_agents_oas_adapter.types.LlmProviderConfig
    options:
      show_root_heading: true
      show_source: true

## Enums

### OASComponentType

::: dapr_agents_oas_adapter.types.OASComponentType
    options:
      show_root_heading: true

### DaprAgentType

::: dapr_agents_oas_adapter.types.DaprAgentType
    options:
      show_root_heading: true

## Usage Examples

### Creating Configurations Programmatically

```python
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    WorkflowDefinition,
    WorkflowTaskDefinition,
    WorkflowEdgeDefinition
)

# Create an agent config
agent = DaprAgentConfig(
    name="research_agent",
    role="Researcher",
    goal="Find information",
    instructions=["Search thoroughly", "Cite sources"],
    tools=["web_search", "summarize"]
)

# Create a workflow
workflow = WorkflowDefinition(
    name="analysis_pipeline",
    tasks=[
        WorkflowTaskDefinition(name="start", task_type="start"),
        WorkflowTaskDefinition(
            name="analyze",
            task_type="llm",
            config={"prompt_template": "Analyze: {{ input }}"}
        ),
        WorkflowTaskDefinition(name="end", task_type="end")
    ],
    edges=[
        WorkflowEdgeDefinition(from_node="start", to_node="analyze"),
        WorkflowEdgeDefinition(from_node="analyze", to_node="end")
    ],
    start_node="start",
    end_nodes=["end"]
)
```

### Task Types

| Type | Description |
|------|-------------|
| `start` | Workflow entry point |
| `end` | Workflow exit point |
| `llm` | LLM call task |
| `tool` | Tool invocation |
| `agent` | Agent delegation |
| `flow` | Child workflow call |
| `map` | Parallel fan-out |
