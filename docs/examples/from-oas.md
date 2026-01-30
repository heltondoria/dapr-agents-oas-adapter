# From OAS Examples

Examples of loading Open Agent Spec specifications and creating Dapr agents/workflows.

## Simple Agent

Load a basic agent specification:

```yaml title="agent.yaml"
component_type: Agent
name: simple_assistant
description: A helpful assistant
system_prompt: |
  You are a helpful assistant that provides clear and concise answers.
  Always be polite and professional.
tools: []
```

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

loader = DaprAgentSpecLoader()

with open("agent.yaml") as f:
    config = loader.load_yaml(f.read())

print(f"Loaded agent: {config.name}")
print(f"System prompt: {config.system_prompt}")
```

## Agent with Tools

Agent with tool capabilities:

```yaml title="research_agent.yaml"
component_type: Agent
name: research_assistant
description: An AI assistant for research tasks
system_prompt: |
  You are a research assistant with access to tools.
  Use web_search to find information.
  Use summarize to condense long texts.
tools:
  - web_search
  - summarize
```

```python
def web_search(query: str) -> list[str]:
    """Search the web for information."""
    # Implementation
    return [f"Search result for: {query}"]

def summarize(text: str) -> str:
    """Summarize text content."""
    # Implementation
    return f"Summary of {len(text)} chars"

loader = DaprAgentSpecLoader(
    tool_registry={
        "web_search": web_search,
        "summarize": summarize
    }
)

config = loader.load_yaml(yaml_content)
agent = loader.create_agent(config)
```

## Simple Workflow

Linear workflow with LLM processing:

```yaml title="simple_workflow.yaml"
component_type: Flow
name: content_generator
description: Generate content from a topic

nodes:
  - component_type: StartNode
    id: start
    name: start

  - component_type: LlmNode
    id: generate
    name: generate_content
    prompt_template: |
      Write a short article about: {{ topic }}

  - component_type: EndNode
    id: end
    name: end

control_flow_connections:
  - from_node: {$component_ref: start}
    to_node: {$component_ref: generate}
  - from_node: {$component_ref: generate}
    to_node: {$component_ref: end}

start_node: {$component_ref: start}
```

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

loader = DaprAgentSpecLoader()
workflow_def = loader.load_yaml(yaml_content)

# Create executable workflow
workflow_fn = loader.create_workflow(workflow_def)

print(f"Workflow: {workflow_def.name}")
print(f"Tasks: {[t.name for t in workflow_def.tasks]}")
```

## Branching Workflow

Workflow with conditional branching:

```yaml title="branching_workflow.yaml"
component_type: Flow
name: content_router
description: Route content based on classification

nodes:
  - component_type: StartNode
    id: start
    name: start

  - component_type: LlmNode
    id: classify
    name: classify_content
    prompt_template: |
      Classify the following content as 'technical' or 'general':
      {{ content }}

  - component_type: ToolNode
    id: technical
    name: handle_technical

  - component_type: ToolNode
    id: general
    name: handle_general

  - component_type: EndNode
    id: end
    name: end

control_flow_connections:
  - from_node: {$component_ref: start}
    to_node: {$component_ref: classify}
  - from_node: {$component_ref: classify}
    to_node: {$component_ref: technical}
    from_branch: technical
  - from_node: {$component_ref: classify}
    to_node: {$component_ref: general}
    from_branch: general
  - from_node: {$component_ref: technical}
    to_node: {$component_ref: end}
  - from_node: {$component_ref: general}
    to_node: {$component_ref: end}

start_node: {$component_ref: start}
```

## Map/Fan-out Workflow

Parallel processing with MapNode:

```yaml title="parallel_workflow.yaml"
component_type: Flow
name: batch_processor
description: Process items in parallel

nodes:
  - component_type: StartNode
    id: start
    name: start

  - component_type: MapNode
    id: process_items
    name: parallel_processor
    parallel: true

  - component_type: EndNode
    id: end
    name: end

control_flow_connections:
  - from_node: {$component_ref: start}
    to_node: {$component_ref: process_items}
  - from_node: {$component_ref: process_items}
    to_node: {$component_ref: end}

start_node: {$component_ref: start}
```

## With Validation

Using StrictLoader for pre-conversion validation:

```python
from dapr_agents_oas_adapter import StrictLoader
from dapr_agents_oas_adapter.validation import OASSchemaValidationError

loader = StrictLoader()

try:
    config = loader.load_yaml(yaml_content)
    print(f"Valid spec loaded: {config.name}")
except OASSchemaValidationError as e:
    print("Validation failed:")
    for issue in e.issues:
        print(f"  - {issue}")
```

## With Caching

Cached loading for repeated operations:

```python
from dapr_agents_oas_adapter import (
    CachedLoader,
    DaprAgentSpecLoader,
    InMemoryCache
)

cache = InMemoryCache(max_size=50, ttl_seconds=600)
loader = CachedLoader(
    loader=DaprAgentSpecLoader(),
    cache=cache
)

# First load - cache miss
config = loader.load_yaml(yaml_content)

# Second load - cache hit
config = loader.load_yaml(yaml_content)

print(f"Cache hits: {loader.stats.hits}")
```

## Async Loading

Non-blocking async loading:

```python
import asyncio
from dapr_agents_oas_adapter import AsyncDaprAgentSpecLoader

async def load_specs(yaml_contents: list[str]):
    async with AsyncDaprAgentSpecLoader() as loader:
        tasks = [loader.load_yaml(c) for c in yaml_contents]
        return await asyncio.gather(*tasks)

configs = asyncio.run(load_specs(yaml_list))
```
