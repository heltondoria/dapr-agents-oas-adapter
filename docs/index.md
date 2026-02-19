# dapr-agents-oas-adapter

**Bidirectional conversion between Open Agent Spec (OAS) and Dapr Agents.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/heltondoria/dapr-agents-oas-adapter/blob/main/LICENSE)

## Overview

`dapr-agents-oas-adapter` enables seamless interoperability between [Open Agent Spec (OAS)](https://oracle.github.io/agent-spec/) specifications and [Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/). This library provides:

- **Import OAS specs** to create executable Dapr Agents and Workflows
- **Export Dapr Agents** to portable OAS format
- **Validation** of OAS specifications before conversion
- **Caching** for improved performance on repeated operations
- **Async support** for non-blocking operations
- **Structured logging** for observability

## Key Features

### Bidirectional Conversion

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader, DaprAgentSpecExporter

# Load OAS spec and create Dapr agent
loader = DaprAgentSpecLoader()
config = loader.load_yaml(oas_yaml)
agent = loader.create_agent(config)

# Export Dapr agent to OAS format
exporter = DaprAgentSpecExporter()
oas_dict = exporter.to_dict(config)
```

### Schema Validation

```python
from dapr_agents_oas_adapter import StrictLoader

# Validate OAS spec before conversion
loader = StrictLoader()
try:
    config = loader.load_dict(oas_dict)
except OASSchemaValidationError as e:
    print(f"Validation failed: {e.issues}")
```

### Caching Support

```python
from dapr_agents_oas_adapter import CachedLoader, InMemoryCache

cache = InMemoryCache(max_size=100, ttl_seconds=300)
loader = CachedLoader(loader=DaprAgentSpecLoader(), cache=cache)

# Subsequent loads hit the cache
config = loader.load_yaml(yaml_content)
```

### Async Operations

```python
from dapr_agents_oas_adapter import AsyncDaprAgentSpecLoader

async with AsyncDaprAgentSpecLoader() as loader:
    config = await loader.load_dict(oas_dict)
```

## Installation

```bash
pip install dapr-agents-oas-adapter
```

Or with uv:

```bash
uv add dapr-agents-oas-adapter
```

## Quick Links

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [API Reference](api/index.md)
- [Examples](examples/from-oas.md)

## Component Mapping

| OAS Component | Dapr Agents |
|---------------|-------------|
| Agent | AssistantAgent / ReActAgent |
| Flow | @workflow decorated function |
| LlmNode | @task with LLM call |
| ToolNode | @task with tool call |
| FlowNode | ctx.call_child_workflow() |
| MapNode | Fan-out with wf.when_all() |
| ControlFlowEdge | Branch routing via from_branch |

## License

Apache License 2.0
