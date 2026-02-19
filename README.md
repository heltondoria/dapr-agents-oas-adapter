# dapr-agents-oas-adapter

**Bidirectional conversion between Open Agent Spec (OAS) and Dapr Agents.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/heltondoria/dapr-agents-oas-adapter/actions/workflows/ci.yml/badge.svg)](https://github.com/heltondoria/dapr-agents-oas-adapter/actions/workflows/ci.yml)

## Overview

`dapr-agents-oas-adapter` enables seamless interoperability between [Open Agent Spec (OAS)](https://oracle.github.io/agent-spec/) specifications and [Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/). Import OAS specifications to create executable Dapr Agents and workflows, or export existing Dapr Agents to portable OAS format.

## Key Features

- **Bidirectional conversion** -- Import OAS specs into Dapr Agents and export back to OAS
- **Schema validation** -- Validate OAS specifications before conversion with detailed error reports
- **Caching** -- In-memory cache with configurable TTL for repeated operations
- **Async support** -- Non-blocking loader for high-throughput applications
- **Structured logging** -- Built-in `structlog` integration for observability

## Installation

```bash
pip install dapr-agents-oas-adapter
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add dapr-agents-oas-adapter
```

## Quick Start

### Load an OAS spec and create a Dapr Agent

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

loader = DaprAgentSpecLoader()
config = loader.load_yaml(oas_yaml)
agent = loader.create_agent(config)
```

### Export a Dapr Agent to OAS format

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter

exporter = DaprAgentSpecExporter()
oas_dict = exporter.to_dict(config)
```

### Validate before conversion

```python
from dapr_agents_oas_adapter import StrictLoader

loader = StrictLoader()
try:
    config = loader.load_dict(oas_dict)
except OASSchemaValidationError as e:
    print(f"Validation failed: {e.issues}")
```

## Component Mapping

| OAS Component | Dapr Agents |
|---------------|-------------|
| Agent | AssistantAgent / ReActAgent |
| Flow | `@workflow` decorated function |
| LlmNode | `@task` with LLM call |
| ToolNode | `@task` with tool call |
| FlowNode | `ctx.call_child_workflow()` |
| MapNode | Fan-out with `wf.when_all()` |
| ControlFlowEdge | Branch routing via `from_branch` |

## Running the Examples

### Prerequisites

- [Dapr CLI](https://docs.dapr.io/getting-started/install-dapr-cli/) installed and initialized
- An OpenAI API key in `examples/_shared/dapr/components/secrets/`

### Run an example

```bash
cd examples/from_oas/04-agent-based-workflows
dapr run -f dapr.yaml -- python app.py
```

Examples are organized in two directions:

- `examples/from_oas/` -- Import OAS specs to create Dapr Agents
- `examples/to_oas/` -- Export Dapr Agents to OAS format

## Development

```bash
git clone https://github.com/heltondoria/dapr-agents-oas-adapter.git
cd dapr-agents-oas-adapter
uv sync --all-groups
```

```bash
uv run pytest                          # Run tests
uv run ruff check .                    # Lint
uv run ruff format --check .           # Check formatting
uv run ty check                        # Type check
uv run codespell .                     # Spell check
uv run vulture .                       # Dead code detection
```

Tests require 90% coverage:

```bash
uv run pytest --cov=src/dapr_agents_oas_adapter --cov-fail-under=90
```

## Documentation

Full documentation is available at [heltondoria.github.io/dapr-agents-oas-adapter](https://heltondoria.github.io/dapr-agents-oas-adapter/).

To serve locally:

```bash
uv run mkdocs serve
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE) for details.
