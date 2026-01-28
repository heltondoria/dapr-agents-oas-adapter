# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dapr-agents-oas-adapter** is a Python library enabling bidirectional conversion between [Open Agent Spec (OAS)](https://oracle.github.io/agent-spec/) and [Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/). It allows OAS specifications to be imported as executable Dapr Agents/workflows and vice versa.

## Common Commands

```bash
# Setup
uv sync --all-groups              # Install all dependencies including dev

# Testing
uv run pytest                     # Run all tests
uv run pytest -v                  # Verbose output
uv run pytest --cov=src/dapr_agents_oas_adapter --cov-fail-under=90  # With coverage (90% required)
uv run pytest tests/test_converters.py -k "test_name"  # Run single test

# Linting & Formatting
uv run ruff check .               # Lint code
uv run ruff check --fix .         # Auto-fix lint issues
uv run ruff format .              # Format code
uv run ruff format --check .      # Check formatting without changes

# Type Checking
uv run ty check                   # Type check (replaces mypy/pyright)

# Code Quality
uv run codespell .                # Spell check
uv run vulture .                  # Dead code detection
```

## Architecture

### Core Pattern: Bidirectional Converters

```
OAS Specification <--> [Converters] <--> Dapr Agents Components
```

**Main Entry Points:**
- `DaprAgentSpecLoader` - Import OAS specs, create Dapr agents/workflows
- `DaprAgentSpecExporter` - Export Dapr agents/workflows to OAS format

### Source Structure

```
src/dapr_agents_oas_adapter/
├── converters/           # Bidirectional conversion logic
│   ├── base.py           # Abstract ComponentConverter[OASType, DaprType]
│   ├── agent.py          # Agent <-> OAS Agent
│   ├── flow.py           # Workflow <-> OAS Flow
│   ├── node.py           # Node types (LLM, Tool, Agent, Flow, Map)
│   ├── tool.py           # Tool definitions including MCP
│   └── llm.py            # LLM configs (OpenAI, Ollama, VLLM, OCI)
├── exporter.py           # DaprAgentSpecExporter - export to OAS
├── loader.py             # DaprAgentSpecLoader - import from OAS
├── types.py              # Type definitions, enums, Pydantic models
├── state.py              # State schema builder
└── utils.py              # Template rendering utilities
```

### Key Data Models (types.py)

- `DaprAgentConfig` - Agent configuration (name, role, goal, instructions, tools)
- `WorkflowDefinition` - Workflow with tasks, edges, start/end nodes
- `WorkflowTaskDefinition` - Individual task (type: llm, tool, agent, flow, map)
- `WorkflowEdgeDefinition` - Task connections (control/data flow)
- `ToolDefinition` - Tool metadata and implementation reference

### Component Mapping

| OAS | Dapr Agents |
|-----|-------------|
| Agent | AssistantAgent / ReActAgent |
| Flow | @workflow decorated function |
| LlmNode | @task with LLM call |
| ToolNode | @task with tool call |
| FlowNode | ctx.call_child_workflow() |
| MapNode | Fan-out with wf.when_all() |
| ControlFlowEdge | Branch routing via from_branch |

### Converter Pattern

All converters inherit from `ComponentConverter[OASType, DaprType]` and implement:
- `from_oas(component)` - Convert OAS to Dapr type
- `to_oas(config)` - Convert Dapr type to OAS
- `can_convert(component)` - Check if converter handles this component

### Template System

Uses `{{ variable }}` syntax for dynamic content in prompts and configurations:
- `extract_template_variables(template)` - Find placeholders
- `render_template(template, values)` - Replace with values

## Workflow Runtime Features

Branching uses `ControlFlowEdge.from_branch` values. Runtime hints in metadata:

```yaml
metadata:
  dapr:
    retry_policy:
      max_attempts: 3
    timeout_seconds: 20
    branch_output_key: decision
    map_input_key: items
```

When workflows have subflows, register child workflows before parent:
```python
workflow_fn = loader.create_workflow(workflow_def)
for child in getattr(workflow_fn, "child_workflows", []):
    runtime.register_workflow(child)
runtime.register_workflow(workflow_fn)
```

## Examples

Examples are in `examples/` with two categories:
- `to_oas/` - Export Dapr agents/workflows to OAS
- `from_oas/` - Import OAS specs to create Dapr agents/workflows

Each example has a `dapr.yaml` for Dapr configuration. Run with:
```bash
dapr run -f dapr.yaml -- python script.py
```

## CI Quality Gates

All PRs must pass: lint, format, type-check, spell-check, dead-code, tests (90% coverage), security scan.
