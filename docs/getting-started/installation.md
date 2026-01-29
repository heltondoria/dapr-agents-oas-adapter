# Installation

## Requirements

- Python 3.12 or higher
- [Dapr](https://dapr.io/) runtime (for agent execution)

## Install with pip

```bash
pip install dapr-agents-oas-adapter
```

## Install with uv

```bash
uv add dapr-agents-oas-adapter
```

## Install from source

```bash
git clone https://github.com/heltondoria/dapr-agents-oas-adapter.git
cd dapr-agents-oas-adapter
uv sync --all-groups
```

## Dependencies

The library automatically installs the following dependencies:

- `pyagentspec>=25.4.1` - Open Agent Spec Python SDK
- `dapr-agents>=0.10.5` - Dapr Agents framework
- `dapr>=1.16.0` - Dapr Python SDK
- `pydantic>=2.12.5` - Data validation
- `structlog>=24.1.0` - Structured logging

## Verify Installation

```python
import dapr_agents_oas_adapter

print(dapr_agents_oas_adapter.__version__)
```

## Development Installation

For development, install with all dependency groups:

```bash
uv sync --all-groups
```

This includes:

- **dev**: Testing and linting tools (pytest, ruff, hypothesis)
- **docs**: Documentation tools (mkdocs, mkdocstrings)
