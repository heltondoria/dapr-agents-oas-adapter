## Real-world examples for `dapr-agents-oas-adapter`

This folder contains runnable examples that validate the adapter in real scenarios with **Dapr Workflows** and the **`conversation.openai`** component.

### Structure

- `to_oas/`: “source” examples that export OAS specs (YAML)
- `from_oas/`: “derived” examples that import YAML and recreate workflows/agents
- `_shared/`: shared utilities (no duplication)

### Prerequisites

- Python 3.12+
- `uv sync --all-groups`
- Dapr CLI + Docker
- `dapr init`
- `OPENAI_API_KEY` set in the environment (or a `.env` file inside the example directory)

