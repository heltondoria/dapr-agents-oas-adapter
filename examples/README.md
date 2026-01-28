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
- OpenAI key configured via the local secret store template:
  - copy `secrets.json.template` to `secrets.json`
  - set `openai-secrets.OPENAI_API_KEY`

