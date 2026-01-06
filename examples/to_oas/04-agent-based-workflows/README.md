## 04-agent-based-workflows (to_oas)

Based on `dapr-agents/quickstarts/04-agent-based-workflows`.

### How to run (from the repo root)

Run these commands from the repository root (the folder containing `pyproject.toml`).

1) Export OAS YAMLs (agents):

```bash
uv run python examples/to_oas/04-agent-based-workflows/export_oas.py
```

1) Run the workflow with Dapr:

```bash
RESOURCES_DIR=$(uv run python examples/to_oas/04-agent-based-workflows/resolve_resources.py)
dapr run --app-id oas-to-agent-wf --resources-path "$RESOURCES_DIR" -- python examples/to_oas/04-agent-based-workflows/workflow_agents.py
rm -rf "$RESOURCES_DIR"
```
