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
cp secrets.json.template secrets.json
# Edit `secrets.json` and set `openai-secrets.OPENAI_API_KEY`.

dapr run -f examples/to_oas/04-agent-based-workflows

# Stop:
dapr stop -f examples/to_oas/04-agent-based-workflows
```
