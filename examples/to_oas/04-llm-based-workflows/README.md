## 04-llm-based-workflows (to_oas)

Based on `dapr-agents/quickstarts/04-llm-based-workflows`.

### How to run

Run these commands from the repository root (the folder containing `pyproject.toml`).

1) Export OAS YAMLs:

```bash
uv run python examples/to_oas/04-llm-based-workflows/export_oas.py
```

2) Run the workflow with Dapr:

```bash
cp secrets.json.template secrets.json
# Edit `secrets.json` and set `openai-secrets.OPENAI_API_KEY`.

dapr run -f examples/to_oas/04-llm-based-workflows

# Stop:
dapr stop -f examples/to_oas/04-llm-based-workflows
```

### Note

This example does not require `python-dotenv`. Secrets are loaded via `secretstores.local.file` (`secrets.json` in the repo root).
