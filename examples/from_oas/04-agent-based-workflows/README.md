## 04-agent-based-workflows (from_oas)

This example **imports** the OAS YAMLs exported by:

- `examples/to_oas/04-agent-based-workflows/export_oas.py`

### How to run (from the repo root)

Run these commands from the repository root (the folder containing `pyproject.toml`).

1) Ensure the YAMLs were exported:

```bash
uv run python examples/to_oas/04-agent-based-workflows/export_oas.py
```

2) Run the workflow using agents created from OAS:

```bash
cp secrets.json.template secrets.json
# Edit `secrets.json` and set `openai-secrets.OPENAI_API_KEY`.

dapr run -f examples/from_oas/04-agent-based-workflows

# Stop:
dapr stop -f examples/from_oas/04-agent-based-workflows
```

