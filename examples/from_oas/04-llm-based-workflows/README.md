## 04-llm-based-workflows (from_oas)

This example **imports** the OAS YAML exported by:

- `examples/to_oas/04-llm-based-workflows/export_oas.py`

### How to run (from the repo root)

Run these commands from the repository root (the folder containing `pyproject.toml`).

1) Ensure the YAML was exported:

```bash
uv run python examples/to_oas/04-llm-based-workflows/export_oas.py
```

2) Run the imported workflow:

```bash
cp secrets.json.template secrets.json
# Edit `secrets.json` and set `openai-secrets.OPENAI_API_KEY`.

dapr run -f examples/from_oas/04-llm-based-workflows

# Stop:
dapr stop -f examples/from_oas/04-llm-based-workflows
```
