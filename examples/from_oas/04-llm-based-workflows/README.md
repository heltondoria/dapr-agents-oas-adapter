## 04-llm-based-workflows (from_oas)

This example **imports** the OAS YAML exported by:

- `examples/to_oas/04-llm-based-workflows/export_oas.py`

### How to run (from the repo root)

1) Ensure the YAML was exported:

```bash
uv run python examples/to_oas/04-llm-based-workflows/export_oas.py
```

2) Run the imported workflow:

```bash
RESOURCES_DIR=$(uv run python examples/from_oas/04-llm-based-workflows/resolve_resources.py)
dapr run --app-id oas-from-llm-wf --resources-path "$RESOURCES_DIR" -- python examples/from_oas/04-llm-based-workflows/app.py
rm -rf "$RESOURCES_DIR"
```
