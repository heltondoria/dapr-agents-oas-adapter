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
RESOURCES_DIR=$(uv run python examples/from_oas/04-agent-based-workflows/resolve_resources.py)
dapr run --app-id oas-from-agent-wf --resources-path "$RESOURCES_DIR" -- python examples/from_oas/04-agent-based-workflows/app.py
rm -rf "$RESOURCES_DIR"
```

