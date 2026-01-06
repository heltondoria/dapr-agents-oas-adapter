## 04-message-router-workflow (from_oas)

This example **imports** the OAS YAML exported by:

- `examples/to_oas/04-message-router-workflow/export_oas.py`

### How to run (from the repo root)

Run these commands from the repository root (the folder containing `pyproject.toml`).

1) Ensure the YAML was exported:

```bash
uv run python examples/to_oas/04-message-router-workflow/export_oas.py
```

2) Start the app (subscriber + workflow runtime):

```bash
RESOURCES_DIR=$(uv run python examples/from_oas/04-message-router-workflow/resolve_resources.py)
dapr run --app-id oas-from-message-workflow --resources-path "$RESOURCES_DIR" -- python examples/from_oas/04-message-router-workflow/app.py
rm -rf "$RESOURCES_DIR"
```

3) In another terminal, publish a message:

```bash
RESOURCES_DIR=$(uv run python examples/from_oas/04-message-router-workflow/resolve_resources.py)
dapr run --app-id oas-from-message-workflow-client --resources-path "$RESOURCES_DIR" -- python examples/from_oas/04-message-router-workflow/message_client.py
rm -rf "$RESOURCES_DIR"
```

