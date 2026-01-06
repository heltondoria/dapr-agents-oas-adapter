## 04-message-router-workflow (to_oas)

Based on `dapr-agents/quickstarts/04-message-router-workflow`.

### How to run (from the repo root)

Run these commands from the repository root (the folder containing `pyproject.toml`).

1) Export the OAS YAML:

```bash
uv run python examples/to_oas/04-message-router-workflow/export_oas.py
```

2) Start the app (subscriber + workflow runtime):

```bash
RESOURCES_DIR=$(uv run python examples/to_oas/04-message-router-workflow/resolve_resources.py)
dapr run --app-id message-workflow --resources-path "$RESOURCES_DIR" -- python examples/to_oas/04-message-router-workflow/app.py
rm -rf "$RESOURCES_DIR"
```

3) In another terminal, publish a message:

```bash
RESOURCES_DIR=$(uv run python examples/to_oas/04-message-router-workflow/resolve_resources.py)
dapr run --app-id message-workflow-client --resources-path "$RESOURCES_DIR" -- python examples/to_oas/04-message-router-workflow/message_client.py
rm -rf "$RESOURCES_DIR"
```
