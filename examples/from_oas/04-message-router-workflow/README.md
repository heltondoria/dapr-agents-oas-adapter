# 04-message-router-workflow (from_oas)

This example **imports** the OAS YAML exported by:

- `examples/to_oas/04-message-router-workflow/export_oas.py`

### How to run (from the repo root)

Run these commands from the repository root (the folder containing `pyproject.toml`).

1) Ensure the YAML was exported:

```bash
uv run python examples/to_oas/04-message-router-workflow/export_oas.py
```

1) Start the app (subscriber + workflow runtime):

```bash
cp secrets.json.template secrets.json
# Edit `secrets.json` and set `openai-secrets.OPENAI_API_KEY`.

dapr run -f examples/from_oas/04-message-router-workflow
```

1) In another terminal, publish a message:

```bash
dapr run -f examples/from_oas/04-message-router-workflow/dapr.client.yaml
```

Stop the server:

```bash
dapr stop -f examples/from_oas/04-message-router-workflow
```

