## 04-message-router-workflow (to_oas)

Based on `dapr-agents/quickstarts/04-message-router-workflow`.

### How to run (from the repo root)

Run these commands from the repository root (the folder containing `pyproject.toml`).

1) Export the OAS YAML:

```bash
uv run python examples/to_oas/04-message-router-workflow/export_oas.py
```

1) Start the app (subscriber + workflow runtime):

```bash
cp secrets.json.template secrets.json
# Edit `secrets.json` and set `openai-secrets.OPENAI_API_KEY`.

dapr run -f examples/to_oas/04-message-router-workflow
```

1) In another terminal, publish a message:

```bash
dapr run -f examples/to_oas/04-message-router-workflow/dapr.client.yaml
```

Stop the server:

```bash
dapr stop -f examples/to_oas/04-message-router-workflow
```
