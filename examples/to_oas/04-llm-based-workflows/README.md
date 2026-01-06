## 04-llm-based-workflows (to_oas)

Based on `dapr-agents/quickstarts/04-llm-based-workflows`.

### How to run

1) Export OAS YAMLs:

```bash
uv run python examples/to_oas/04-llm-based-workflows/export_oas.py
```

2) Run the workflow with Dapr:

```bash
RESOURCES_DIR=$(uv run python examples/to_oas/04-llm-based-workflows/resolve_resources.py)
dapr run --app-id oas-to-llm-wf --resources-path "$RESOURCES_DIR" -- python examples/to_oas/04-llm-based-workflows/workflow_single.py
rm -rf "$RESOURCES_DIR"
```

### Note

This example does not require `python-dotenv`. If you want to use a `.env` file, install `python-dotenv`, or export `OPENAI_API_KEY` in your shell.
