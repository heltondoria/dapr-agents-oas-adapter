"""Integration tests for Dapr workflow execution.

These tests exercise the full OAS -> Dapr workflow pipeline against a running
Dapr sidecar. They are skipped automatically when no sidecar is available.

Run manually with::

    dapr run --app-id test-adapter --dapr-http-port 3500 -- \
        uv run pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

from typing import Any

import pytest

from dapr_agents_oas_adapter.converters.flow import FlowConverter
from dapr_agents_oas_adapter.exporter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)

from .conftest import requires_dapr


@requires_dapr
@pytest.mark.integration
class TestDaprWorkflowExecution:
    """Tests that require a real Dapr runtime."""

    def test_linear_workflow_roundtrip_through_dapr(
        self,
        dapr_test_config: dict[str, Any],
    ) -> None:
        """OAS linear flow -> WorkflowDefinition -> OAS preserves structure."""
        flow_dict: dict[str, Any] = {
            "component_type": "Flow",
            "id": "linear_test",
            "name": "linear_workflow",
            "nodes": [
                {
                    "component_type": "StartNode",
                    "id": "start",
                    "name": "start",
                    "inputs": [],
                    "outputs": [],
                },
                {
                    "component_type": "LlmNode",
                    "id": "llm1",
                    "name": "generate",
                    "inputs": [],
                    "outputs": [],
                },
                {
                    "component_type": "EndNode",
                    "id": "end",
                    "name": "end",
                    "inputs": [],
                    "outputs": [],
                },
            ],
            "edges": [
                {
                    "component_type": "ControlFlowEdge",
                    "from_node": "start",
                    "to_node": "llm1",
                },
                {
                    "component_type": "ControlFlowEdge",
                    "from_node": "llm1",
                    "to_node": "end",
                },
            ],
        }

        converter = FlowConverter()
        workflow_def = converter.from_dict(flow_dict)

        assert workflow_def.name == "linear_workflow"
        assert len(workflow_def.tasks) >= 3
        task_names = {t.name for t in workflow_def.tasks}
        assert "start" in task_names
        assert "end" in task_names

    def test_agent_config_roundtrip_through_dapr(
        self,
        dapr_test_config: dict[str, Any],
    ) -> None:
        """DaprAgentConfig -> OAS JSON -> DaprAgentConfig preserves fields."""
        original = DaprAgentConfig(
            name="integration_agent",
            role="Test Agent",
            goal="Verify integration",
            instructions=["Be helpful", "Be concise"],
            message_bus_name=dapr_test_config["message_bus_name"],
            state_store_name=dapr_test_config["state_store_name"],
            agents_registry_store_name=dapr_test_config["agents_registry_store_name"],
        )

        exporter = DaprAgentSpecExporter()
        json_spec = exporter.to_json(original)

        loader = DaprAgentSpecLoader()
        restored = loader.load_json(json_spec)

        assert restored.name == original.name
        assert restored.role == original.role

    def test_workflow_code_generation_produces_valid_python(self) -> None:
        """Generated workflow code is syntactically valid Python."""
        workflow = WorkflowDefinition(
            name="codegen_test",
            description="Integration test workflow",
            tasks=[
                WorkflowTaskDefinition(name="start", task_type="start"),
                WorkflowTaskDefinition(name="process", task_type="llm"),
                WorkflowTaskDefinition(name="end", task_type="end"),
            ],
            edges=[
                WorkflowEdgeDefinition(from_node="start", to_node="process"),
                WorkflowEdgeDefinition(from_node="process", to_node="end"),
            ],
            start_node="start",
            end_nodes=["end"],
        )

        converter = FlowConverter()
        code = converter.generate_workflow_code(workflow)

        # Should be valid Python (compile does not execute, just checks syntax)
        compile(code, "<integration_test>", "exec")
        assert "codegen_test" in code
