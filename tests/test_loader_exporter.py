"""Tests for loader and exporter modules."""

from unittest.mock import MagicMock, patch

import pytest

from dapr_agents_oas_adapter.converters.base import ConversionError
from dapr_agents_oas_adapter.exporter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)


class TestDaprAgentSpecLoader:
    """Tests for DaprAgentSpecLoader."""

    def test_init_with_tool_registry(self) -> None:
        """Test initialization with tool registry."""
        def my_tool() -> str:
            return "result"

        loader = DaprAgentSpecLoader(tool_registry={"my_tool": my_tool})
        assert "my_tool" in loader.tool_registry
        assert loader.tool_registry["my_tool"] is my_tool

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        loader = DaprAgentSpecLoader()

        def search(query: str) -> list[str]:
            return [query]

        loader.register_tool("search", search)
        assert "search" in loader.tool_registry
        assert loader.tool_registry["search"] is search

    def test_load_dict_agent(self) -> None:
        """Test loading agent from dictionary."""
        loader = DaprAgentSpecLoader()
        agent_dict = {
            "component_type": "Agent",
            "name": "test_agent",
            "description": "Test agent",
            "system_prompt": "You are helpful.",
            "llm_config": {
                "component_type": "OpenAIConfig",
                "model_id": "gpt-4",
            },
            "tools": [],
        }
        result = loader.load_dict(agent_dict)
        assert isinstance(result, DaprAgentConfig)
        assert result.name == "test_agent"

    def test_load_dict_flow(self) -> None:
        """Test loading flow from dictionary."""
        loader = DaprAgentSpecLoader()
        flow_dict = {
            "component_type": "Flow",
            "name": "test_flow",
            "description": "Test flow",
            "nodes": [
                {
                    "component_type": "StartNode",
                    "id": "start",
                    "name": "start",
                    "inputs": [],
                    "outputs": [],
                }
            ],
            "control_flow_connections": [],
            "data_flow_connections": [],
        }
        result = loader.load_dict(flow_dict)
        assert isinstance(result, WorkflowDefinition)
        assert result.name == "test_flow"

    def test_load_dict_unsupported_type(self) -> None:
        """Test loading unsupported component type."""
        loader = DaprAgentSpecLoader()
        with pytest.raises(ConversionError):
            loader.load_dict({"component_type": "Unknown"})

    def test_generate_workflow_code(self) -> None:
        """Test workflow code generation."""
        loader = DaprAgentSpecLoader()
        workflow = WorkflowDefinition(
            name="test_workflow",
            tasks=[
                WorkflowTaskDefinition(name="start", task_type="start"),
                WorkflowTaskDefinition(name="end", task_type="end"),
            ],
            edges=[],
            start_node="start",
            end_nodes=["end"],
        )
        code = loader.generate_workflow_code(workflow)
        assert "test_workflow" in code
        assert "@workflow" in code


class TestDaprAgentSpecExporter:
    """Tests for DaprAgentSpecExporter."""

    def test_to_dict_agent(self) -> None:
        """Test exporting agent config to dictionary."""
        exporter = DaprAgentSpecExporter()
        config = DaprAgentConfig(
            name="assistant",
            role="Helper",
            goal="Help users",
            instructions=["Be helpful", "Be concise"],
            tools=["search"],
        )
        result = exporter.to_dict(config)
        assert result["component_type"] == "Agent"
        assert result["name"] == "assistant"
        assert result["agentspec_version"] == "25.4.1"

    def test_to_dict_workflow(self) -> None:
        """Test exporting workflow definition to dictionary."""
        exporter = DaprAgentSpecExporter()
        workflow = WorkflowDefinition(
            name="my_workflow",
            description="Test workflow",
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
        result = exporter.to_dict(workflow)
        assert result["component_type"] == "Flow"
        assert result["name"] == "my_workflow"
        assert result["agentspec_version"] == "25.4.1"

    def test_from_dapr_agent(self) -> None:
        """Test extracting config from Dapr agent."""
        exporter = DaprAgentSpecExporter()

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.role = "Assistant"
        mock_agent.goal = "Help users"
        mock_agent.instructions = ["Be helpful"]
        mock_agent.tools = []
        mock_agent.message_bus_name = "kafka"
        mock_agent.state_store_name = "redis"
        mock_agent.agents_registry_store_name = "postgres"
        mock_agent.service_port = 9000

        result = exporter.from_dapr_agent(mock_agent)
        assert result.name == "test_agent"
        assert result.role == "Assistant"
        assert result.goal == "Help users"
        assert result.message_bus_name == "kafka"
        assert result.service_port == 9000

    def test_from_dapr_workflow(self) -> None:
        """Test extracting definition from Dapr workflow function."""
        exporter = DaprAgentSpecExporter()

        def my_workflow(ctx: any, params: dict) -> dict:
            """My test workflow."""
            return {"result": "done"}

        def task1(ctx: any, input_data: dict) -> dict:
            """First task."""
            return input_data

        def task2(ctx: any, data: dict) -> dict:
            """Second task."""
            return data

        result = exporter.from_dapr_workflow(my_workflow, [task1, task2])
        assert result.name == "my_workflow"
        assert "My test workflow" in result.description
        assert len(result.tasks) == 4  # start + 2 tasks + end

    def test_export_agent_to_json(self) -> None:
        """Test convenience method for agent JSON export."""
        exporter = DaprAgentSpecExporter()

        mock_agent = MagicMock()
        mock_agent.name = "json_agent"
        mock_agent.role = "Helper"
        mock_agent.goal = "Assist"
        mock_agent.instructions = []
        mock_agent.tools = []
        mock_agent.message_bus_name = "pubsub"
        mock_agent.state_store_name = "state"
        mock_agent.agents_registry_store_name = "registry"
        mock_agent.service_port = 8000

        with patch.object(exporter, "to_json", return_value='{"test": "json"}'):
            result = exporter.export_agent_to_json(mock_agent)
            assert result == '{"test": "json"}'


class TestLoaderExporterIntegration:
    """Integration tests for loader and exporter."""

    def test_roundtrip_agent_config(self) -> None:
        """Test roundtrip conversion of agent config."""
        loader = DaprAgentSpecLoader()
        exporter = DaprAgentSpecExporter()

        # Create original config
        original = DaprAgentConfig(
            name="roundtrip_agent",
            role="Tester",
            goal="Test roundtrip",
            instructions=["Step 1", "Step 2"],
            tools=["tool1", "tool2"],
            service_port=9999,
        )

        # Export to dict
        exported = exporter.to_dict(original)
        assert exported["name"] == "roundtrip_agent"

        # Load back
        loaded = loader.load_dict(exported)
        assert isinstance(loaded, DaprAgentConfig)
        assert loaded.name == original.name
        assert loaded.role == original.role

    def test_roundtrip_workflow_definition(self) -> None:
        """Test roundtrip conversion of workflow definition."""
        loader = DaprAgentSpecLoader()
        exporter = DaprAgentSpecExporter()

        # Create original workflow
        original = WorkflowDefinition(
            name="roundtrip_workflow",
            description="Test roundtrip workflow",
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

        # Export to dict
        exported = exporter.to_dict(original)
        assert exported["component_type"] == "Flow"
        assert exported["name"] == "roundtrip_workflow"

        # Load back
        loaded = loader.load_dict(exported)
        assert isinstance(loaded, WorkflowDefinition)
        assert loaded.name == original.name
        assert loaded.description == original.description

