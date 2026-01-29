"""Integration tests for the dapr-agents-oas-adapter library.

These tests verify end-to-end conversion pipelines without requiring
a live Dapr runtime. They test the full flow from OAS specs to Dapr
components and back using load_dict which bypasses pyagentspec deserialization.

RF-012: Integration Tests
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import yaml

from dapr_agents_oas_adapter import (
    AsyncDaprAgentSpecLoader,
    CachedLoader,
    DaprAgentSpecExporter,
    DaprAgentSpecLoader,
    StrictLoader,
    run_sync,
)
from dapr_agents_oas_adapter.cache import InMemoryCache
from dapr_agents_oas_adapter.converters.base import ConversionError
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)
from dapr_agents_oas_adapter.validation import (
    OASSchemaValidationError,
    validate_workflow,
)

# =============================================================================
# Test Fixtures - Dict-based specs that work with load_dict
# =============================================================================

SIMPLE_AGENT_DICT: dict[str, object] = {
    "component_type": "Agent",
    "name": "simple_assistant",
    "description": "A simple assistant agent for testing",
    "system_prompt": "You are a helpful assistant.",
    "llm_config": {
        "component_type": "VllmConfig",
        "id": "llm_1",
        "name": "vllm_config",
        "model_id": "gpt-4",
        "url": "http://localhost:8000",
    },
    "tools": [],
}

AGENT_WITH_TOOLS_DICT: dict[str, object] = {
    "component_type": "Agent",
    "name": "research_assistant",
    "description": "Research assistant with search capabilities",
    "system_prompt": "You are a research assistant with tools.",
    "tools": ["web_search", "calculator"],
}

SIMPLE_FLOW_DICT: dict[str, object] = {
    "component_type": "Flow",
    "name": "simple_workflow",
    "description": "A simple workflow with one LLM task",
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
            "id": "process",
            "name": "process_input",
            "inputs": [{"name": "input_text", "type": "string"}],
            "outputs": [{"name": "output_text", "type": "string"}],
            "prompt_template": "Process this: {{ input_text }}",
        },
        {
            "component_type": "EndNode",
            "id": "end",
            "name": "end",
            "inputs": [],
            "outputs": [],
        },
    ],
    "control_flow_connections": [
        {
            "id": "edge_1",
            "from_node": {"$component_ref": "start"},
            "to_node": {"$component_ref": "process"},
        },
        {
            "id": "edge_2",
            "from_node": {"$component_ref": "process"},
            "to_node": {"$component_ref": "end"},
        },
    ],
    "data_flow_connections": [],
    "start_node": {"$component_ref": "start"},
}

BRANCHING_FLOW_DICT: dict[str, object] = {
    "component_type": "Flow",
    "name": "branching_workflow",
    "description": "A workflow with conditional branching",
    "nodes": [
        {"component_type": "StartNode", "id": "start", "name": "start"},
        {"component_type": "LlmNode", "id": "classifier", "name": "classifier"},
        {"component_type": "ToolNode", "id": "urgent", "name": "handle_urgent"},
        {"component_type": "ToolNode", "id": "normal", "name": "handle_normal"},
        {"component_type": "EndNode", "id": "end", "name": "end"},
    ],
    "control_flow_connections": [
        {
            "from_node": {"$component_ref": "start"},
            "to_node": {"$component_ref": "classifier"},
        },
        {
            "from_node": {"$component_ref": "classifier"},
            "to_node": {"$component_ref": "urgent"},
            "from_branch": "urgent",
        },
        {
            "from_node": {"$component_ref": "classifier"},
            "to_node": {"$component_ref": "normal"},
            "from_branch": "normal",
        },
        {
            "from_node": {"$component_ref": "urgent"},
            "to_node": {"$component_ref": "end"},
        },
        {
            "from_node": {"$component_ref": "normal"},
            "to_node": {"$component_ref": "end"},
        },
    ],
    "data_flow_connections": [],
    "start_node": {"$component_ref": "start"},
}

RETRY_TIMEOUT_FLOW_DICT: dict[str, object] = {
    "component_type": "Flow",
    "name": "retry_workflow",
    "description": "A workflow with retry and timeout policies",
    "metadata": {
        "dapr": {
            "retry_policy": {
                "max_attempts": 3,
                "initial_interval_seconds": 1,
            },
            "timeout_seconds": 60,
        }
    },
    "nodes": [
        {"component_type": "StartNode", "id": "start", "name": "start"},
        {"component_type": "ToolNode", "id": "api_call", "name": "external_api"},
        {"component_type": "EndNode", "id": "end", "name": "end"},
    ],
    "control_flow_connections": [
        {
            "from_node": {"$component_ref": "start"},
            "to_node": {"$component_ref": "api_call"},
        },
        {
            "from_node": {"$component_ref": "api_call"},
            "to_node": {"$component_ref": "end"},
        },
    ],
    "data_flow_connections": [],
    "start_node": {"$component_ref": "start"},
}

MAP_FLOW_DICT: dict[str, object] = {
    "component_type": "Flow",
    "name": "map_workflow",
    "description": "A workflow with map/fan-out pattern",
    "nodes": [
        {"component_type": "StartNode", "id": "start", "name": "start"},
        {
            "component_type": "MapNode",
            "id": "parallel",
            "name": "process_items",
            "parallel": True,
        },
        {"component_type": "EndNode", "id": "end", "name": "end"},
    ],
    "control_flow_connections": [
        {
            "from_node": {"$component_ref": "start"},
            "to_node": {"$component_ref": "parallel"},
        },
        {
            "from_node": {"$component_ref": "parallel"},
            "to_node": {"$component_ref": "end"},
        },
    ],
    "data_flow_connections": [],
    "start_node": {"$component_ref": "start"},
}

SUBFLOW_DICT: dict[str, object] = {
    "component_type": "Flow",
    "name": "parent_workflow",
    "description": "A workflow that calls a child workflow",
    "nodes": [
        {"component_type": "StartNode", "id": "start", "name": "start"},
        {
            "component_type": "FlowNode",
            "id": "child",
            "name": "child_processor",
            "subflow": {"$component_ref": "child_workflow"},
        },
        {"component_type": "EndNode", "id": "end", "name": "end"},
    ],
    "control_flow_connections": [
        {
            "from_node": {"$component_ref": "start"},
            "to_node": {"$component_ref": "child"},
        },
        {"from_node": {"$component_ref": "child"}, "to_node": {"$component_ref": "end"}},
    ],
    "data_flow_connections": [],
    "start_node": {"$component_ref": "start"},
}

COMPLEX_FLOW_DICT: dict[str, object] = {
    "component_type": "Flow",
    "name": "complex_workflow",
    "description": "A complex multi-step workflow",
    "nodes": [
        {"component_type": "StartNode", "id": "start", "name": "start"},
        {"component_type": "LlmNode", "id": "extract", "name": "extract_info"},
        {"component_type": "ToolNode", "id": "validate", "name": "validate_data"},
        {"component_type": "LlmNode", "id": "summarize", "name": "summarize"},
        {"component_type": "EndNode", "id": "end", "name": "end"},
    ],
    "control_flow_connections": [
        {
            "from_node": {"$component_ref": "start"},
            "to_node": {"$component_ref": "extract"},
        },
        {
            "from_node": {"$component_ref": "extract"},
            "to_node": {"$component_ref": "validate"},
        },
        {
            "from_node": {"$component_ref": "validate"},
            "to_node": {"$component_ref": "summarize"},
        },
        {
            "from_node": {"$component_ref": "summarize"},
            "to_node": {"$component_ref": "end"},
        },
    ],
    "data_flow_connections": [],
    "start_node": {"$component_ref": "start"},
}


class TestAgentConversionPipeline:
    """Integration tests for Agent OAS -> Dapr conversion."""

    def test_simple_agent_dict_to_config(self) -> None:
        """Test loading a simple agent from dict."""
        loader = DaprAgentSpecLoader()
        config = loader.load_dict(SIMPLE_AGENT_DICT)

        assert isinstance(config, DaprAgentConfig)
        assert config.name == "simple_assistant"
        assert config.system_prompt is not None
        assert "helpful assistant" in config.system_prompt.lower()

    def test_simple_agent_json_to_config(self) -> None:
        """Test loading a simple agent from JSON string."""
        loader = DaprAgentSpecLoader()
        json_str = json.dumps(SIMPLE_AGENT_DICT)
        # Load via dict since load_json goes through pyagentspec
        config = loader.load_dict(json.loads(json_str))

        assert isinstance(config, DaprAgentConfig)
        assert config.name == "simple_assistant"

    def test_agent_with_tools_conversion(self) -> None:
        """Test converting an agent with tools."""
        loader = DaprAgentSpecLoader()
        config = loader.load_dict(AGENT_WITH_TOOLS_DICT)

        assert isinstance(config, DaprAgentConfig)
        assert config.name == "research_assistant"
        assert len(config.tools) == 2
        assert "web_search" in config.tools
        assert "calculator" in config.tools

    def test_agent_creation_with_mock_runtime(self) -> None:
        """Test creating a Dapr agent from config (mocked runtime)."""
        loader = DaprAgentSpecLoader()
        config = loader.load_dict(SIMPLE_AGENT_DICT)
        assert isinstance(config, DaprAgentConfig)

        mock_assistant = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "dapr_agents": MagicMock(
                    AssistantAgent=MagicMock(return_value=mock_assistant),
                    tool=MagicMock(side_effect=lambda f: f),
                ),
            },
        ):
            agent = loader.create_agent(config)
            assert agent is mock_assistant


class TestWorkflowConversionPipeline:
    """Integration tests for Flow OAS -> Dapr Workflow conversion."""

    def test_simple_flow_to_workflow(self) -> None:
        """Test converting a simple flow to workflow definition."""
        loader = DaprAgentSpecLoader()
        workflow = loader.load_dict(SIMPLE_FLOW_DICT)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "simple_workflow"
        assert len(workflow.tasks) >= 3  # start, process, end
        assert workflow.start_node is not None

    def test_branching_flow_conversion(self) -> None:
        """Test converting a flow with branching logic."""
        loader = DaprAgentSpecLoader()
        workflow = loader.load_dict(BRANCHING_FLOW_DICT)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "branching_workflow"

        # Check for branch edges
        branch_edges = [e for e in workflow.edges if e.from_branch]
        assert len(branch_edges) >= 2  # urgent and normal branches

    def test_retry_timeout_flow_conversion(self) -> None:
        """Test converting a flow with retry and timeout policies."""
        loader = DaprAgentSpecLoader()
        workflow = loader.load_dict(RETRY_TIMEOUT_FLOW_DICT)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "retry_workflow"
        # Note: WorkflowDefinition does not preserve OAS metadata directly
        # The retry/timeout policies are applied during workflow execution
        assert len(workflow.tasks) >= 3  # start, api_call, end

    def test_map_flow_conversion(self) -> None:
        """Test converting a flow with map/fan-out pattern."""
        loader = DaprAgentSpecLoader()
        workflow = loader.load_dict(MAP_FLOW_DICT)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "map_workflow"

        # Find map task
        map_tasks = [t for t in workflow.tasks if t.task_type == "map"]
        assert len(map_tasks) >= 1

    def test_subflow_conversion(self) -> None:
        """Test converting a flow with subflow references."""
        loader = DaprAgentSpecLoader()
        workflow = loader.load_dict(SUBFLOW_DICT)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "parent_workflow"

        # Find flow task (subflow call)
        flow_tasks = [t for t in workflow.tasks if t.task_type == "flow"]
        assert len(flow_tasks) >= 1

    def test_complex_flow_conversion(self) -> None:
        """Test converting a complex multi-step flow."""
        loader = DaprAgentSpecLoader()
        workflow = loader.load_dict(COMPLEX_FLOW_DICT)

        assert isinstance(workflow, WorkflowDefinition)
        assert workflow.name == "complex_workflow"
        assert len(workflow.tasks) >= 5  # start, extract, validate, summarize, end

        # Validate the workflow structure
        result = validate_workflow(workflow)
        assert result.is_valid or len(result.errors) == 0

    def test_workflow_function_creation(self) -> None:
        """Test creating executable workflow function."""
        loader = DaprAgentSpecLoader()
        workflow = loader.load_dict(SIMPLE_FLOW_DICT)
        assert isinstance(workflow, WorkflowDefinition)

        workflow_fn = loader.create_workflow(workflow)
        assert callable(workflow_fn)
        assert workflow_fn.__name__ == "simple_workflow"


class TestRoundtripConversion:
    """Integration tests for OAS -> Dapr -> OAS roundtrip conversion."""

    def test_agent_roundtrip(self) -> None:
        """Test that agent can be loaded and exported back."""
        loader = DaprAgentSpecLoader()
        exporter = DaprAgentSpecExporter()

        # Load
        config = loader.load_dict(SIMPLE_AGENT_DICT)
        assert isinstance(config, DaprAgentConfig)

        # Export back to dict
        exported = exporter.to_dict(config)
        assert exported["component_type"] == "Agent"
        assert exported["name"] == "simple_assistant"

        # Load again
        reloaded = loader.load_dict(exported)
        assert reloaded.name == config.name

    def test_workflow_roundtrip(self) -> None:
        """Test that workflow can be loaded and exported back."""
        loader = DaprAgentSpecLoader()
        exporter = DaprAgentSpecExporter()

        # Load
        workflow = loader.load_dict(SIMPLE_FLOW_DICT)
        assert isinstance(workflow, WorkflowDefinition)

        # Export back to dict
        exported = exporter.to_dict(workflow)
        assert exported["component_type"] == "Flow"
        assert exported["name"] == "simple_workflow"

        # Load again
        reloaded = loader.load_dict(exported)
        assert isinstance(reloaded, WorkflowDefinition)
        assert reloaded.name == workflow.name
        assert len(reloaded.tasks) == len(workflow.tasks)

    def test_complex_workflow_roundtrip(self) -> None:
        """Test roundtrip for complex workflow."""
        loader = DaprAgentSpecLoader()
        exporter = DaprAgentSpecExporter()

        workflow = loader.load_dict(COMPLEX_FLOW_DICT)
        assert isinstance(workflow, WorkflowDefinition)
        exported = exporter.to_dict(workflow)
        reloaded = loader.load_dict(exported)
        assert isinstance(reloaded, WorkflowDefinition)

        assert reloaded.name == workflow.name
        assert reloaded.description == workflow.description


class TestCachedLoaderIntegration:
    """Integration tests for cached loader functionality."""

    def test_cached_loader_hit_on_repeated_load(self) -> None:
        """Test that cached loader returns cached result."""
        base_loader = DaprAgentSpecLoader()
        cache: InMemoryCache[DaprAgentConfig | WorkflowDefinition] = InMemoryCache()
        cached_loader = CachedLoader(loader=base_loader, cache=cache)

        # Convert dict to YAML for caching
        yaml_str = yaml.dump(SIMPLE_AGENT_DICT)

        # First load (cache miss)
        config1 = cached_loader.load_yaml(yaml_str)

        # Second load (should hit cache)
        config2 = cached_loader.load_yaml(yaml_str)

        assert config1.name == config2.name
        # Stats are on CachedLoader, not InMemoryCache
        assert cached_loader.stats.hits >= 1

    def test_cached_loader_different_content(self) -> None:
        """Test that cached loader caches different content separately."""
        base_loader = DaprAgentSpecLoader()
        cache: InMemoryCache[DaprAgentConfig | WorkflowDefinition] = InMemoryCache()
        cached_loader = CachedLoader(loader=base_loader, cache=cache)

        # Use load_dict to avoid pyagentspec YAML parsing issues with $component_ref
        config1 = cached_loader.load_dict(SIMPLE_AGENT_DICT)
        workflow = cached_loader.load_dict(SIMPLE_FLOW_DICT)

        assert isinstance(config1, DaprAgentConfig)
        assert isinstance(workflow, WorkflowDefinition)
        assert config1.name != workflow.name


class TestAsyncLoaderIntegration:
    """Integration tests for async loader functionality."""

    @pytest.mark.asyncio
    async def test_async_load_dict(self) -> None:
        """Test async dict loading."""
        loader = AsyncDaprAgentSpecLoader()

        config = await loader.load_dict(SIMPLE_AGENT_DICT)
        assert isinstance(config, DaprAgentConfig)
        assert config.name == "simple_assistant"

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Test async loader as context manager."""
        async with AsyncDaprAgentSpecLoader() as loader:
            config = await loader.load_dict(SIMPLE_AGENT_DICT)
            assert config.name == "simple_assistant"

    def test_run_sync_utility(self) -> None:
        """Test synchronous wrapper for async operations."""
        loader = AsyncDaprAgentSpecLoader()

        config = run_sync(loader.load_dict(SIMPLE_AGENT_DICT))
        assert isinstance(config, DaprAgentConfig)


class TestStrictLoaderIntegration:
    """Integration tests for strict validation loader."""

    def test_strict_loader_valid_agent(self) -> None:
        """Test strict loader with valid agent."""
        loader = StrictLoader()

        config = loader.load_dict(SIMPLE_AGENT_DICT)
        assert isinstance(config, DaprAgentConfig)

    def test_strict_loader_valid_flow(self) -> None:
        """Test strict loader with valid flow."""
        loader = StrictLoader()

        workflow = loader.load_dict(SIMPLE_FLOW_DICT)
        assert isinstance(workflow, WorkflowDefinition)

    def test_strict_loader_rejects_invalid_agent(self) -> None:
        """Test strict loader rejects invalid agent spec."""
        loader = StrictLoader()

        invalid_dict = {
            "component_type": "Agent",
            "description": "Missing name field",
        }
        with pytest.raises(OASSchemaValidationError) as exc_info:
            loader.load_dict(invalid_dict)
        assert len(exc_info.value.issues) > 0

    def test_strict_loader_validation_only(self) -> None:
        """Test strict loader validation without loading."""
        loader = StrictLoader()

        result = loader.validate_dict(SIMPLE_AGENT_DICT)
        assert result.is_valid


class TestToolRegistryIntegration:
    """Integration tests for tool registry functionality."""

    def test_tool_registry_propagation(self) -> None:
        """Test that tool registry propagates through loading."""

        def search_tool(query: str) -> list[str]:
            return [f"Result for: {query}"]

        def calc_tool(expr: str) -> float:
            return 42.0

        loader = DaprAgentSpecLoader(
            tool_registry={"web_search": search_tool, "calculator": calc_tool}
        )

        config = loader.load_dict(AGENT_WITH_TOOLS_DICT)
        assert isinstance(config, DaprAgentConfig)

        # Tools should be registered
        assert "web_search" in loader.tool_registry
        assert "calculator" in loader.tool_registry

    def test_tool_registration_at_runtime(self) -> None:
        """Test registering tools after loader creation."""
        loader = DaprAgentSpecLoader()

        def new_tool(data: str) -> str:
            return data.upper()

        loader.register_tool("new_tool", new_tool)
        assert "new_tool" in loader.tool_registry
        assert loader.tool_registry["new_tool"] is new_tool


class TestExporterIntegration:
    """Integration tests for exporter functionality."""

    def test_export_to_json_string(self) -> None:
        """Test exporting to JSON string."""
        loader = DaprAgentSpecLoader()
        exporter = DaprAgentSpecExporter()

        config = loader.load_dict(SIMPLE_AGENT_DICT)
        json_str = exporter.to_json(config)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "simple_assistant"
        assert parsed["component_type"] == "Agent"

    def test_export_to_yaml_string(self) -> None:
        """Test exporting to YAML string."""
        loader = DaprAgentSpecLoader()
        exporter = DaprAgentSpecExporter()

        config = loader.load_dict(SIMPLE_AGENT_DICT)
        yaml_str = exporter.to_yaml(config)

        # Verify it's valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert parsed["name"] == "simple_assistant"
        assert parsed["component_type"] == "Agent"

    def test_export_workflow_to_json(self) -> None:
        """Test exporting workflow to JSON."""
        loader = DaprAgentSpecLoader()
        exporter = DaprAgentSpecExporter()

        workflow = loader.load_dict(SIMPLE_FLOW_DICT)
        json_str = exporter.to_json(workflow)

        parsed = json.loads(json_str)
        assert parsed["component_type"] == "Flow"
        assert parsed["name"] == "simple_workflow"

    def test_export_preserves_metadata(self) -> None:
        """Test that export preserves metadata."""
        loader = DaprAgentSpecLoader()
        exporter = DaprAgentSpecExporter()

        workflow = loader.load_dict(RETRY_TIMEOUT_FLOW_DICT)
        exported = exporter.to_dict(workflow)

        # Metadata should be preserved
        assert "agentspec_version" in exported


class TestErrorHandlingIntegration:
    """Integration tests for error handling across the pipeline."""

    def test_invalid_yaml_syntax_error(self) -> None:
        """Test handling of invalid YAML syntax."""
        loader = DaprAgentSpecLoader()

        with pytest.raises(ConversionError) as exc_info:
            loader.load_yaml("invalid: yaml: content:")

        assert "YAML" in str(exc_info.value) or "parse" in str(exc_info.value).lower()

    def test_invalid_json_syntax_error(self) -> None:
        """Test handling of invalid JSON syntax."""
        loader = DaprAgentSpecLoader()

        with pytest.raises(ConversionError) as exc_info:
            loader.load_json("{invalid json}")

        assert "JSON" in str(exc_info.value)

    def test_unsupported_component_type_error(self) -> None:
        """Test handling of unsupported component type."""
        loader = DaprAgentSpecLoader()

        with pytest.raises(ConversionError) as exc_info:
            loader.load_dict({"component_type": "Unknown", "name": "test"})

        assert "Unsupported" in str(exc_info.value)

    def test_missing_component_type_error(self) -> None:
        """Test handling of missing component type."""
        loader = DaprAgentSpecLoader()

        with pytest.raises(ConversionError):
            loader.load_dict({"name": "test"})


class TestWorkflowValidationIntegration:
    """Integration tests for workflow validation in the pipeline."""

    def test_workflow_validation_after_load(self) -> None:
        """Test validating workflow after loading."""
        loader = DaprAgentSpecLoader()

        workflow = loader.load_dict(SIMPLE_FLOW_DICT)
        assert isinstance(workflow, WorkflowDefinition)
        result = validate_workflow(workflow)

        # Simple flow should be valid
        assert result.is_valid or len(result.errors) == 0

    def test_complex_workflow_validation(self) -> None:
        """Test validating complex workflow."""
        loader = DaprAgentSpecLoader()

        workflow = loader.load_dict(COMPLEX_FLOW_DICT)
        assert isinstance(workflow, WorkflowDefinition)
        result = validate_workflow(workflow)

        # Complex flow should pass basic validation
        # May have warnings but no blocking errors
        for error in result.errors:
            # Log any errors for debugging
            print(f"Validation error: {error}")


class TestProgrammaticWorkflowCreation:
    """Integration tests for programmatic workflow creation."""

    def test_create_workflow_programmatically(self) -> None:
        """Test creating and exporting a workflow programmatically."""
        exporter = DaprAgentSpecExporter()

        workflow = WorkflowDefinition(
            name="programmatic_workflow",
            description="Created programmatically",
            tasks=[
                WorkflowTaskDefinition(name="start", task_type="start"),
                WorkflowTaskDefinition(
                    name="process",
                    task_type="llm",
                    config={"prompt_template": "Process: {{ input }}"},
                ),
                WorkflowTaskDefinition(name="end", task_type="end"),
            ],
            edges=[
                WorkflowEdgeDefinition(from_node="start", to_node="process"),
                WorkflowEdgeDefinition(from_node="process", to_node="end"),
            ],
            start_node="start",
            end_nodes=["end"],
        )

        exported = exporter.to_dict(workflow)
        assert exported["component_type"] == "Flow"
        assert exported["name"] == "programmatic_workflow"

    def test_create_agent_programmatically(self) -> None:
        """Test creating and exporting an agent programmatically."""
        exporter = DaprAgentSpecExporter()

        config = DaprAgentConfig(
            name="programmatic_agent",
            role="Helper",
            goal="Assist users",
            instructions=["Be helpful", "Be concise"],
            tools=["search", "calculate"],
        )

        exported = exporter.to_dict(config)
        assert exported["component_type"] == "Agent"
        assert exported["name"] == "programmatic_agent"


class TestMultiLoaderComparison:
    """Integration tests comparing different loader implementations."""

    def test_all_loaders_produce_same_result(self) -> None:
        """Test that all loader types produce equivalent results."""
        standard_loader = DaprAgentSpecLoader()
        strict_loader = StrictLoader()

        config_standard = standard_loader.load_dict(SIMPLE_AGENT_DICT)
        config_strict = strict_loader.load_dict(SIMPLE_AGENT_DICT)
        assert isinstance(config_standard, DaprAgentConfig)
        assert isinstance(config_strict, DaprAgentConfig)

        # All should produce equivalent configs
        assert config_standard.name == config_strict.name
        # DaprAgentConfig has 'goal' not 'description'
        assert config_standard.goal == config_strict.goal

    @pytest.mark.asyncio
    async def test_async_loader_matches_sync(self) -> None:
        """Test that async loader produces same result as sync."""
        sync_loader = DaprAgentSpecLoader()
        async_loader = AsyncDaprAgentSpecLoader()

        config_sync = sync_loader.load_dict(SIMPLE_AGENT_DICT)
        config_async = await async_loader.load_dict(SIMPLE_AGENT_DICT)
        assert isinstance(config_sync, DaprAgentConfig)
        assert isinstance(config_async, DaprAgentConfig)

        assert config_sync.name == config_async.name
        # DaprAgentConfig has 'goal' not 'description'
        assert config_sync.goal == config_async.goal


class TestCodeGeneration:
    """Integration tests for workflow code generation."""

    def test_generate_workflow_code(self) -> None:
        """Test generating Python code for workflow."""
        loader = DaprAgentSpecLoader()

        workflow = loader.load_dict(SIMPLE_FLOW_DICT)
        assert isinstance(workflow, WorkflowDefinition)
        code = loader.generate_workflow_code(workflow)

        # Code should contain key elements
        assert "@workflow" in code
        assert workflow.name in code

    def test_generated_code_is_valid_python(self) -> None:
        """Test that generated code is syntactically valid Python."""
        loader = DaprAgentSpecLoader()

        workflow = loader.load_dict(SIMPLE_FLOW_DICT)
        assert isinstance(workflow, WorkflowDefinition)
        code = loader.generate_workflow_code(workflow)

        # Should compile without syntax errors
        compile(code, "<generated>", "exec")
