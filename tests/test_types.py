"""Tests for types module."""

import pytest
from pydantic import ValidationError

from dapr_agents_oas_adapter.types import (
    DAPR_PROVIDER_TO_OAS_LLM,
    DAPR_TO_OAS_AGENT_TYPE,
    JSON_SCHEMA_TO_PYTHON,
    OAS_LLM_TO_DAPR_PROVIDER,
    OAS_TO_DAPR_AGENT_TYPE,
    PYTHON_TO_JSON_SCHEMA,
    DaprAgentConfig,
    DaprAgentType,
    LlmProviderConfig,
    OASComponentType,
    OrchestratorType,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)


class TestEnums:
    """Tests for enumeration types."""

    def test_oas_component_type_values(self) -> None:
        """Test OAS component type enum values."""
        assert OASComponentType.AGENT.value == "Agent"
        assert OASComponentType.FLOW.value == "Flow"
        assert OASComponentType.LLM_NODE.value == "LlmNode"
        assert OASComponentType.SERVER_TOOL.value == "ServerTool"

    def test_dapr_agent_type_values(self) -> None:
        """Test Dapr agent type enum values."""
        assert DaprAgentType.ASSISTANT_AGENT.value == "AssistantAgent"
        assert DaprAgentType.REACT_AGENT.value == "ReActAgent"
        assert DaprAgentType.DURABLE_AGENT.value == "DurableAgent"

    def test_orchestrator_type_values(self) -> None:
        """Test orchestrator type enum values."""
        assert OrchestratorType.LLM.value == "LLMOrchestrator"
        assert OrchestratorType.RANDOM.value == "RandomOrchestrator"
        assert OrchestratorType.ROUND_ROBIN.value == "RoundRobinOrchestrator"


class TestLlmProviderConfig:
    """Tests for LlmProviderConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values for LLM client config."""
        config = LlmProviderConfig(
            provider="openai",
            model_name="gpt-4",
        )
        assert config.provider == "openai"
        assert config.model_name == "gpt-4"
        assert config.base_url is None
        assert config.api_key is None
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.extra_params == {}

    def test_all_values(self) -> None:
        """Test LLM client config with all values."""
        config = LlmProviderConfig(
            provider="vllm",
            model_name="llama-3",
            base_url="http://localhost:8000",
            api_key="secret",
            temperature=0.5,
            max_tokens=1000,
            extra_params={"top_p": 0.9},
        )
        assert config.provider == "vllm"
        assert config.model_name == "llama-3"
        assert config.base_url == "http://localhost:8000"
        assert config.api_key == "secret"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.extra_params == {"top_p": 0.9}


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_tool_definition(self) -> None:
        """Test tool definition creation."""
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            inputs=[{"title": "query", "type": "string"}],
            outputs=[{"title": "results", "type": "array"}],
        )
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert len(tool.inputs) == 1
        assert len(tool.outputs) == 1
        assert tool.implementation is None

    def test_tool_with_implementation(self) -> None:
        """Test tool definition with implementation."""

        def search_func(query: str) -> list[str]:
            return [query]

        tool = ToolDefinition(
            name="search",
            description="Search",
            inputs=[],
            outputs=[],
            implementation=search_func,
        )
        assert tool.implementation is search_func


class TestWorkflowTaskDefinition:
    """Tests for WorkflowTaskDefinition dataclass."""

    def test_task_definition(self) -> None:
        """Test workflow task definition."""
        task = WorkflowTaskDefinition(
            name="process_data",
            task_type="llm",
            config={"prompt_template": "Process: {{input}}"},
            inputs=["input"],
            outputs=["result"],
        )
        assert task.name == "process_data"
        assert task.task_type == "llm"
        assert "prompt_template" in task.config
        assert task.inputs == ["input"]
        assert task.outputs == ["result"]


class TestWorkflowDefinition:
    """Tests for WorkflowDefinition dataclass."""

    def test_workflow_definition(self) -> None:
        """Test workflow definition."""
        workflow = WorkflowDefinition(
            name="my_workflow",
            description="Test workflow",
            tasks=[
                WorkflowTaskDefinition(name="start", task_type="start"),
                WorkflowTaskDefinition(name="end", task_type="end"),
            ],
            edges=[
                WorkflowEdgeDefinition(from_node="start", to_node="end"),
            ],
            start_node="start",
            end_nodes=["end"],
        )
        assert workflow.name == "my_workflow"
        assert len(workflow.tasks) == 2
        assert len(workflow.edges) == 1
        assert workflow.start_node == "start"
        assert workflow.end_nodes == ["end"]


class TestDaprAgentConfig:
    """Tests for DaprAgentConfig model."""

    def test_default_values(self) -> None:
        """Test default values for Dapr agent config."""
        config = DaprAgentConfig(name="test_agent")
        assert config.name == "test_agent"
        assert config.role is None
        assert config.goal is None
        assert config.instructions == []
        assert config.system_prompt is None
        assert config.tools == []
        assert config.message_bus_name == "messagepubsub"
        assert config.state_store_name == "statestore"
        assert config.agents_registry_store_name == "agentsregistry"
        assert config.service_port == 8000

    def test_full_config(self) -> None:
        """Test Dapr agent config with all values."""
        config = DaprAgentConfig(
            name="assistant",
            role="Helper",
            goal="Help users",
            instructions=["Be helpful", "Be concise"],
            system_prompt="You are a helpful assistant.",
            tools=["search", "calculator"],
            message_bus_name="kafka",
            state_store_name="redis",
            agents_registry_store_name="postgres",
            service_port=9000,
        )
        assert config.name == "assistant"
        assert config.role == "Helper"
        assert config.goal == "Help users"
        assert len(config.instructions) == 2
        assert len(config.tools) == 2
        assert config.service_port == 9000


class TestMappings:
    """Tests for type mappings."""

    def test_oas_to_dapr_agent_type(self) -> None:
        """Test OAS to Dapr agent type mapping."""
        assert OAS_TO_DAPR_AGENT_TYPE["Agent"] == DaprAgentType.ASSISTANT_AGENT
        assert OAS_TO_DAPR_AGENT_TYPE["ReActAgent"] == DaprAgentType.REACT_AGENT

    def test_dapr_to_oas_agent_type(self) -> None:
        """Test Dapr to OAS agent type mapping."""
        assert DAPR_TO_OAS_AGENT_TYPE[DaprAgentType.ASSISTANT_AGENT] == "Agent"
        assert DAPR_TO_OAS_AGENT_TYPE[DaprAgentType.REACT_AGENT] == "Agent"

    def test_llm_provider_mappings(self) -> None:
        """Test LLM provider mappings."""
        assert OAS_LLM_TO_DAPR_PROVIDER["VllmConfig"] == "vllm"
        assert OAS_LLM_TO_DAPR_PROVIDER["OpenAIConfig"] == "openai"
        assert DAPR_PROVIDER_TO_OAS_LLM["vllm"] == "VllmConfig"
        assert DAPR_PROVIDER_TO_OAS_LLM["openai"] == "OpenAIConfig"

    def test_json_schema_type_mappings(self) -> None:
        """Test JSON Schema type mappings."""
        assert JSON_SCHEMA_TO_PYTHON["string"] is str
        assert JSON_SCHEMA_TO_PYTHON["integer"] is int
        assert JSON_SCHEMA_TO_PYTHON["number"] is float
        assert JSON_SCHEMA_TO_PYTHON["boolean"] is bool
        assert PYTHON_TO_JSON_SCHEMA[str] == "string"
        assert PYTHON_TO_JSON_SCHEMA[int] == "integer"


class TestPydanticFeatures:
    """Tests for Pydantic-specific features (RF-002 acceptance criteria)."""

    def test_llm_client_config_model_json_schema(self) -> None:
        """Verify LlmProviderConfig generates valid JSON schema."""
        schema = LlmProviderConfig.model_json_schema()
        assert schema["type"] == "object"
        assert "provider" in schema["properties"]
        assert "model_name" in schema["properties"]
        assert "provider" in schema["required"]
        assert "model_name" in schema["required"]

    def test_llm_client_config_rejects_extra_fields(self) -> None:
        """Verify LlmProviderConfig rejects unknown fields."""
        with pytest.raises(ValidationError) as exc_info:
            LlmProviderConfig(
                provider="openai",
                model_name="gpt-4",
                unknown_field="should_fail",  # type: ignore[call-arg]
            )
        assert "extra" in str(exc_info.value).lower()

    def test_llm_client_config_requires_mandatory_fields(self) -> None:
        """Verify LlmProviderConfig enforces required fields."""
        with pytest.raises(ValidationError) as exc_info:
            LlmProviderConfig()  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "provider" in field_names
        assert "model_name" in field_names

    def test_tool_definition_model_json_schema(self) -> None:
        """Verify ToolDefinition generates valid JSON schema."""
        schema = ToolDefinition.model_json_schema()
        # Schema may have $defs for complex types
        props = schema.get("properties", {})
        required = schema.get("required", [])
        assert "name" in props
        assert "description" in props
        assert "name" in required
        assert "description" in required

    def test_tool_definition_rejects_extra_fields(self) -> None:
        """Verify ToolDefinition rejects unknown fields."""
        with pytest.raises(ValidationError) as exc_info:
            ToolDefinition(
                name="test",
                description="test",
                extra_field="should_fail",  # type: ignore[call-arg]
            )
        assert "extra" in str(exc_info.value).lower()

    def test_workflow_task_definition_model_json_schema(self) -> None:
        """Verify WorkflowTaskDefinition generates valid JSON schema."""
        schema = WorkflowTaskDefinition.model_json_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "task_type" in schema["properties"]
        assert "name" in schema["required"]
        assert "task_type" in schema["required"]

    def test_workflow_edge_definition_model_json_schema(self) -> None:
        """Verify WorkflowEdgeDefinition generates valid JSON schema."""
        schema = WorkflowEdgeDefinition.model_json_schema()
        assert schema["type"] == "object"
        assert "from_node" in schema["properties"]
        assert "to_node" in schema["properties"]
        assert "from_node" in schema["required"]
        assert "to_node" in schema["required"]

    def test_workflow_definition_model_json_schema(self) -> None:
        """Verify WorkflowDefinition generates valid JSON schema."""
        schema = WorkflowDefinition.model_json_schema()
        # Schema uses $defs for recursive types; look in the right place
        if "$defs" in schema and "WorkflowDefinition" in schema["$defs"]:
            props = schema["$defs"]["WorkflowDefinition"].get("properties", {})
            required = schema["$defs"]["WorkflowDefinition"].get("required", [])
        else:
            props = schema.get("properties", {})
            required = schema.get("required", [])
        assert "name" in props
        assert "tasks" in props
        assert "edges" in props
        assert "name" in required

    def test_workflow_definition_nested_validation(self) -> None:
        """Verify WorkflowDefinition validates nested models."""
        # Valid nested structure
        workflow = WorkflowDefinition(
            name="test",
            tasks=[WorkflowTaskDefinition(name="t1", task_type="llm")],
            edges=[WorkflowEdgeDefinition(from_node="t1", to_node="t2")],
        )
        assert len(workflow.tasks) == 1
        assert len(workflow.edges) == 1

    def test_dapr_agent_config_model_json_schema(self) -> None:
        """Verify DaprAgentConfig generates valid JSON schema."""
        schema = DaprAgentConfig.model_json_schema()
        assert schema["type"] == "object"
        assert "name" in schema["properties"]
        assert "role" in schema["properties"]
        assert "instructions" in schema["properties"]
        assert "name" in schema["required"]

    def test_dapr_agent_config_rejects_extra_fields(self) -> None:
        """Verify DaprAgentConfig rejects unknown fields."""
        with pytest.raises(ValidationError) as exc_info:
            DaprAgentConfig(
                name="test",
                unknown_field="should_fail",  # type: ignore[call-arg]
            )
        assert "extra" in str(exc_info.value).lower()

    def test_model_serialization_roundtrip(self) -> None:
        """Verify models can be serialized and deserialized."""
        original = WorkflowDefinition(
            name="roundtrip_test",
            description="Test roundtrip serialization",
            tasks=[
                WorkflowTaskDefinition(
                    name="task1",
                    task_type="llm",
                    config={"prompt": "Hello"},
                )
            ],
            edges=[
                WorkflowEdgeDefinition(
                    from_node="start",
                    to_node="task1",
                    from_branch="yes",
                )
            ],
            start_node="start",
            end_nodes=["end"],
        )
        # Serialize to dict
        data = original.model_dump()
        # Deserialize back
        restored = WorkflowDefinition.model_validate(data)

        assert restored.name == original.name
        assert restored.description == original.description
        assert len(restored.tasks) == len(original.tasks)
        assert restored.tasks[0].name == original.tasks[0].name
        assert len(restored.edges) == len(original.edges)
        assert restored.edges[0].from_branch == original.edges[0].from_branch

    def test_model_json_roundtrip(self) -> None:
        """Verify models can be serialized to JSON and back."""
        original = LlmProviderConfig(
            provider="openai",
            model_name="gpt-4",
            temperature=0.5,
            extra_params={"top_p": 0.9},
        )
        # Serialize to JSON
        json_str = original.model_dump_json()
        # Deserialize back
        restored = LlmProviderConfig.model_validate_json(json_str)

        assert restored.provider == original.provider
        assert restored.model_name == original.model_name
        assert restored.temperature == original.temperature
        assert restored.extra_params == original.extra_params


class TestFrozenModels:
    """Tests for frozen model immutability."""

    def test_llm_client_config_frozen_raises_validation_error(self) -> None:
        """Verify assigning to a field on a frozen LlmProviderConfig raises ValidationError."""
        config = LlmProviderConfig(provider="openai", model_name="gpt-4")
        with pytest.raises(ValidationError):
            config.provider = "ollama"

    def test_dapr_agent_config_frozen_raises_validation_error(self) -> None:
        """Verify assigning to a field on a frozen DaprAgentConfig raises ValidationError."""
        config = DaprAgentConfig(name="test_agent")
        with pytest.raises(ValidationError):
            config.name = "new_name"

    def test_tool_definition_default_tool_type(self) -> None:
        """Verify ToolDefinition has default tool_type of 'function'."""
        tool = ToolDefinition(name="test", description="test tool")
        assert tool.tool_type == "function"
