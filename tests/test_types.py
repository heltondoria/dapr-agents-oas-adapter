"""Tests for types module."""

from dapr_agents_oas_adapter.types import (
    DAPR_PROVIDER_TO_OAS_LLM,
    DAPR_TO_OAS_AGENT_TYPE,
    JSON_SCHEMA_TO_PYTHON,
    OAS_LLM_TO_DAPR_PROVIDER,
    OAS_TO_DAPR_AGENT_TYPE,
    PYTHON_TO_JSON_SCHEMA,
    DaprAgentConfig,
    DaprAgentType,
    LlmClientConfig,
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


class TestLlmClientConfig:
    """Tests for LlmClientConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values for LLM client config."""
        config = LlmClientConfig(
            provider="openai",
            model_id="gpt-4",
        )
        assert config.provider == "openai"
        assert config.model_id == "gpt-4"
        assert config.url is None
        assert config.api_key is None
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.extra_params == {}

    def test_all_values(self) -> None:
        """Test LLM client config with all values."""
        config = LlmClientConfig(
            provider="vllm",
            model_id="llama-3",
            url="http://localhost:8000",
            api_key="secret",
            temperature=0.5,
            max_tokens=1000,
            extra_params={"top_p": 0.9},
        )
        assert config.provider == "vllm"
        assert config.model_id == "llama-3"
        assert config.url == "http://localhost:8000"
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
