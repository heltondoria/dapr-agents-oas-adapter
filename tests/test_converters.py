"""Tests for converter modules."""


from dapr_agents_oas_adapter.converters.agent import AgentConverter
from dapr_agents_oas_adapter.converters.base import (
    ConversionError,
    ConverterRegistry,
    ValidationError,
)
from dapr_agents_oas_adapter.converters.flow import FlowConverter
from dapr_agents_oas_adapter.converters.llm import LlmConfigConverter
from dapr_agents_oas_adapter.converters.node import NodeConverter
from dapr_agents_oas_adapter.converters.tool import ToolConverter
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    LlmClientConfig,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)


class TestLlmConfigConverter:
    """Tests for LlmConfigConverter."""

    def test_can_convert_llm_client_config(self) -> None:
        """Test can_convert with LlmClientConfig."""
        converter = LlmConfigConverter()
        config = LlmClientConfig(provider="openai", model_id="gpt-4")
        assert converter.can_convert(config) is True

    def test_can_convert_dict(self) -> None:
        """Test can_convert with dictionary."""
        converter = LlmConfigConverter()
        config_dict = {"component_type": "VllmConfig", "model_id": "llama-3"}
        assert converter.can_convert(config_dict) is True

    def test_from_dict(self) -> None:
        """Test from_dict conversion."""
        converter = LlmConfigConverter()
        config_dict = {
            "component_type": "OpenAIConfig",
            "model_id": "gpt-4",
            "temperature": 0.5,
            "max_tokens": 1000,
        }
        result = converter.from_dict(config_dict)
        assert result.provider == "openai"
        assert result.model_id == "gpt-4"
        assert result.temperature == 0.5
        assert result.max_tokens == 1000

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        converter = LlmConfigConverter()
        config = LlmClientConfig(
            provider="vllm",
            model_id="llama-3",
            url="http://localhost:8000",
            temperature=0.8,
        )
        result = converter.to_dict(config)
        assert result["component_type"] == "VllmConfig"
        assert result["model_id"] == "llama-3"
        assert result["url"] == "http://localhost:8000"


class TestToolConverter:
    """Tests for ToolConverter."""

    def test_from_callable(self) -> None:
        """Test creating ToolDefinition from callable."""
        def search_web(query: str) -> list[str]:
            """Search the web for information."""
            return ["result1", "result2"]

        converter = ToolConverter()
        tool_def = converter.from_callable(search_web)

        assert tool_def.name == "search_web"
        assert "Search the web" in tool_def.description
        assert len(tool_def.inputs) == 1
        assert tool_def.inputs[0]["title"] == "query"
        assert tool_def.implementation is search_web

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        converter = ToolConverter()
        tool_def = ToolDefinition(
            name="calculator",
            description="Perform calculations",
            inputs=[{"title": "expression", "type": "string"}],
            outputs=[{"title": "result", "type": "number"}],
        )
        result = converter.to_dict(tool_def)
        assert result["component_type"] == "ServerTool"
        assert result["name"] == "calculator"
        assert result["description"] == "Perform calculations"

    def test_from_dict(self) -> None:
        """Test from_dict conversion."""
        converter = ToolConverter()
        tool_dict = {
            "component_type": "ServerTool",
            "name": "weather",
            "description": "Get weather info",
            "inputs": [{"title": "city", "type": "string"}],
            "outputs": [{"title": "forecast", "type": "string"}],
        }
        result = converter.from_dict(tool_dict)
        assert result.name == "weather"
        assert result.description == "Get weather info"
        assert len(result.inputs) == 1

    def test_with_tool_registry(self) -> None:
        """Test tool converter with tool registry."""
        def my_tool() -> str:
            return "result"

        converter = ToolConverter(tool_registry={"my_tool": my_tool})
        tool_dict = {"name": "my_tool", "description": "My tool"}
        result = converter.from_dict(tool_dict)
        assert result.implementation is my_tool


class TestAgentConverter:
    """Tests for AgentConverter."""

    def test_from_dict(self) -> None:
        """Test from_dict conversion."""
        converter = AgentConverter()
        agent_dict = {
            "component_type": "Agent",
            "name": "assistant",
            "description": "A helpful assistant",
            "system_prompt": "You are helpful. Always be polite.",
            "llm_config": {"component_type": "OpenAIConfig", "model_id": "gpt-4"},
            "tools": [],
        }
        result = converter.from_dict(agent_dict)
        assert result.name == "assistant"
        assert result.goal == "A helpful assistant"
        assert len(result.instructions) > 0

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        converter = AgentConverter()
        config = DaprAgentConfig(
            name="test_agent",
            role="Helper",
            goal="Help users",
            instructions=["Be helpful"],
            tools=["search"],
        )
        result = converter.to_dict(config)
        assert result["component_type"] == "Agent"
        assert result["name"] == "test_agent"

    def test_can_convert(self) -> None:
        """Test can_convert method."""
        converter = AgentConverter()
        config = DaprAgentConfig(name="test")
        assert converter.can_convert(config) is True
        assert converter.can_convert({"component_type": "Agent"}) is True
        assert converter.can_convert({"component_type": "Flow"}) is False


class TestNodeConverter:
    """Tests for NodeConverter."""

    def test_from_dict_llm_node(self) -> None:
        """Test from_dict for LLM node."""
        converter = NodeConverter()
        node_dict = {
            "component_type": "LlmNode",
            "name": "generate_response",
            "prompt_template": "Answer: {{question}}",
            "inputs": [{"title": "question", "type": "string"}],
            "outputs": [{"title": "answer", "type": "string"}],
        }
        result = converter.from_dict(node_dict)
        assert result.name == "generate_response"
        assert result.task_type == "llm"
        assert result.config.get("prompt_template") == "Answer: {{question}}"

    def test_from_dict_tool_node(self) -> None:
        """Test from_dict for tool node."""
        converter = NodeConverter()
        node_dict = {
            "component_type": "ToolNode",
            "name": "call_api",
            "inputs": [{"title": "endpoint", "type": "string"}],
            "outputs": [{"title": "response", "type": "object"}],
        }
        result = converter.from_dict(node_dict)
        assert result.name == "call_api"
        assert result.task_type == "tool"

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        converter = NodeConverter()
        task = WorkflowTaskDefinition(
            name="process",
            task_type="llm",
            config={"prompt_template": "Process {{data}}"},
            inputs=["data"],
            outputs=["result"],
        )
        result = converter.to_dict(task)
        assert result["component_type"] == "LlmNode"
        assert result["name"] == "process"


class TestFlowConverter:
    """Tests for FlowConverter."""

    def test_from_dict(self) -> None:
        """Test from_dict conversion."""
        converter = FlowConverter()
        flow_dict = {
            "component_type": "Flow",
            "name": "my_workflow",
            "description": "A test workflow",
            "nodes": [
                {
                    "component_type": "StartNode",
                    "id": "start_1",
                    "name": "start",
                    "inputs": [{"title": "input", "type": "string"}],
                    "outputs": [{"title": "input", "type": "string"}],
                },
                {
                    "component_type": "EndNode",
                    "id": "end_1",
                    "name": "end",
                    "inputs": [{"title": "result", "type": "string"}],
                    "outputs": [{"title": "result", "type": "string"}],
                },
            ],
            "control_flow_connections": [
                {
                    "from_node": {"$component_ref": "start_1"},
                    "to_node": {"$component_ref": "end_1"},
                }
            ],
            "data_flow_connections": [],
            "start_node": {"$component_ref": "start_1"},
            "inputs": [],
            "outputs": [],
        }
        result = converter.from_dict(flow_dict)
        assert result.name == "my_workflow"
        assert result.description == "A test workflow"
        assert len(result.tasks) == 2

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        converter = FlowConverter()
        workflow = WorkflowDefinition(
            name="test_flow",
            description="Test flow",
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
        result = converter.to_dict(workflow)
        assert result["component_type"] == "Flow"
        assert result["name"] == "test_flow"
        assert "$referenced_components" in result

    def test_generate_workflow_code(self) -> None:
        """Test workflow code generation."""
        converter = FlowConverter()
        workflow = WorkflowDefinition(
            name="sample_workflow",
            description="Sample workflow for testing",
            tasks=[
                WorkflowTaskDefinition(name="start", task_type="start"),
                WorkflowTaskDefinition(
                    name="process_data",
                    task_type="llm",
                    config={"prompt_template": "Process: {{input}}"},
                ),
                WorkflowTaskDefinition(name="end", task_type="end"),
            ],
            edges=[
                WorkflowEdgeDefinition(from_node="start", to_node="process_data"),
                WorkflowEdgeDefinition(from_node="process_data", to_node="end"),
            ],
            start_node="start",
            end_nodes=["end"],
        )
        code = converter.generate_workflow_code(workflow)
        assert "from dapr_agents.workflow import" in code
        assert "@workflow(name='sample_workflow')" in code
        assert "def sample_workflow_workflow" in code


class TestConverterRegistry:
    """Tests for ConverterRegistry."""

    def test_register_and_get_converter(self) -> None:
        """Test registering and retrieving converters."""
        registry = ConverterRegistry()
        llm_converter = LlmConfigConverter()
        tool_converter = ToolConverter()

        registry.register(llm_converter)
        registry.register(tool_converter)

        config = LlmClientConfig(provider="openai", model_id="gpt-4")
        found = registry.get_converter(config)
        assert found is llm_converter

    def test_no_converter_found(self) -> None:
        """Test when no converter is found."""
        registry = ConverterRegistry()
        result = registry.get_converter("unknown")
        assert result is None


class TestExceptions:
    """Tests for custom exceptions."""

    def test_conversion_error(self) -> None:
        """Test ConversionError exception."""
        error = ConversionError("Failed to convert", component={"type": "test"})
        assert "Failed to convert" in str(error)
        assert error.component == {"type": "test"}

    def test_validation_error(self) -> None:
        """Test ValidationError exception."""
        error = ValidationError("Invalid field", field="name")
        assert "Invalid field" in str(error)
        assert error.field == "name"

