"""Pytest configuration and fixtures."""

from typing import Any, Callable

import pytest

from dapr_agents_oas_adapter.exporter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    LlmClientConfig,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)


@pytest.fixture
def sample_tool_registry() -> dict[str, Callable[..., Any]]:
    """Provide a sample tool registry for testing."""
    def search_tool(query: str) -> list[str]:
        """Search the web."""
        return [f"Result for: {query}"]

    def calculator_tool(expression: str) -> float:
        """Calculate an expression."""
        return eval(expression)

    return {
        "search": search_tool,
        "calculator": calculator_tool,
    }


@pytest.fixture
def loader(sample_tool_registry: dict[str, Callable[..., Any]]) -> DaprAgentSpecLoader:
    """Provide a configured loader instance."""
    return DaprAgentSpecLoader(tool_registry=sample_tool_registry)


@pytest.fixture
def exporter() -> DaprAgentSpecExporter:
    """Provide an exporter instance."""
    return DaprAgentSpecExporter()


@pytest.fixture
def sample_agent_config() -> DaprAgentConfig:
    """Provide a sample agent configuration."""
    return DaprAgentConfig(
        name="test_assistant",
        role="Test Helper",
        goal="Help with testing",
        instructions=["Be helpful", "Be accurate", "Be concise"],
        system_prompt="You are a helpful test assistant.",
        tools=["search", "calculator"],
        message_bus_name="testpubsub",
        state_store_name="teststate",
        agents_registry_store_name="testagents",
        service_port=8080,
    )


@pytest.fixture
def sample_workflow_definition() -> WorkflowDefinition:
    """Provide a sample workflow definition."""
    return WorkflowDefinition(
        name="test_workflow",
        description="A test workflow for unit tests",
        tasks=[
            WorkflowTaskDefinition(
                name="start",
                task_type="start",
                inputs=["user_input"],
                outputs=["user_input"],
            ),
            WorkflowTaskDefinition(
                name="analyze",
                task_type="llm",
                config={"prompt_template": "Analyze: {{user_input}}"},
                inputs=["user_input"],
                outputs=["analysis"],
            ),
            WorkflowTaskDefinition(
                name="process",
                task_type="tool",
                config={"tool_name": "calculator"},
                inputs=["analysis"],
                outputs=["result"],
            ),
            WorkflowTaskDefinition(
                name="end",
                task_type="end",
                inputs=["result"],
                outputs=["result"],
            ),
        ],
        edges=[
            WorkflowEdgeDefinition(
                from_node="start",
                to_node="analyze",
                data_mapping={"user_input": "user_input"},
            ),
            WorkflowEdgeDefinition(
                from_node="analyze",
                to_node="process",
                data_mapping={"analysis": "analysis"},
            ),
            WorkflowEdgeDefinition(
                from_node="process",
                to_node="end",
                data_mapping={"result": "result"},
            ),
        ],
        start_node="start",
        end_nodes=["end"],
        inputs=[{"title": "user_input", "type": "string"}],
        outputs=[{"title": "result", "type": "string"}],
    )


@pytest.fixture
def sample_llm_config() -> LlmClientConfig:
    """Provide a sample LLM configuration."""
    return LlmClientConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="test-api-key",
        temperature=0.7,
        max_tokens=1000,
        extra_params={"top_p": 0.9},
    )


@pytest.fixture
def sample_tool_definition() -> ToolDefinition:
    """Provide a sample tool definition."""
    return ToolDefinition(
        name="weather_tool",
        description="Get weather information for a city",
        inputs=[
            {"title": "city", "type": "string", "description": "City name"},
        ],
        outputs=[
            {"title": "temperature", "type": "number"},
            {"title": "conditions", "type": "string"},
        ],
    )


@pytest.fixture
def sample_agent_json() -> str:
    """Provide sample agent JSON for testing."""
    return """{
        "component_type": "Agent",
        "id": "agent_123",
        "name": "JSON Test Agent",
        "description": "An agent loaded from JSON",
        "llm_config": {
            "component_type": "OpenAIConfig",
            "id": "llm_456",
            "name": "gpt4_config",
            "model_id": "gpt-4"
        },
        "system_prompt": "You are a helpful assistant.",
        "tools": [],
        "inputs": [],
        "outputs": [],
        "agentspec_version": "25.4.1"
    }"""


@pytest.fixture
def sample_flow_json() -> str:
    """Provide sample flow JSON for testing."""
    return """{
        "component_type": "Flow",
        "id": "flow_789",
        "name": "JSON Test Flow",
        "description": "A flow loaded from JSON",
        "start_node": {"$component_ref": "start_node"},
        "nodes": [],
        "control_flow_connections": [],
        "data_flow_connections": [],
        "inputs": [],
        "outputs": [],
        "$referenced_components": {
            "start_node": {
                "component_type": "StartNode",
                "id": "start_node",
                "name": "start",
                "inputs": [],
                "outputs": []
            }
        },
        "agentspec_version": "25.4.1"
    }"""

