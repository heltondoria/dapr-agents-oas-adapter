"""Property-based tests using hypothesis.

These tests verify invariants of the data models and converters
using generated test data rather than specific examples.
"""

from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from dapr_agents_oas_adapter.converters.flow import FlowConverter
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    DaprAgentType,
    LlmClientConfig,
    PropertySchema,
    ToolDefinition,
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)
from dapr_agents_oas_adapter.utils import IDGenerator

# =============================================================================
# Custom Strategies for Domain Types
# =============================================================================


@st.composite
def valid_identifiers(draw: st.DrawFn) -> str:
    """Generate valid Python/OAS identifiers."""
    first_char = draw(st.sampled_from("abcdefghijklmnopqrstuvwxyz_"))
    rest = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=0, max_size=20))
    return first_char + rest


@st.composite
def property_schemas(draw: st.DrawFn) -> PropertySchema:
    """Generate valid PropertySchema dicts."""
    return {
        "title": draw(valid_identifiers()),
        "type": draw(
            st.sampled_from(["string", "integer", "number", "boolean", "array", "object"])
        ),
        "description": draw(st.text(min_size=0, max_size=100) | st.none()),
        "default": draw(st.none() | st.text(max_size=20) | st.integers(-100, 100)),
    }


@st.composite
def retry_policy_dicts(draw: st.DrawFn) -> dict[str, Any]:
    """Generate valid retry policy configuration dicts."""
    return {
        "max_attempts": draw(st.integers(min_value=1, max_value=10)),
        "initial_backoff_seconds": draw(st.integers(min_value=1, max_value=60)),
        "max_backoff_seconds": draw(st.integers(min_value=1, max_value=300)),
        "backoff_multiplier": draw(st.floats(min_value=1.0, max_value=5.0)),
        "retry_timeout": draw(st.none() | st.integers(min_value=1, max_value=3600)),
    }


@st.composite
def llm_client_configs(draw: st.DrawFn) -> LlmClientConfig:
    """Generate valid LlmClientConfig instances."""
    return LlmClientConfig(
        provider=draw(st.sampled_from(["openai", "ollama", "vllm", "oci"])),
        model_name=draw(
            st.text(min_size=1, max_size=50, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_.")
        ),
        url=draw(st.none() | st.just("http://localhost:8000")),
        api_key=draw(st.none() | st.text(min_size=1, max_size=50)),
        temperature=draw(st.floats(min_value=0.0, max_value=2.0)),
        max_tokens=draw(st.none() | st.integers(min_value=1, max_value=4096)),
        extra_params=draw(st.fixed_dictionaries({})),
    )


@st.composite
def tool_definitions(draw: st.DrawFn) -> ToolDefinition:
    """Generate valid ToolDefinition instances."""
    name = draw(valid_identifiers())
    description = draw(st.text(min_size=1, max_size=200))
    inputs = [draw(property_schemas()) for _ in range(draw(st.integers(min_value=0, max_value=3)))]
    outputs = [draw(property_schemas()) for _ in range(draw(st.integers(min_value=0, max_value=2)))]
    return ToolDefinition(
        name=name,
        description=description,
        inputs=inputs,
        outputs=outputs,
        implementation=None,
        transport_config=None,
    )


@st.composite
def workflow_task_definitions(draw: st.DrawFn) -> WorkflowTaskDefinition:
    """Generate valid WorkflowTaskDefinition instances."""
    task_type = draw(st.sampled_from(["llm", "tool", "agent", "flow", "map", "start", "end"]))
    return WorkflowTaskDefinition(
        name=draw(valid_identifiers()),
        task_type=task_type,
        config=draw(st.fixed_dictionaries({})),
        inputs=draw(st.lists(valid_identifiers(), min_size=0, max_size=3)),
        outputs=draw(st.lists(valid_identifiers(), min_size=0, max_size=3)),
    )


@st.composite
def workflow_edge_definitions(draw: st.DrawFn, task_names: list[str]) -> WorkflowEdgeDefinition:
    """Generate valid WorkflowEdgeDefinition for given task names."""
    if len(task_names) < 2:
        # Need at least 2 tasks for an edge
        return WorkflowEdgeDefinition(
            from_node="start",
            to_node="end",
        )
    from_node = draw(st.sampled_from(task_names))
    to_node = draw(st.sampled_from([n for n in task_names if n != from_node] or task_names))
    return WorkflowEdgeDefinition(
        from_node=from_node,
        to_node=to_node,
        from_branch=draw(st.none() | st.text(min_size=1, max_size=20)),
        data_mapping=draw(st.fixed_dictionaries({})),
    )


@st.composite
def simple_workflow_definitions(draw: st.DrawFn) -> WorkflowDefinition:
    """Generate valid WorkflowDefinition with simple structure."""
    name = draw(valid_identifiers())

    # Always include start and end
    tasks = [
        WorkflowTaskDefinition(name="start", task_type="start"),
        WorkflowTaskDefinition(
            name=draw(valid_identifiers()),
            task_type=draw(st.sampled_from(["llm", "tool"])),
        ),
        WorkflowTaskDefinition(name="end", task_type="end"),
    ]

    # Connect start -> middle -> end
    edges = [
        WorkflowEdgeDefinition(from_node="start", to_node=tasks[1].name),
        WorkflowEdgeDefinition(from_node=tasks[1].name, to_node="end"),
    ]

    return WorkflowDefinition(
        name=name,
        description=draw(st.none() | st.text(min_size=1, max_size=100)),
        tasks=tasks,
        edges=edges,
        start_node="start",
        end_nodes=["end"],
    )


@st.composite
def dapr_agent_configs(draw: st.DrawFn) -> DaprAgentConfig:
    """Generate valid DaprAgentConfig instances."""
    name = draw(valid_identifiers())
    role = draw(st.text(min_size=1, max_size=50))
    goal = draw(st.text(min_size=1, max_size=100))
    instructions = draw(st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=5))
    system_prompt = draw(st.none() | st.text(min_size=1, max_size=200))
    tools = [draw(valid_identifiers()) for _ in range(draw(st.integers(min_value=0, max_value=3)))]
    # These fields have defaults and don't accept None, so generate valid strings
    message_bus = draw(valid_identifiers())
    state_store = draw(valid_identifiers())
    registry_store = draw(valid_identifiers())
    port = draw(st.integers(min_value=1024, max_value=65535))
    agent_type = draw(st.sampled_from(list(DaprAgentType)))
    tool_defs = [
        draw(tool_definitions()) for _ in range(draw(st.integers(min_value=0, max_value=2)))
    ]

    return DaprAgentConfig(
        name=name,
        role=role,
        goal=goal,
        instructions=instructions,
        system_prompt=system_prompt,
        tools=tools,
        llm_config=None,
        message_bus_name=message_bus,
        state_store_name=state_store,
        agents_registry_store_name=registry_store,
        service_port=port,
        agent_type=agent_type,
        tool_definitions=tool_defs,
    )


# =============================================================================
# Property Tests for Data Models
# =============================================================================


class TestPropertySchemaProperties:
    """Property-based tests for PropertySchema (dict type alias)."""

    @given(property_schemas())
    @settings(max_examples=50)
    def test_schema_is_valid_dict(self, schema: PropertySchema) -> None:
        """PropertySchema dicts have required keys."""
        assert isinstance(schema, dict)
        assert "title" in schema
        assert "type" in schema
        assert schema["type"] in ["string", "integer", "number", "boolean", "array", "object"]

    @given(property_schemas())
    @settings(max_examples=50)
    def test_schema_title_is_identifier(self, schema: PropertySchema) -> None:
        """PropertySchema title is a valid identifier."""
        title = schema["title"]
        assert len(title) > 0
        assert title[0].isalpha() or title[0] == "_"


class TestLlmClientConfigProperties:
    """Property-based tests for LlmClientConfig."""

    @given(llm_client_configs())
    @settings(max_examples=50)
    def test_config_is_valid(self, config: LlmClientConfig) -> None:
        """LlmClientConfig instances are always valid."""
        assert config.provider in ["openai", "ollama", "vllm", "oci"]
        assert len(config.model_name) > 0
        assert 0.0 <= config.temperature <= 2.0

    @given(llm_client_configs())
    @settings(max_examples=50)
    def test_config_roundtrip(self, config: LlmClientConfig) -> None:
        """LlmClientConfig can be serialized and deserialized."""
        data = config.model_dump()
        restored = LlmClientConfig.model_validate(data)
        assert restored.provider == config.provider
        assert restored.model_name == config.model_name
        assert restored.temperature == config.temperature


class TestRetryPolicyDictProperties:
    """Property-based tests for retry policy dicts."""

    @given(retry_policy_dicts())
    @settings(max_examples=50)
    def test_policy_invariants(self, policy: dict[str, Any]) -> None:
        """Retry policy dicts maintain invariants."""
        assert policy["max_attempts"] >= 1
        assert policy["initial_backoff_seconds"] >= 1
        assert policy["backoff_multiplier"] >= 1.0

    @given(retry_policy_dicts())
    @settings(max_examples=50)
    def test_policy_has_required_keys(self, policy: dict[str, Any]) -> None:
        """Retry policy dicts have required keys."""
        assert "max_attempts" in policy
        assert "initial_backoff_seconds" in policy
        assert "backoff_multiplier" in policy


class TestToolDefinitionProperties:
    """Property-based tests for ToolDefinition."""

    @given(tool_definitions())
    @settings(max_examples=50)
    def test_tool_has_required_fields(self, tool: ToolDefinition) -> None:
        """ToolDefinition always has name and description."""
        assert len(tool.name) > 0
        assert len(tool.description) > 0

    @given(tool_definitions())
    @settings(max_examples=50)
    def test_tool_roundtrip(self, tool: ToolDefinition) -> None:
        """ToolDefinition can be serialized and deserialized."""
        data = tool.model_dump()
        restored = ToolDefinition.model_validate(data)
        assert restored.name == tool.name
        assert restored.description == tool.description
        assert len(restored.inputs) == len(tool.inputs)


class TestWorkflowTaskDefinitionProperties:
    """Property-based tests for WorkflowTaskDefinition."""

    @given(workflow_task_definitions())
    @settings(max_examples=50)
    def test_task_has_valid_type(self, task: WorkflowTaskDefinition) -> None:
        """WorkflowTaskDefinition always has valid task_type."""
        valid_types = {"llm", "tool", "agent", "flow", "map", "start", "end"}
        assert task.task_type in valid_types

    @given(workflow_task_definitions())
    @settings(max_examples=50)
    def test_task_roundtrip(self, task: WorkflowTaskDefinition) -> None:
        """WorkflowTaskDefinition can be serialized and deserialized."""
        data = task.model_dump()
        restored = WorkflowTaskDefinition.model_validate(data)
        assert restored.name == task.name
        assert restored.task_type == task.task_type


class TestWorkflowDefinitionProperties:
    """Property-based tests for WorkflowDefinition."""

    @given(simple_workflow_definitions())
    @settings(max_examples=30)
    def test_workflow_has_structure(self, workflow: WorkflowDefinition) -> None:
        """WorkflowDefinition always has valid structure."""
        assert len(workflow.name) > 0
        assert len(workflow.tasks) >= 1
        # Should have at least start or end if specified
        if workflow.start_node:
            assert any(t.name == workflow.start_node for t in workflow.tasks)

    @given(simple_workflow_definitions())
    @settings(max_examples=30)
    def test_workflow_roundtrip(self, workflow: WorkflowDefinition) -> None:
        """WorkflowDefinition can be serialized and deserialized."""
        data = workflow.model_dump()
        restored = WorkflowDefinition.model_validate(data)
        assert restored.name == workflow.name
        assert len(restored.tasks) == len(workflow.tasks)
        assert len(restored.edges) == len(workflow.edges)


class TestDaprAgentConfigProperties:
    """Property-based tests for DaprAgentConfig."""

    @given(dapr_agent_configs())
    @settings(max_examples=30)
    def test_agent_has_required_fields(self, agent: DaprAgentConfig) -> None:
        """DaprAgentConfig always has required fields."""
        assert len(agent.name) > 0
        # role and goal are set by our strategy to non-empty strings
        assert agent.role is not None
        assert len(agent.role) > 0
        assert agent.goal is not None
        assert len(agent.goal) > 0

    @given(dapr_agent_configs())
    @settings(max_examples=30)
    def test_agent_roundtrip(self, agent: DaprAgentConfig) -> None:
        """DaprAgentConfig can be serialized and deserialized."""
        data = agent.model_dump()
        restored = DaprAgentConfig.model_validate(data)
        assert restored.name == agent.name
        assert restored.role == agent.role
        assert restored.agent_type == agent.agent_type


# =============================================================================
# Property Tests for IDGenerator
# =============================================================================


class TestIDGeneratorProperties:
    """Property-based tests for IDGenerator."""

    @given(st.integers(min_value=0, max_value=1000000))
    @settings(max_examples=50)
    def test_seeded_generator_deterministic(self, seed: int) -> None:
        """Seeded generator produces same IDs for same seed."""
        gen1 = IDGenerator(seed=seed)
        gen2 = IDGenerator(seed=seed)

        ids1 = [gen1.generate("test") for _ in range(5)]
        ids2 = [gen2.generate("test") for _ in range(5)]

        assert ids1 == ids2

    @given(st.text(min_size=0, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"))
    @settings(max_examples=50)
    def test_prefix_preserved(self, prefix: str) -> None:
        """Generated IDs always start with the given prefix."""
        gen = IDGenerator(seed=42)
        generated_id = gen.generate(prefix)

        if prefix:
            assert generated_id.startswith(prefix)

    @given(st.integers(min_value=0, max_value=1000))
    @settings(max_examples=30)
    def test_ids_unique_within_generator(self, seed: int) -> None:
        """IDs from same generator are unique."""
        gen = IDGenerator(seed=seed)
        ids = [gen.generate("item") for _ in range(100)]
        assert len(ids) == len(set(ids))


# =============================================================================
# Property Tests for Converter Roundtrips
# =============================================================================


class TestFlowConverterProperties:
    """Property-based tests for FlowConverter roundtrips."""

    @given(simple_workflow_definitions())
    @settings(max_examples=20)
    def test_workflow_to_dict_roundtrip(self, workflow: WorkflowDefinition) -> None:
        """WorkflowDefinition can roundtrip through to_dict/from_dict."""
        # Reset ID generator for deterministic IDs
        IDGenerator.reset_instance(seed=12345)

        converter = FlowConverter()
        dict_form = converter.to_dict(workflow)

        # Reset again for from_dict
        IDGenerator.reset_instance(seed=12345)
        restored = converter.from_dict(dict_form)

        assert restored.name == workflow.name
        assert len(restored.tasks) == len(workflow.tasks)

        # Task names should be preserved
        original_names = {t.name for t in workflow.tasks}
        restored_names = {t.name for t in restored.tasks}
        assert original_names == restored_names

    @given(simple_workflow_definitions())
    @settings(max_examples=20)
    def test_workflow_dict_has_required_fields(self, workflow: WorkflowDefinition) -> None:
        """Workflow dict representation has required OAS fields."""
        IDGenerator.reset_instance(seed=42)
        converter = FlowConverter()
        dict_form = converter.to_dict(workflow)

        assert "component_type" in dict_form
        assert dict_form["component_type"] == "Flow"
        assert "name" in dict_form
        assert "id" in dict_form
        assert "nodes" in dict_form
