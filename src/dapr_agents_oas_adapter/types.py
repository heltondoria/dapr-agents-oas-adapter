"""Type definitions and mappings for OAS <-> Dapr Agents conversion."""

from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

# Type aliases for clarity
type ToolRegistry = dict[str, Callable[..., Any]]
type PropertySchema = dict[str, Any]


class NamedCallable(Protocol):
    """Callable that exposes function-like metadata (useful for type checkers).

    Not every Python `Callable` guarantees a `__name__` attribute (e.g., instances with `__call__`),
    but generated functions/wrappers typically expose these fields and tests rely on them.
    """

    __name__: str
    __doc__: str | None

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


class OASComponentType(str, Enum):
    """Open Agent Spec component types."""

    AGENT = "Agent"
    FLOW = "Flow"
    LLM_NODE = "LlmNode"
    TOOL_NODE = "ToolNode"
    AGENT_NODE = "AgentNode"
    FLOW_NODE = "FlowNode"
    MAP_NODE = "MapNode"
    START_NODE = "StartNode"
    END_NODE = "EndNode"
    SERVER_TOOL = "ServerTool"
    REMOTE_TOOL = "RemoteTool"
    MCP_TOOL = "MCPTool"
    CONTROL_FLOW_EDGE = "ControlFlowEdge"
    DATA_FLOW_EDGE = "DataFlowEdge"
    # LLM Config types
    VLLM_CONFIG = "VllmConfig"
    OPENAI_CONFIG = "OpenAIConfig"
    OLLAMA_CONFIG = "OllamaConfig"
    OCI_GENAI_CONFIG = "OciGenAiConfig"


class DaprAgentType(str, Enum):
    """Dapr Agents agent types."""

    AGENT = "Agent"
    ASSISTANT_AGENT = "AssistantAgent"
    DURABLE_AGENT = "DurableAgent"
    REACT_AGENT = "ReActAgent"


class OrchestratorType(str, Enum):
    """Dapr Agents orchestrator types."""

    LLM = "LLMOrchestrator"
    RANDOM = "RandomOrchestrator"
    ROUND_ROBIN = "RoundRobinOrchestrator"


class LlmClientConfig(BaseModel):
    """Configuration for Dapr LLM client."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(description="LLM provider name (e.g. openai, ollama, vllm)")
    model_name: str = Field(description="Model name or identifier")
    url: str | None = Field(default=None, description="Provider API URL")
    api_key: str | None = Field(default=None, description="Provider API key")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")
    extra_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )


class ToolDefinition(BaseModel):
    """Definition for a converted tool."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    tool_type: str = Field(default="function", description="Type of tool (function, mcp, api)")
    inputs: list[PropertySchema] = Field(default_factory=list, description="Input schemas")
    outputs: list[PropertySchema] = Field(default_factory=list, description="Output schemas")
    # Callable can't be serialized to JSON schema, so exclude from schema generation
    implementation: SkipJsonSchema[Callable[..., Any] | None] = Field(
        default=None, description="Tool implementation callable"
    )
    transport_config: dict[str, Any] | None = Field(
        default=None, description="MCP transport configuration (SSE/HTTP)"
    )


class WorkflowTaskDefinition(BaseModel):
    """Definition for a workflow task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Task name")
    task_type: str = Field(description="Task type (llm, tool, agent, flow)")
    config: dict[str, Any] = Field(default_factory=dict, description="Task-specific configuration")
    inputs: list[str] = Field(default_factory=list, description="Input variable names")
    outputs: list[str] = Field(default_factory=list, description="Output variable names")


class WorkflowEdgeDefinition(BaseModel):
    """Definition for workflow edges (control and data flow)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_node: str = Field(description="Source node name")
    to_node: str = Field(description="Target node name")
    from_branch: str | None = Field(default=None, description="Source branch identifier")
    condition: str | None = Field(default=None, description="Edge condition expression")
    data_mapping: dict[str, str] = Field(
        default_factory=dict, description="Data mapping from source to target"
    )


class WorkflowDefinition(BaseModel):
    """Definition for a converted workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Workflow name")
    description: str | None = Field(default=None, description="Workflow description")
    flow_id: str | None = Field(default=None, description="OAS flow identifier")
    tasks: list[WorkflowTaskDefinition] = Field(
        default_factory=list, description="Task definitions"
    )
    edges: list[WorkflowEdgeDefinition] = Field(
        default_factory=list, description="Edge definitions"
    )
    start_node: str | None = Field(default=None, description="Start node name")
    end_nodes: list[str] = Field(default_factory=list, description="End node names")
    inputs: list[PropertySchema] = Field(default_factory=list, description="Workflow input schemas")
    outputs: list[PropertySchema] = Field(
        default_factory=list, description="Workflow output schemas"
    )
    # Optional subflows referenced by FlowNode/MapNode (keyed by flow id).
    subflows: dict[str, "WorkflowDefinition"] = Field(
        default_factory=dict, description="Nested subflow definitions"
    )


class DaprAgentConfig(BaseModel):
    """Configuration model for Dapr Agent creation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Agent name")
    role: str | None = Field(default=None, description="Agent role description")
    goal: str | None = Field(default=None, description="Agent goal description")
    instructions: list[str] = Field(default_factory=list, description="Agent instruction prompts")
    system_prompt: str | None = Field(default=None, description="System prompt text")
    tools: list[str] = Field(default_factory=list, description="Tool names available to agent")
    message_bus_name: str = Field(
        default="messagepubsub", description="Dapr pub/sub component name"
    )
    state_store_name: str = Field(default="statestore", description="Dapr state store name")
    agents_registry_store_name: str = Field(
        default="agentsregistry", description="Dapr agents registry store name"
    )
    service_port: int = Field(default=8000, description="Service listening port")
    # Additional fields for type safety
    agent_type: str | None = Field(default=None, description="Agent class type name")
    llm_config: dict[str, Any] | None = Field(default=None, description="LLM client configuration")
    tool_definitions: list[dict[str, Any]] = Field(
        default_factory=list, description="Tool definition dictionaries"
    )
    input_variables: list[str] = Field(default_factory=list, description="Input variable names")
    # DurableAgent-specific configuration fields
    agent_topic: str | None = Field(
        default=None, description="Dapr pub/sub topic for agent messages"
    )
    broadcast_topic: str | None = Field(
        default=None, description="Dapr pub/sub topic for broadcast messages"
    )
    state_key_prefix: str | None = Field(default=None, description="Prefix for state store keys")
    memory_store_name: str | None = Field(default=None, description="Dapr state store for memory")
    memory_session_id: str | None = Field(default=None, description="Memory session identifier")
    registry_team_name: str | None = Field(default=None, description="Team name in agents registry")


# Component type mappings
OAS_TO_DAPR_AGENT_TYPE: dict[str, DaprAgentType] = {
    "Agent": DaprAgentType.ASSISTANT_AGENT,
    "ReActAgent": DaprAgentType.REACT_AGENT,
}

DAPR_TO_OAS_AGENT_TYPE: dict[DaprAgentType, str] = {
    DaprAgentType.AGENT: "Agent",
    DaprAgentType.ASSISTANT_AGENT: "Agent",
    DaprAgentType.DURABLE_AGENT: "Agent",
    DaprAgentType.REACT_AGENT: "Agent",
}

# LLM provider mappings
OAS_LLM_TO_DAPR_PROVIDER: dict[str, str] = {
    "VllmConfig": "vllm",
    "OpenAIConfig": "openai",
    "OllamaConfig": "ollama",
    "OciGenAiConfig": "oci",
}

DAPR_PROVIDER_TO_OAS_LLM: dict[str, str] = {
    "vllm": "VllmConfig",
    "openai": "OpenAIConfig",
    "ollama": "OllamaConfig",
    "oci": "OciGenAiConfig",
}

# JSON Schema type to Python type mappings
JSON_SCHEMA_TO_PYTHON: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}

PYTHON_TO_JSON_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}
