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

    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str
    url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """Definition for a converted tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str
    inputs: list[PropertySchema] = Field(default_factory=list)
    outputs: list[PropertySchema] = Field(default_factory=list)
    # Callable can't be serialized to JSON schema, so exclude from schema generation
    implementation: SkipJsonSchema[Callable[..., Any] | None] = None
    mcp_transport: dict[str, Any] | None = None  # MCP transport config (SSE/HTTP)


class WorkflowTaskDefinition(BaseModel):
    """Definition for a workflow task."""

    model_config = ConfigDict(extra="forbid")

    name: str
    task_type: str  # "llm", "tool", "agent", "flow"
    config: dict[str, Any] = Field(default_factory=dict)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)


class WorkflowEdgeDefinition(BaseModel):
    """Definition for workflow edges (control and data flow)."""

    model_config = ConfigDict(extra="forbid")

    from_node: str
    to_node: str
    from_branch: str | None = None
    condition: str | None = None
    data_mapping: dict[str, str] = Field(default_factory=dict)


class WorkflowDefinition(BaseModel):
    """Definition for a converted workflow."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    flow_id: str | None = None
    tasks: list[WorkflowTaskDefinition] = Field(default_factory=list)
    edges: list[WorkflowEdgeDefinition] = Field(default_factory=list)
    start_node: str | None = None
    end_nodes: list[str] = Field(default_factory=list)
    inputs: list[PropertySchema] = Field(default_factory=list)
    outputs: list[PropertySchema] = Field(default_factory=list)
    # Optional subflows referenced by FlowNode/MapNode (keyed by flow id).
    subflows: dict[str, "WorkflowDefinition"] = Field(default_factory=dict)


class DaprAgentConfig(BaseModel):
    """Configuration model for Dapr Agent creation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str | None = None
    goal: str | None = None
    instructions: list[str] = Field(default_factory=list)
    system_prompt: str | None = None
    tools: list[str] = Field(default_factory=list)
    message_bus_name: str = "messagepubsub"
    state_store_name: str = "statestore"
    agents_registry_store_name: str = "agentsregistry"
    service_port: int = 8000
    # Additional fields for type safety
    agent_type: str | None = None
    llm_config: dict[str, Any] | None = None
    tool_definitions: list[dict[str, Any]] = Field(default_factory=list)
    input_variables: list[str] = Field(default_factory=list)
    # DurableAgent-specific configuration fields
    agent_topic: str | None = None
    broadcast_topic: str | None = None
    state_key_prefix: str | None = None
    memory_store_name: str | None = None
    memory_session_id: str | None = None
    registry_team_name: str | None = None


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
