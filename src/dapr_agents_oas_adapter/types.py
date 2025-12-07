"""Type definitions and mappings for OAS <-> Dapr Agents conversion."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeAlias

from pydantic import BaseModel

# Type aliases for clarity
ToolRegistry: TypeAlias = dict[str, Callable[..., Any]]
PropertySchema: TypeAlias = dict[str, Any]


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


@dataclass
class LlmClientConfig:
    """Configuration for Dapr LLM client."""

    provider: str
    model_id: str
    url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Definition for a converted tool."""

    name: str
    description: str
    inputs: list[PropertySchema]
    outputs: list[PropertySchema]
    implementation: Callable[..., Any] | None = None


@dataclass
class WorkflowTaskDefinition:
    """Definition for a workflow task."""

    name: str
    task_type: str  # "llm", "tool", "agent", "flow"
    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


@dataclass
class WorkflowEdgeDefinition:
    """Definition for workflow edges (control and data flow)."""

    from_node: str
    to_node: str
    from_branch: str | None = None
    condition: str | None = None
    data_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Definition for a converted workflow."""

    name: str
    description: str | None = None
    tasks: list[WorkflowTaskDefinition] = field(default_factory=list)
    edges: list[WorkflowEdgeDefinition] = field(default_factory=list)
    start_node: str | None = None
    end_nodes: list[str] = field(default_factory=list)
    inputs: list[PropertySchema] = field(default_factory=list)
    outputs: list[PropertySchema] = field(default_factory=list)


class DaprAgentConfig(BaseModel):
    """Configuration model for Dapr Agent creation."""

    name: str
    role: str | None = None
    goal: str | None = None
    instructions: list[str] = []
    system_prompt: str | None = None
    tools: list[str] = []
    message_bus_name: str = "messagepubsub"
    state_store_name: str = "statestore"
    agents_registry_store_name: str = "agentsregistry"
    service_port: int = 8000

    model_config = {"extra": "allow"}


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

