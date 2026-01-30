"""Agent converter for OAS <-> Dapr Agents."""

from collections.abc import Callable
from typing import Any

from pyagentspec import Property
from pyagentspec.agent import Agent as OASAgent

from dapr_agents_oas_adapter.converters.base import (
    ComponentConverter,
    ConversionError,
)
from dapr_agents_oas_adapter.converters.llm import LlmConfigConverter
from dapr_agents_oas_adapter.converters.tool import ToolConverter
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    DaprAgentType,
    ToolDefinition,
    ToolRegistry,
)
from dapr_agents_oas_adapter.utils import (
    extract_template_variables,
    generate_id,
)


class AgentConverter(ComponentConverter[OASAgent, DaprAgentConfig]):
    """Converter for OAS Agent <-> Dapr Agent configuration.

    Supports conversion between OAS Agent and various Dapr Agent types
    (AssistantAgent, ReActAgent, DurableAgent).
    """

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        """Initialize the converter.

        Args:
            tool_registry: Dictionary mapping tool names to their implementations
        """
        super().__init__(tool_registry)
        self._llm_converter = LlmConfigConverter()
        self._tool_converter = ToolConverter(tool_registry)

    def from_oas(self, component: OASAgent) -> DaprAgentConfig:
        """Convert an OAS Agent to a Dapr Agent configuration.

        Args:
            component: The OAS Agent to convert

        Returns:
            DaprAgentConfig with equivalent settings

        Raises:
            ConversionError: If the agent cannot be converted
        """
        self.validate_oas_component(component)

        # Extract basic properties
        name = component.name
        # description is extracted via _extract_role_and_goal

        # Extract system prompt and parse for variables
        system_prompt = getattr(component, "system_prompt", "") or ""
        template_vars = extract_template_variables(system_prompt)

        # Determine agent type based on configuration
        agent_type = self._determine_agent_type(component)

        # Extract tools
        tools = self._extract_tools(component)

        # Build role and goal from description/system_prompt
        role, goal = self._extract_role_and_goal(component)

        # Build instructions from system prompt
        instructions = self._build_instructions(system_prompt)

        # Extract metadata for Dapr configuration
        metadata = self.get_component_metadata(component)

        return DaprAgentConfig(
            name=name,
            role=role,
            goal=goal,
            instructions=instructions,
            system_prompt=system_prompt,
            tools=[t.name for t in tools],
            message_bus_name=metadata.get("message_bus_name", "messagepubsub"),
            state_store_name=metadata.get("state_store_name", "statestore"),
            agents_registry_store_name=metadata.get("agents_registry_store_name", "agentsregistry"),
            service_port=metadata.get("service_port", 8000),
            agent_type=agent_type.value,
            llm_config=self._extract_llm_config(component),
            tool_definitions=[self._tool_converter.to_dict(t) for t in tools],
            input_variables=template_vars,
            # DurableAgent-specific fields from metadata
            agent_topic=metadata.get("agent_topic"),
            broadcast_topic=metadata.get("broadcast_topic"),
            state_key_prefix=metadata.get("state_key_prefix"),
            memory_store_name=metadata.get("memory_store_name"),
            memory_session_id=metadata.get("memory_session_id"),
            registry_team_name=metadata.get("registry_team_name"),
        )

    def to_oas(self, component: DaprAgentConfig) -> OASAgent:
        """Convert a Dapr Agent configuration to an OAS Agent.

        Args:
            component: The Dapr Agent configuration to convert

        Returns:
            OAS Agent with equivalent settings
        """
        from pyagentspec.llms import VllmConfig

        agent_id = generate_id("agent")

        # Build LLM config
        llm_config_dict = component.llm_config
        if llm_config_dict:
            llm_config = self._llm_converter.from_dict(llm_config_dict)
            oas_llm = self._llm_converter.to_oas(llm_config)
        else:
            # Create default LLM config
            oas_llm = VllmConfig(
                id=generate_id("llm"),
                name="default_llm",
                model_id="gpt-4",
                url="https://api.openai.com/v1",
            )

        # Build tools
        tool_defs = component.tool_definitions or []
        oas_tools = []
        for tool_dict in tool_defs:
            tool_def = self._tool_converter.from_dict(tool_dict)
            oas_tools.append(self._tool_converter.to_oas(tool_def))

        # Build system prompt
        system_prompt = component.system_prompt or self._build_system_prompt(component)

        # Build inputs from variables as Property objects
        inputs = self._build_inputs_as_properties(component)

        # Build metadata with DurableAgent-specific fields and core config
        # Use `is not None` to preserve empty strings (not truthy checks)
        metadata: dict[str, Any] = {}
        if component.agent_type is not None:
            metadata["dapr_agent_type"] = component.agent_type
        if component.agent_topic is not None:
            metadata["agent_topic"] = component.agent_topic
        if component.broadcast_topic is not None:
            metadata["broadcast_topic"] = component.broadcast_topic
        if component.state_key_prefix is not None:
            metadata["state_key_prefix"] = component.state_key_prefix
        if component.memory_store_name is not None:
            metadata["memory_store_name"] = component.memory_store_name
        if component.memory_session_id is not None:
            metadata["memory_session_id"] = component.memory_session_id
        if component.registry_team_name is not None:
            metadata["registry_team_name"] = component.registry_team_name
        # Include core configuration fields if they differ from defaults
        if component.message_bus_name != "messagepubsub":
            metadata["message_bus_name"] = component.message_bus_name
        if component.state_store_name != "statestore":
            metadata["state_store_name"] = component.state_store_name
        if component.agents_registry_store_name != "agentsregistry":
            metadata["agents_registry_store_name"] = component.agents_registry_store_name
        if component.service_port != 8000:
            metadata["service_port"] = component.service_port

        return OASAgent(
            id=agent_id,
            name=component.name,
            description=component.goal,
            llm_config=oas_llm,
            system_prompt=system_prompt,
            tools=oas_tools,
            inputs=inputs,
            outputs=None,
            metadata=metadata if metadata else None,
        )

    def can_convert(self, component: Any) -> bool:
        """Check if this converter can handle the given component.

        Args:
            component: The component to check

        Returns:
            True if this converter can handle the component
        """
        if isinstance(component, OASAgent):
            return True
        if isinstance(component, DaprAgentConfig):
            return True
        if isinstance(component, dict):
            comp_type = component.get("component_type", "")
            return comp_type == "Agent"
        return False

    def from_dict(self, agent_dict: dict[str, Any]) -> DaprAgentConfig:
        """Convert a dictionary representation to DaprAgentConfig.

        Args:
            agent_dict: Dictionary with agent configuration

        Returns:
            DaprAgentConfig with the converted settings
        """
        # Extract LLM config
        llm_config = agent_dict.get("llm_config", {})

        # Extract tools and convert to proper format
        tools = agent_dict.get("tools", [])
        tool_names = [t.get("name", "") if isinstance(t, dict) else str(t) for t in tools]
        # Ensure tool_definitions are properly formatted dictionaries
        tool_definitions = []
        for t in tools:
            if isinstance(t, dict):
                tool_definitions.append(
                    self._tool_converter.to_dict(self._tool_converter.from_dict(t))
                )
            else:
                tool_definitions.append(
                    {
                        "name": str(t),
                        "description": "",
                        "inputs": [],
                        "outputs": [],
                    }
                )

        # Extract system prompt
        system_prompt = agent_dict.get("system_prompt", "")
        template_vars = extract_template_variables(system_prompt)

        # Build role and goal
        role = agent_dict.get("role") or agent_dict.get("name", "")
        goal = agent_dict.get("goal") or agent_dict.get("description", "")

        # Extract metadata for DurableAgent-specific fields
        # Use `or {}` to handle both missing keys and explicit None values
        metadata = agent_dict.get("metadata") or {}

        # Helper to get value from dict with fallback to metadata
        # Uses `is not None` to preserve empty strings (not `or` which treats "" as falsy)
        def _get_with_fallback(key: str, meta_key: str | None = None) -> Any:
            value = agent_dict.get(key)
            if value is not None:
                return value
            return metadata.get(meta_key or key)

        return DaprAgentConfig(
            name=agent_dict.get("name", ""),
            role=role,
            goal=goal,
            instructions=self._build_instructions(system_prompt),
            system_prompt=system_prompt,
            tools=tool_names,
            message_bus_name=agent_dict.get("message_bus_name", "messagepubsub"),
            state_store_name=agent_dict.get("state_store_name", "statestore"),
            agents_registry_store_name=agent_dict.get(
                "agents_registry_store_name", "agentsregistry"
            ),
            service_port=agent_dict.get("service_port", 8000),
            # Use _get_with_fallback to preserve empty strings (not treat as falsy)
            agent_type=_get_with_fallback("agent_type", "dapr_agent_type"),
            llm_config=llm_config,
            tool_definitions=tool_definitions,
            input_variables=template_vars,
            # DurableAgent-specific fields from metadata or direct dict keys
            agent_topic=_get_with_fallback("agent_topic"),
            broadcast_topic=_get_with_fallback("broadcast_topic"),
            state_key_prefix=_get_with_fallback("state_key_prefix"),
            memory_store_name=_get_with_fallback("memory_store_name"),
            memory_session_id=_get_with_fallback("memory_session_id"),
            registry_team_name=_get_with_fallback("registry_team_name"),
        )

    def to_dict(self, config: DaprAgentConfig) -> dict[str, Any]:
        """Convert DaprAgentConfig to a dictionary representation.

        Args:
            config: The DaprAgentConfig to convert

        Returns:
            Dictionary representation of the agent
        """
        llm_config = config.llm_config or {}
        tool_defs = config.tool_definitions or []

        # Build metadata with DurableAgent-specific fields if present
        # Use `is not None` to preserve empty strings (not truthy checks)
        metadata: dict[str, Any] = {}
        if config.agent_type is not None:
            metadata["dapr_agent_type"] = config.agent_type
        if config.agent_topic is not None:
            metadata["agent_topic"] = config.agent_topic
        if config.broadcast_topic is not None:
            metadata["broadcast_topic"] = config.broadcast_topic
        if config.state_key_prefix is not None:
            metadata["state_key_prefix"] = config.state_key_prefix
        if config.memory_store_name is not None:
            metadata["memory_store_name"] = config.memory_store_name
        if config.memory_session_id is not None:
            metadata["memory_session_id"] = config.memory_session_id
        if config.registry_team_name is not None:
            metadata["registry_team_name"] = config.registry_team_name

        # Include core configuration fields if they differ from defaults
        if config.message_bus_name != "messagepubsub":
            metadata["message_bus_name"] = config.message_bus_name
        if config.state_store_name != "statestore":
            metadata["state_store_name"] = config.state_store_name
        if config.agents_registry_store_name != "agentsregistry":
            metadata["agents_registry_store_name"] = config.agents_registry_store_name
        if config.service_port != 8000:
            metadata["service_port"] = config.service_port

        result: dict[str, Any] = {
            "component_type": "Agent",
            "id": generate_id("agent"),
            "name": config.name,
            "role": config.role,
            "goal": config.goal,
            "description": config.goal,
            "llm_config": llm_config,
            "system_prompt": config.system_prompt or self._build_system_prompt(config),
            "tools": tool_defs,
            "inputs": self._build_inputs(config),
            "outputs": [],
            # Include core config fields at top level for easier access
            "message_bus_name": config.message_bus_name,
            "state_store_name": config.state_store_name,
            "agents_registry_store_name": config.agents_registry_store_name,
            "service_port": config.service_port,
        }

        if metadata:
            result["metadata"] = metadata

        return result

    def create_dapr_agent(
        self,
        config: DaprAgentConfig,
        tool_implementations: dict[str, Callable[..., Any]] | None = None,
    ) -> Any:
        """Create a Dapr Agent instance from configuration.

        This method creates the actual Dapr Agent object that can be started.

        Args:
            config: The agent configuration
            tool_implementations: Optional tool implementations

        Returns:
            A Dapr Agent instance (AssistantAgent, ReActAgent, or DurableAgent)

        Raises:
            ConversionError: If agent creation fails
        """
        try:
            from dapr_agents import AssistantAgent  # type: ignore[import-not-found]
            from dapr_agents import tool as dapr_tool  # type: ignore[import-not-found]

            # Merge tool registries
            all_tools = {**self._tool_registry}
            if tool_implementations:
                all_tools.update(tool_implementations)

            # Create tool functions with @tool decorator
            decorated_tools = []
            for tool_name in config.tools:
                if tool_name in all_tools:
                    func = all_tools[tool_name]
                    # Apply @tool decorator if not already applied
                    if not hasattr(func, "_is_dapr_tool"):
                        func = dapr_tool(func)
                    decorated_tools.append(func)

            # Determine agent class
            agent_type = config.agent_type or DaprAgentType.ASSISTANT_AGENT.value

            if agent_type == DaprAgentType.REACT_AGENT.value:
                from dapr_agents import ReActAgent  # type: ignore[import-not-found]

                return ReActAgent(
                    name=config.name,
                    role=config.role or config.name,
                    instructions=config.instructions,
                    tools=decorated_tools,
                )
            elif agent_type == DaprAgentType.DURABLE_AGENT.value:
                from dapr_agents import DurableAgent  # type: ignore[import-not-found]
                from dapr_agents.agents.configs import (  # type: ignore[import-not-found]
                    AgentMemoryConfig,
                    AgentPubSubConfig,
                    AgentRegistryConfig,
                    AgentStateConfig,
                )
                from dapr_agents.memory import (
                    ConversationDaprStateMemory,  # type: ignore[import-not-found]
                )
                from dapr_agents.storage.daprstores.stateservice import (
                    StateStoreService,  # type: ignore[import-not-found]
                )

                # Create LLM client (required for DurableAgent)
                llm_client = self._create_llm_client(config.llm_config)

                # Configure PubSub
                # Use `is not None` to preserve empty strings (not `or` which treats "" as falsy)
                pubsub_config = AgentPubSubConfig(
                    pubsub_name=config.message_bus_name,
                    agent_topic=(
                        config.agent_topic
                        if config.agent_topic is not None
                        else f"{config.name}.requests"
                    ),
                    broadcast_topic=(
                        config.broadcast_topic
                        if config.broadcast_topic is not None
                        else "broadcast"
                    ),
                )

                # Configure State
                state_store = StateStoreService(
                    store_name=config.state_store_name,
                    key_prefix=(
                        config.state_key_prefix
                        if config.state_key_prefix is not None
                        else f"{config.name}:"
                    ),
                )
                state_config = AgentStateConfig(store=state_store)

                # Configure Memory
                memory_store = ConversationDaprStateMemory(
                    store_name=(
                        config.memory_store_name
                        if config.memory_store_name is not None
                        else "memorystore"
                    ),
                    session_id=(
                        config.memory_session_id
                        if config.memory_session_id is not None
                        else f"{config.name}-session"
                    ),
                )
                memory_config = AgentMemoryConfig(store=memory_store)

                # Configure Registry
                registry_store = StateStoreService(
                    store_name=config.agents_registry_store_name,
                )
                registry_config = (
                    AgentRegistryConfig(
                        store=registry_store,
                        team_name=config.registry_team_name,
                    )
                    if config.registry_team_name
                    else None
                )

                # Create DurableAgent with all configurations
                durable_agent_kwargs: dict[str, Any] = {
                    "name": config.name,
                    "role": config.role or config.name,
                    "goal": config.goal,
                    "instructions": config.instructions,
                    "tools": decorated_tools,
                    "llm": llm_client,
                    "pubsub": pubsub_config,
                    "state": state_config,
                    "memory": memory_config,
                }
                if registry_config:
                    durable_agent_kwargs["registry"] = registry_config

                return DurableAgent(**durable_agent_kwargs)
            else:
                return AssistantAgent(
                    name=config.name,
                    role=config.role or config.name,
                    goal=config.goal,
                    instructions=config.instructions,
                    tools=decorated_tools,
                    message_bus_name=config.message_bus_name,
                    state_store_name=config.state_store_name,
                    agents_registry_store_name=config.agents_registry_store_name,
                    service_port=config.service_port,
                )

        except ImportError as e:
            raise ConversionError(
                "Failed to import Dapr Agents",
                config,
                suggestion="Install dapr-agents: pip install dapr-agents",
                caused_by=e,
            ) from e
        except Exception as e:
            raise ConversionError(
                "Failed to create Dapr Agent",
                config,
                suggestion="Check agent configuration and ensure all required fields are set",
                caused_by=e,
            ) from e

    def _create_llm_client(self, llm_config: dict[str, Any] | None) -> Any:
        """Create a Dapr LLM client from configuration.

        Args:
            llm_config: Dictionary with LLM configuration

        Returns:
            A Dapr LLM client instance

        Raises:
            ConversionError: If client creation fails
        """
        try:
            provider = llm_config.get("provider", "openai") if llm_config else "openai"
            model_id = llm_config.get("model_id", "gpt-4") if llm_config else "gpt-4"

            if provider == "openai":
                from dapr_agents import OpenAIChatClient  # type: ignore[import-not-found]

                return OpenAIChatClient(model=model_id)  # type: ignore[abstract]
            elif provider == "ollama":
                # Dapr Agents does not expose a dedicated Ollama client in all versions.
                # Treat Ollama as an OpenAI-compatible endpoint.
                from dapr_agents import OpenAIChatClient  # type: ignore[import-not-found]

                url = (
                    llm_config.get("url", "http://localhost:11434")
                    if llm_config
                    else "http://localhost:11434"
                )
                return OpenAIChatClient(model=model_id, base_url=url)  # type: ignore[abstract]
            elif provider == "vllm":
                from dapr_agents import OpenAIChatClient  # type: ignore[import-not-found]

                url = llm_config.get("url") if llm_config else None
                return OpenAIChatClient(model=model_id, base_url=url)  # type: ignore[abstract]
            else:
                # Default to OpenAI-compatible client
                from dapr_agents import OpenAIChatClient  # type: ignore[import-not-found]

                return OpenAIChatClient(model=model_id)  # type: ignore[abstract]
        except ImportError as e:
            raise ConversionError(
                "Failed to import LLM client",
                suggestion="Install dapr-agents: pip install dapr-agents",
                caused_by=e,
            ) from e

    def _determine_agent_type(self, component: OASAgent) -> DaprAgentType:
        """Determine the appropriate Dapr agent type for an OAS Agent."""
        # Check metadata for explicit type
        # Use `is not None` to preserve empty strings (not truthy checks)
        if component.metadata:
            explicit_type = component.metadata.get("dapr_agent_type")
            if explicit_type is not None:
                try:
                    return DaprAgentType(explicit_type)
                except ValueError:
                    pass

            # Check for DurableAgent-specific configuration in metadata
            durable_agent_keys = [
                "agent_topic",
                "broadcast_topic",
                "registry_team_name",
                "memory_store_name",
                "memory_session_id",
                "state_key_prefix",
            ]
            if any(key in component.metadata for key in durable_agent_keys):
                return DaprAgentType.DURABLE_AGENT

        # Check if agent has tools (suggests ReActAgent)
        tools = getattr(component, "tools", [])
        if tools and len(tools) > 0:
            # Agents with tools that need reasoning -> ReActAgent
            system_prompt = getattr(component, "system_prompt", "") or ""
            if "reason" in system_prompt.lower() or "think" in system_prompt.lower():
                return DaprAgentType.REACT_AGENT

        # Default to AssistantAgent
        return DaprAgentType.ASSISTANT_AGENT

    def _extract_tools(self, component: OASAgent) -> list[ToolDefinition]:
        """Extract tool definitions from an OAS Agent."""
        tools: list[ToolDefinition] = []
        oas_tools = getattr(component, "tools", [])

        for tool in oas_tools:
            if tool:
                tool_def = self._tool_converter.from_oas(tool)
                tools.append(tool_def)

        return tools

    def _extract_llm_config(self, component: OASAgent) -> dict[str, Any]:
        """Extract LLM configuration from an OAS Agent."""
        llm_config = getattr(component, "llm_config", None)
        if llm_config:
            dapr_config = self._llm_converter.from_oas(llm_config)
            return self._llm_converter.to_dict(dapr_config)
        return {}

    def _extract_role_and_goal(self, component: OASAgent) -> tuple[str, str]:
        """Extract role and goal from an OAS Agent."""
        description = component.description or ""
        system_prompt = getattr(component, "system_prompt", "") or ""

        # Role is typically the agent name or first line of system prompt
        role = component.name

        # Goal is the description or extracted from system prompt
        goal = description
        if not goal and system_prompt:
            # Try to extract goal from system prompt
            lines = system_prompt.strip().split("\n")
            if lines:
                goal = lines[0][:200]  # First line, truncated

        return role, goal

    def _build_instructions(self, system_prompt: str) -> list[str]:
        """Build instructions list from system prompt."""
        if not system_prompt:
            return []

        # Split system prompt into instruction lines
        lines = [
            line.strip()
            for line in system_prompt.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Return non-empty lines as instructions
        return lines[:10]  # Limit to 10 instructions

    def _build_system_prompt(self, config: DaprAgentConfig) -> str:
        """Build a system prompt from Dapr config."""
        parts = []

        if config.role:
            parts.append(f"You are {config.role}.")

        if config.goal:
            parts.append(f"Your goal is to {config.goal}.")

        if config.instructions:
            parts.append("\nInstructions:")
            for instruction in config.instructions:
                parts.append(f"- {instruction}")

        return "\n".join(parts)

    def _build_inputs(self, config: DaprAgentConfig) -> list[dict[str, Any]]:
        """Build OAS inputs from Dapr config as dictionaries."""
        inputs: list[dict[str, Any]] = []
        input_vars = config.input_variables or []

        for var in input_vars:
            inputs.append(
                {
                    "title": var,
                    "type": "string",
                }
            )

        return inputs

    def _build_inputs_as_properties(self, config: DaprAgentConfig) -> list[Property]:
        """Build OAS inputs from Dapr config as Property objects."""
        inputs: list[Property] = []
        input_vars = config.input_variables or []

        for var in input_vars:
            inputs.append(
                Property(
                    title=var,
                    type="string",
                )
            )

        return inputs
