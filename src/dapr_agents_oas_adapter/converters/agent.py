"""Agent converter for OAS <-> Dapr Agents."""

from typing import Any, Callable

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
            agents_registry_store_name=metadata.get(
                "agents_registry_store_name", "agentsregistry"
            ),
            service_port=metadata.get("service_port", 8000),
            # Store additional config via model_config extra="allow"
            agent_type=agent_type.value,
            llm_config=self._extract_llm_config(component),
            tool_definitions=[self._tool_converter.to_dict(t) for t in tools],
            input_variables=template_vars,
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
        llm_config_dict = getattr(component, "llm_config", None)
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
        tool_defs = getattr(component, "tool_definitions", [])
        oas_tools = []
        for tool_dict in tool_defs:
            tool_def = self._tool_converter.from_dict(tool_dict)
            oas_tools.append(self._tool_converter.to_oas(tool_def))

        # Build system prompt
        system_prompt = component.system_prompt or self._build_system_prompt(component)

        # Build inputs from variables
        inputs = self._build_inputs(component)

        return OASAgent(
            id=agent_id,
            name=component.name,
            description=component.goal,
            llm_config=oas_llm,
            system_prompt=system_prompt,
            tools=oas_tools,
            inputs=inputs,
            outputs=[],
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
                tool_definitions.append({
                    "name": str(t),
                    "description": "",
                    "inputs": [],
                    "outputs": [],
                })

        # Extract system prompt
        system_prompt = agent_dict.get("system_prompt", "")
        template_vars = extract_template_variables(system_prompt)

        # Build role and goal
        role = agent_dict.get("role") or agent_dict.get("name", "")
        goal = agent_dict.get("goal") or agent_dict.get("description", "")

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
            agent_type=agent_dict.get("agent_type", DaprAgentType.ASSISTANT_AGENT.value),
            llm_config=llm_config,
            tool_definitions=tool_definitions,
            input_variables=template_vars,
        )

    def to_dict(self, config: DaprAgentConfig) -> dict[str, Any]:
        """Convert DaprAgentConfig to a dictionary representation.

        Args:
            config: The DaprAgentConfig to convert

        Returns:
            Dictionary representation of the agent
        """
        llm_config = getattr(config, "llm_config", {})
        tool_defs = getattr(config, "tool_definitions", [])

        return {
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
        }

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
            A Dapr Agent instance (AssistantAgent or ReActAgent)

        Raises:
            ConversionError: If agent creation fails
        """
        try:
            from dapr_agents import AssistantAgent
            from dapr_agents import tool as dapr_tool

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
            agent_type = getattr(config, "agent_type", DaprAgentType.ASSISTANT_AGENT.value)

            if agent_type == DaprAgentType.REACT_AGENT.value:
                from dapr_agents import ReActAgent

                return ReActAgent(
                    name=config.name,
                    role=config.role or config.name,
                    instructions=config.instructions,
                    tools=decorated_tools,
                )
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
                f"Failed to import Dapr Agents: {e}. "
                "Make sure dapr-agents is installed.",
                config,
            ) from e
        except Exception as e:
            raise ConversionError(
                f"Failed to create Dapr Agent: {e}",
                config,
            ) from e

    def _determine_agent_type(self, component: OASAgent) -> DaprAgentType:
        """Determine the appropriate Dapr agent type for an OAS Agent."""
        # Check metadata for explicit type
        if component.metadata:
            explicit_type = component.metadata.get("dapr_agent_type")
            if explicit_type:
                try:
                    return DaprAgentType(explicit_type)
                except ValueError:
                    pass

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
        """Build OAS inputs from Dapr config."""
        inputs: list[dict[str, Any]] = []
        input_vars = getattr(config, "input_variables", [])

        for var in input_vars:
            inputs.append({
                "title": var,
                "type": "string",
            })

        return inputs

