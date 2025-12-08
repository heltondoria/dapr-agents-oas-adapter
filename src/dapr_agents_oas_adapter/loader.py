"""Loader for converting OAS specifications to Dapr Agents components."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pyagentspec import Component
from pyagentspec.agent import Agent as OASAgent
from pyagentspec.flows.flow import Flow
from pyagentspec.serialization import AgentSpecDeserializer

from dapr_agents_oas_adapter.converters.agent import AgentConverter
from dapr_agents_oas_adapter.converters.base import ConversionError
from dapr_agents_oas_adapter.converters.flow import FlowConverter
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    ToolRegistry,
    WorkflowDefinition,
)


class DaprAgentSpecLoader:
    """Loader for converting OAS specifications to Dapr Agents components.

    This class provides methods to load OAS (Open Agent Spec) configurations
    from JSON or YAML format and convert them to Dapr Agents components
    that can be executed.

    Example:
        ```python
        loader = DaprAgentSpecLoader(
            tool_registry={
                "search_tool": search_function,
                "calculator": calc_function,
            }
        )

        # Load from JSON string
        agent = loader.load_json(json_string)

        # Load from YAML file
        workflow = loader.load_yaml_file("workflow.yaml")

        # Create executable Dapr agent
        dapr_agent = loader.create_agent(agent)
        await dapr_agent.start()
        ```
    """

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        """Initialize the loader.

        Args:
            tool_registry: Dictionary mapping tool names to their callable
                          implementations. Required for tools defined in
                          OAS specifications to be executable.
        """
        self._tool_registry = tool_registry or {}
        self._deserializer = AgentSpecDeserializer()
        self._agent_converter = AgentConverter(self._tool_registry)
        self._flow_converter = FlowConverter(self._tool_registry)

    @property
    def tool_registry(self) -> ToolRegistry:
        """Get the current tool registry."""
        return self._tool_registry

    @tool_registry.setter
    def tool_registry(self, registry: ToolRegistry) -> None:
        """Update the tool registry."""
        self._tool_registry = registry
        self._agent_converter.tool_registry = registry
        self._flow_converter.tool_registry = registry

    def register_tool(self, name: str, implementation: Callable[..., Any]) -> None:
        """Register a tool implementation.

        Args:
            name: The tool name as defined in the OAS specification
            implementation: The callable implementation
        """
        self._tool_registry[name] = implementation
        self._agent_converter.tool_registry = self._tool_registry
        self._flow_converter.tool_registry = self._tool_registry

    def load_json(self, json_content: str) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from JSON string.

        Args:
            json_content: JSON string containing the OAS specification

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the JSON cannot be parsed or converted
        """
        try:
            component = self._deserializer.from_json(json_content)
            return self.load_component(component)
        except Exception as e:
            raise ConversionError(f"Failed to load JSON: {e}") from e

    def load_yaml(self, yaml_content: str) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from YAML string.

        Args:
            yaml_content: YAML string containing the OAS specification

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the YAML cannot be parsed or converted
        """
        try:
            component = self._deserializer.from_yaml(yaml_content)
            return self.load_component(component)
        except Exception as e:
            raise ConversionError(f"Failed to load YAML: {e}") from e

    def load_json_file(self, file_path: str | Path) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the file cannot be read or converted
        """
        path = Path(file_path)
        if not path.exists():
            raise ConversionError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        return self.load_json(content)

    def load_yaml_file(self, file_path: str | Path) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the file cannot be read or converted
        """
        path = Path(file_path)
        if not path.exists():
            raise ConversionError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")
        return self.load_yaml(content)

    def load_component(self, component: Component) -> DaprAgentConfig | WorkflowDefinition:
        """Load a PyAgentSpec Component and convert to Dapr format.

        Args:
            component: The OAS Component to convert

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the component type is not supported
        """
        if isinstance(component, OASAgent):
            return self._agent_converter.from_oas(component)
        elif isinstance(component, Flow):
            return self._flow_converter.from_oas(component)
        else:
            raise ConversionError(
                f"Unsupported component type: {type(component).__name__}",
                component,
            )

    def load_dict(self, spec_dict: dict[str, Any]) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a dictionary.

        Args:
            spec_dict: Dictionary containing the OAS specification

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the component type is not supported
        """
        component_type = spec_dict.get("component_type", "")

        if component_type == "Agent":
            return self._agent_converter.from_dict(spec_dict)
        elif component_type == "Flow":
            return self._flow_converter.from_dict(spec_dict)
        else:
            raise ConversionError(f"Unsupported component type: {component_type}")

    def create_agent(
        self,
        config: DaprAgentConfig,
        additional_tools: dict[str, Callable[..., Any]] | None = None,
    ) -> Any:
        """Create an executable Dapr Agent from configuration.

        Args:
            config: The agent configuration
            additional_tools: Additional tool implementations to include

        Returns:
            A Dapr Agent instance (AssistantAgent or ReActAgent)

        Raises:
            ConversionError: If agent creation fails
        """
        tools = {**self._tool_registry}
        if additional_tools:
            tools.update(additional_tools)

        return self._agent_converter.create_dapr_agent(config, tools)

    def create_workflow(
        self,
        workflow_def: WorkflowDefinition,
        task_implementations: dict[str, Callable[..., Any]] | None = None,
    ) -> Callable[..., Any]:
        """Create an executable Dapr workflow from definition.

        Args:
            workflow_def: The workflow definition
            task_implementations: Task implementations for workflow activities

        Returns:
            A workflow function that can be registered with Dapr

        Raises:
            ConversionError: If workflow creation fails
        """
        return self._flow_converter.create_dapr_workflow(workflow_def, task_implementations)

    def generate_workflow_code(self, workflow_def: WorkflowDefinition) -> str:
        """Generate Python code for a Dapr workflow.

        Args:
            workflow_def: The workflow definition

        Returns:
            Python code string that can be saved and executed
        """
        return self._flow_converter.generate_workflow_code(workflow_def)

    def load_and_create_agent(
        self,
        json_or_yaml: str,
        is_yaml: bool = False,
        additional_tools: dict[str, Callable[..., Any]] | None = None,
    ) -> Any:
        """Convenience method to load and create an agent in one step.

        Args:
            json_or_yaml: The specification string
            is_yaml: Whether the input is YAML (default: JSON)
            additional_tools: Additional tool implementations

        Returns:
            A Dapr Agent instance ready to start

        Raises:
            ConversionError: If loading or creation fails
        """
        config = self.load_yaml(json_or_yaml) if is_yaml else self.load_json(json_or_yaml)

        if not isinstance(config, DaprAgentConfig):
            raise ConversionError(
                "Expected Agent specification, got Flow. "
                "Use load_and_create_workflow for Flow specifications."
            )

        return self.create_agent(config, additional_tools)

    def load_and_create_workflow(
        self,
        json_or_yaml: str,
        is_yaml: bool = False,
        task_implementations: dict[str, Callable[..., Any]] | None = None,
    ) -> Callable[..., Any]:
        """Convenience method to load and create a workflow in one step.

        Args:
            json_or_yaml: The specification string
            is_yaml: Whether the input is YAML (default: JSON)
            task_implementations: Task implementations

        Returns:
            A workflow function ready to register

        Raises:
            ConversionError: If loading or creation fails
        """
        config = self.load_yaml(json_or_yaml) if is_yaml else self.load_json(json_or_yaml)

        if not isinstance(config, WorkflowDefinition):
            raise ConversionError(
                "Expected Flow specification, got Agent. "
                "Use load_and_create_agent for Agent specifications."
            )

        return self.create_workflow(config, task_implementations)
