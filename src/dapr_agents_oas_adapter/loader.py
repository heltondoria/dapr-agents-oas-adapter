"""Loader for converting OAS specifications to Dapr Agents components."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dapr_agents_oas_adapter.validation import ValidationResult

from pyagentspec.agent import Agent as OASAgent
from pyagentspec.flows.flow import Flow
from pyagentspec.serialization import AgentSpecDeserializer

from dapr_agents_oas_adapter.converters.agent import AgentConverter
from dapr_agents_oas_adapter.converters.base import ConversionError
from dapr_agents_oas_adapter.converters.flow import FlowConverter
from dapr_agents_oas_adapter.logging import get_logger, log_operation
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    NamedCallable,
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
        self._logger = get_logger("DaprAgentSpecLoader")

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
        except ConversionError:
            raise
        except Exception as e:
            raise ConversionError(
                "Failed to load JSON",
                suggestion="Ensure the JSON is valid and follows the OAS schema",
                caused_by=e,
            ) from e

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
        except ConversionError:
            raise
        except Exception as e:
            raise ConversionError(
                "Failed to load YAML",
                suggestion="Ensure the YAML is valid and follows the OAS schema",
                caused_by=e,
            ) from e

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
            raise ConversionError(
                f"File not found: {file_path}",
                suggestion="Check the file path exists and is accessible",
            )

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
            raise ConversionError(
                f"File not found: {file_path}",
                suggestion="Check the file path exists and is accessible",
            )

        content = path.read_text(encoding="utf-8")
        return self.load_yaml(content)

    def load_component(self, component: Any) -> DaprAgentConfig | WorkflowDefinition:
        """Load a PyAgentSpec Component and convert to Dapr format.

        Args:
            component: The OAS Component (or object) to convert

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            ConversionError: If the component type is not supported
        """
        component_type = type(component).__name__
        component_name = getattr(component, "name", None) or getattr(component, "id", None)

        with log_operation(
            "load_component",
            self._logger,
            component_type=component_type,
            component_name=component_name,
        ):
            if isinstance(component, OASAgent):
                agent_config = self._agent_converter.from_oas(component)
                self._logger.debug("agent_converted", agent_name=agent_config.name)
                return agent_config
            elif isinstance(component, Flow):
                workflow_def = self._flow_converter.from_oas(component)
                self._logger.debug(
                    "workflow_converted",
                    workflow_name=workflow_def.name,
                    task_count=len(workflow_def.tasks),
                )
                return workflow_def
            else:
                raise ConversionError(
                    f"Unsupported component type: {component_type}",
                    component,
                    suggestion="Only Agent and Flow component types are supported",
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
        component_name = spec_dict.get("name") or spec_dict.get("id")

        with log_operation(
            "load_dict",
            self._logger,
            component_type=component_type,
            component_name=component_name,
        ):
            if component_type == "Agent":
                agent_config = self._agent_converter.from_dict(spec_dict)
                self._logger.debug("agent_loaded_from_dict", agent_name=agent_config.name)
                return agent_config
            elif component_type == "Flow":
                workflow_def = self._flow_converter.from_dict(spec_dict)
                self._logger.debug(
                    "workflow_loaded_from_dict",
                    workflow_name=workflow_def.name,
                    task_count=len(workflow_def.tasks),
                )
                return workflow_def
            else:
                raise ConversionError(
                    f"Unsupported component type: {component_type}",
                    spec_dict,
                    suggestion="Set 'component_type' to 'Agent' or 'Flow' in the specification",
                )

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
        with log_operation(
            "create_agent",
            self._logger,
            agent_name=config.name,
            agent_type=config.agent_type,
            tool_count=len(config.tools),
        ):
            tools = {**self._tool_registry}
            if additional_tools:
                tools.update(additional_tools)
                self._logger.debug(
                    "additional_tools_merged",
                    additional_tool_count=len(additional_tools),
                )

            return self._agent_converter.create_dapr_agent(config, tools)

    def create_workflow(
        self,
        workflow_def: WorkflowDefinition,
        task_implementations: dict[str, Callable[..., Any]] | None = None,
    ) -> NamedCallable:
        """Create an executable Dapr workflow from definition.

        Args:
            workflow_def: The workflow definition
            task_implementations: Task implementations for workflow activities

        Returns:
            A workflow function that can be registered with Dapr

        Raises:
            ConversionError: If workflow creation fails
        """
        with log_operation(
            "create_workflow",
            self._logger,
            workflow_name=workflow_def.name,
            task_count=len(workflow_def.tasks),
            edge_count=len(workflow_def.edges),
        ):
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
                "Expected Agent specification, got Flow",
                config,
                suggestion="Use load_and_create_workflow() for Flow specifications",
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
                "Expected Flow specification, got Agent",
                config,
                suggestion="Use load_and_create_agent() for Agent specifications",
            )

        return self.create_workflow(config, task_implementations)


class StrictLoader:
    """A validating wrapper around DaprAgentSpecLoader.

    This loader validates OAS specifications against schema rules before
    attempting conversion, providing early error detection and better
    error messages.

    Example:
        ```python
        from dapr_agents_oas_adapter import StrictLoader

        # Create a strict loader
        loader = StrictLoader()

        # This will validate the dict before loading
        config = loader.load_dict(spec_dict)

        # Invalid specs will raise OASSchemaValidationError
        try:
            config = loader.load_dict(invalid_dict)
        except OASSchemaValidationError as e:
            for issue in e.issues:
                print(issue)
        ```
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
        *,
        warn_on_unknown_fields: bool = True,
    ) -> None:
        """Initialize the strict loader.

        Args:
            tool_registry: Dictionary mapping tool names to their callable
                          implementations.
            warn_on_unknown_fields: If True, unknown fields trigger warnings
        """
        from dapr_agents_oas_adapter.validation import OASSchemaValidator

        self._loader = DaprAgentSpecLoader(tool_registry)
        self._validator = OASSchemaValidator()
        self._warn_on_unknown_fields = warn_on_unknown_fields
        self._logger = get_logger("StrictLoader")

    @property
    def loader(self) -> DaprAgentSpecLoader:
        """Get the underlying loader."""
        return self._loader

    @property
    def tool_registry(self) -> ToolRegistry:
        """Get the current tool registry."""
        return self._loader.tool_registry

    @tool_registry.setter
    def tool_registry(self, registry: ToolRegistry) -> None:
        """Update the tool registry."""
        self._loader.tool_registry = registry

    def register_tool(self, name: str, implementation: Callable[..., Any]) -> None:
        """Register a tool implementation.

        Args:
            name: The tool name as defined in the OAS specification
            implementation: The callable implementation
        """
        self._loader.register_tool(name, implementation)

    def load_dict(
        self,
        spec_dict: dict[str, Any],
        *,
        validate: bool = True,
    ) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from a dictionary with validation.

        Args:
            spec_dict: Dictionary containing the OAS specification
            validate: Whether to validate before loading (default: True)

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            OASSchemaValidationError: If validation fails
            ConversionError: If the component type is not supported
        """
        if validate:
            with log_operation(
                "strict_validate_dict",
                self._logger,
                component_type=spec_dict.get("component_type"),
            ):
                self._validator.validate_component(spec_dict, raise_on_error=True)

        return self._loader.load_dict(spec_dict)

    def validate_dict(self, spec_dict: dict[str, Any]) -> ValidationResult:
        """Validate a dictionary without loading it.

        Args:
            spec_dict: Dictionary containing the OAS specification

        Returns:
            ValidationResult with all issues found
        """
        return self._validator.validate_component(spec_dict)

    def load_json(
        self,
        json_content: str,
        *,
        validate: bool = True,
    ) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from JSON string with optional validation.

        Note: Full validation is only available for load_dict. This method
        parses JSON then delegates to load_dict for validation.

        Args:
            json_content: JSON string containing the OAS specification
            validate: Whether to validate before loading (default: True)

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            OASSchemaValidationError: If validation fails
            ConversionError: If the JSON cannot be parsed or converted
        """
        import json

        try:
            spec_dict = json.loads(json_content)
        except json.JSONDecodeError as e:
            raise ConversionError(
                "Invalid JSON",
                suggestion="Ensure the JSON syntax is valid",
                caused_by=e,
            ) from e

        if validate and isinstance(spec_dict, dict):
            return self.load_dict(spec_dict, validate=True)

        return self._loader.load_json(json_content)

    def load_yaml(
        self,
        yaml_content: str,
        *,
        validate: bool = True,
    ) -> DaprAgentConfig | WorkflowDefinition:
        """Load an OAS specification from YAML string with optional validation.

        Note: Full validation is only available for load_dict. This method
        parses YAML then delegates to load_dict for validation.

        Args:
            yaml_content: YAML string containing the OAS specification
            validate: Whether to validate before loading (default: True)

        Returns:
            DaprAgentConfig for Agent components, WorkflowDefinition for Flows

        Raises:
            OASSchemaValidationError: If validation fails
            ConversionError: If the YAML cannot be parsed or converted
        """
        try:
            import yaml

            spec_dict = yaml.safe_load(yaml_content)
        except Exception as e:
            raise ConversionError(
                "Invalid YAML",
                suggestion="Ensure the YAML syntax is valid",
                caused_by=e,
            ) from e

        if validate and isinstance(spec_dict, dict):
            return self.load_dict(spec_dict, validate=True)

        return self._loader.load_yaml(yaml_content)

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
        return self._loader.create_agent(config, additional_tools)

    def create_workflow(
        self,
        workflow_def: WorkflowDefinition,
        task_implementations: dict[str, Callable[..., Any]] | None = None,
    ) -> NamedCallable:
        """Create an executable Dapr workflow from definition.

        Args:
            workflow_def: The workflow definition
            task_implementations: Task implementations for workflow activities

        Returns:
            A workflow function that can be registered with Dapr

        Raises:
            ConversionError: If workflow creation fails
        """
        return self._loader.create_workflow(workflow_def, task_implementations)
