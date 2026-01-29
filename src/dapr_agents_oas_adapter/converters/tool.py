"""Tool converter for OAS <-> Dapr Agents."""

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pyagentspec.tools import RemoteTool, ServerTool, Tool

# MCPTool may not be available in all versions
try:
    from pyagentspec.tools import MCPTool  # type: ignore[attr-defined]
except ImportError:
    MCPTool = None  # type: ignore[misc, assignment]

from dapr_agents_oas_adapter.converters.base import (
    ComponentConverter,
    ConversionError,
)
from dapr_agents_oas_adapter.types import (
    PYTHON_TO_JSON_SCHEMA,
    NamedCallable,
    PropertySchema,
    ToolDefinition,
    ToolRegistry,
)
from dapr_agents_oas_adapter.utils import build_json_schema_property, generate_id


class ToolConverter(ComponentConverter[Tool, ToolDefinition]):
    """Converter for OAS Tool <-> Dapr tool definition.

    Supports conversion between OAS Tool types (ServerTool, RemoteTool, MCPTool)
    and Dapr Agents tool definitions.
    """

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        """Initialize the converter with an optional tool registry.

        Args:
            tool_registry: Dictionary mapping tool names to their implementations
        """
        super().__init__(tool_registry)

    def from_oas(self, component: Tool) -> ToolDefinition:
        """Convert an OAS Tool to a Dapr ToolDefinition.

        Args:
            component: The OAS Tool to convert

        Returns:
            ToolDefinition with equivalent settings

        Raises:
            ConversionError: If the tool type is not supported
        """
        self.validate_oas_component(component)

        # Get tool implementation from registry if available
        implementation = self._tool_registry.get(component.name)

        # Extract inputs and outputs
        inputs = self._extract_properties(getattr(component, "inputs", []))
        outputs = self._extract_properties(getattr(component, "outputs", []))

        return ToolDefinition(
            name=component.name,
            description=component.description or "",
            inputs=inputs,
            outputs=outputs,
            implementation=implementation,
        )

    def to_oas(self, component: ToolDefinition) -> Tool:
        """Convert a Dapr ToolDefinition to an OAS Tool.

        Args:
            component: The Dapr ToolDefinition to convert

        Returns:
            OAS ServerTool with equivalent settings
        """
        tool_id = generate_id("tool")

        return ServerTool(
            id=tool_id,
            name=component.name,
            description=component.description,
        )

    def can_convert(self, component: Any) -> bool:
        """Check if this converter can handle the given component.

        Args:
            component: The component to check

        Returns:
            True if this converter can handle the component
        """
        tool_types = [Tool, ServerTool, RemoteTool]
        if MCPTool is not None:
            tool_types.append(MCPTool)
        if isinstance(component, tuple(tool_types)):
            return True
        if isinstance(component, ToolDefinition):
            return True
        if isinstance(component, dict):
            comp_type = component.get("component_type", "")
            return comp_type in ("ServerTool", "RemoteTool", "MCPTool")
        return False

    def from_callable(self, func: Callable[..., Any], name: str | None = None) -> ToolDefinition:
        """Create a ToolDefinition from a Python callable.

        Args:
            func: The callable to convert
            name: Optional name override (defaults to function name)

        Returns:
            ToolDefinition representing the callable
        """
        func_name = self._callable_name(func, override=name)
        description = inspect.getdoc(func) or f"Tool: {func_name}"

        # Extract input parameters from signature
        inputs = self._extract_inputs_from_callable(func)

        # Try to extract return type
        outputs = self._extract_outputs_from_callable(func)

        return ToolDefinition(
            name=func_name,
            description=description.strip(),
            inputs=inputs,
            outputs=outputs,
            implementation=func,
        )

    def to_callable(self, tool_def: ToolDefinition) -> Callable[..., Any] | None:
        """Get the callable implementation for a ToolDefinition.

        Args:
            tool_def: The ToolDefinition

        Returns:
            The callable implementation, or None if not available
        """
        if tool_def.implementation:
            return tool_def.implementation
        return self._tool_registry.get(tool_def.name)

    def from_dict(self, tool_dict: dict[str, Any]) -> ToolDefinition:
        """Convert a dictionary representation to ToolDefinition.

        Args:
            tool_dict: Dictionary with tool configuration

        Returns:
            ToolDefinition with the converted settings
        """
        name = tool_dict.get("name", "")
        implementation = self._tool_registry.get(name)

        # Extract MCP transport config if this is an MCPTool
        # Always set mcp_transport for MCPTool to preserve component type,
        # even if client_transport is empty
        mcp_transport = None
        if tool_dict.get("component_type") == "MCPTool":
            client_transport = tool_dict.get("client_transport", {})
            mcp_transport = {
                "type": client_transport.get("component_type", "SSETransport"),
                "url": client_transport.get("url"),
                "headers": client_transport.get("headers"),
                "session_parameters": client_transport.get("session_parameters"),
            }

        return ToolDefinition(
            name=name,
            description=tool_dict.get("description", ""),
            inputs=tool_dict.get("inputs", []),
            outputs=tool_dict.get("outputs", []),
            implementation=implementation,
            mcp_transport=mcp_transport,
        )

    def to_dict(self, tool_def: ToolDefinition) -> dict[str, Any]:
        """Convert ToolDefinition to a dictionary representation.

        Args:
            tool_def: The ToolDefinition to convert

        Returns:
            Dictionary representation of the tool
        """
        # Export as MCPTool if there's transport config
        if tool_def.mcp_transport:
            transport = tool_def.mcp_transport
            client_transport: dict[str, Any] = {
                "component_type": transport.get("type", "SSETransport"),
                "id": generate_id("transport"),
                "name": f"{tool_def.name}_transport",
            }
            # Add optional transport fields (use 'is not None' to preserve empty values)
            if transport.get("url") is not None:
                client_transport["url"] = transport["url"]
            if transport.get("headers") is not None:
                client_transport["headers"] = transport["headers"]
            if transport.get("session_parameters") is not None:
                client_transport["session_parameters"] = transport["session_parameters"]

            return {
                "component_type": "MCPTool",
                "id": generate_id("tool"),
                "name": tool_def.name,
                "description": tool_def.description,
                "inputs": tool_def.inputs,
                "outputs": tool_def.outputs,
                "client_transport": client_transport,
            }

        return {
            "component_type": "ServerTool",
            "id": generate_id("tool"),
            "name": tool_def.name,
            "description": tool_def.description,
            "inputs": tool_def.inputs,
            "outputs": tool_def.outputs,
        }

    def create_dapr_tool(self, tool_def: ToolDefinition) -> NamedCallable:
        """Create a Dapr-compatible tool function from a ToolDefinition.

        This wraps the tool implementation with Dapr's @tool decorator pattern.

        Args:
            tool_def: The ToolDefinition to create a tool from

        Returns:
            A callable that can be used as a Dapr tool

        Raises:
            ConversionError: If no implementation is available
        """
        implementation = tool_def.implementation or self._tool_registry.get(tool_def.name)

        if implementation is None:
            raise ConversionError(
                f"No implementation found for tool: {tool_def.name}",
                tool_def,
                suggestion="Provide an implementation via tool_def.implementation or tool_registry",
            )

        # Wrap with metadata for Dapr
        def tool_wrapper(*args: Any, **kwargs: Any) -> Any:
            return implementation(*args, **kwargs)

        # Set function metadata
        tool_wrapper.__name__ = tool_def.name
        tool_wrapper.__doc__ = tool_def.description
        tool_wrapper.__annotations__ = self._build_annotations_from_schema(tool_def.inputs)

        return tool_wrapper

    @staticmethod
    def _callable_name(func: Callable[..., Any], override: str | None = None) -> str:
        """Get a callable name with a stable, testable fallback."""
        if override:
            return override
        return getattr(func, "__name__", type(func).__name__)

    def _extract_properties(self, props: list[Any]) -> list[PropertySchema]:
        """Extract property schemas from OAS properties."""
        result: list[PropertySchema] = []
        for prop in props:
            if isinstance(prop, dict):
                result.append(prop)
            elif hasattr(prop, "model_dump"):
                result.append(prop.model_dump())
            elif hasattr(prop, "__dict__"):
                result.append(dict(prop.__dict__))
        return result

    def _extract_inputs_from_callable(self, func: Callable[..., Any]) -> list[PropertySchema]:
        """Extract input schemas from a callable's signature."""
        inputs: list[PropertySchema] = []
        sig = inspect.signature(func)
        type_hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = type_hints.get(param_name, str)
            default = None if param.default is inspect.Parameter.empty else param.default

            inputs.append(
                build_json_schema_property(
                    title=param_name,
                    py_type=param_type if isinstance(param_type, type) else str,
                    default=default,
                )
            )

        return inputs

    def _extract_outputs_from_callable(self, func: Callable[..., Any]) -> list[PropertySchema]:
        """Extract output schema from a callable's return type."""
        type_hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
        return_type = type_hints.get("return", str)

        if return_type is type(None):
            return []

        json_type = PYTHON_TO_JSON_SCHEMA.get(
            return_type if isinstance(return_type, type) else str,
            "string",
        )

        return [{"title": "result", "type": json_type}]

    def _build_annotations_from_schema(self, inputs: list[PropertySchema]) -> dict[str, type]:
        """Build Python annotations from JSON Schema properties."""
        from dapr_agents_oas_adapter.utils import json_schema_to_python_type

        annotations: dict[str, type] = {}
        for prop in inputs:
            title = prop.get("title", "")
            if title:
                annotations[title] = json_schema_to_python_type(prop)
        return annotations


class MCPToolConverter(ToolConverter):
    """Specialized converter for MCP tools.

    Handles conversion of MCPTool components which connect to
    Model Context Protocol servers.
    """

    def from_oas(self, component: Tool) -> ToolDefinition:
        """Convert an OAS MCPTool to a Dapr ToolDefinition.

        Args:
            component: The OAS MCPTool to convert

        Returns:
            ToolDefinition with MCP-specific settings
        """
        base_def = super().from_oas(component)

        # Extract MCP-specific configuration from client_transport
        # Works regardless of whether MCPTool class is available
        # Use 'is not None' to preserve MCPTool type even with empty transport
        transport = getattr(component, "client_transport", None)
        if transport is not None:
            base_def.mcp_transport = self._extract_transport_config(transport)

        return base_def

    def _extract_transport_config(self, transport: Any) -> dict[str, Any]:
        """Extract MCP transport configuration.

        Args:
            transport: The transport object (SSETransport, etc.)

        Returns:
            Dictionary with transport configuration
        """
        config: dict[str, Any] = {}

        # Get transport type from class name or component_type
        transport_type = getattr(transport, "component_type", None)
        if not transport_type:
            transport_type = type(transport).__name__
        config["type"] = transport_type

        if hasattr(transport, "url"):
            config["url"] = transport.url
        if hasattr(transport, "headers"):
            config["headers"] = transport.headers
        if hasattr(transport, "session_parameters"):
            config["session_parameters"] = transport.session_parameters

        return config
