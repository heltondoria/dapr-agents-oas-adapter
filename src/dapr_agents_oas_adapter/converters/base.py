"""Base converter class for OAS <-> Dapr Agents conversion."""

from abc import ABC, abstractmethod
from typing import Any

from pyagentspec import Component

from dapr_agents_oas_adapter.types import ToolRegistry


class ComponentConverter[OASType: Component, DaprType](ABC):
    """Abstract base class for component converters.

    This class defines the interface for bidirectional conversion between
    Open Agent Spec (OAS) components and Dapr Agents components.

    Type Parameters:
        OASType: The OAS component type (e.g., Agent, Tool, Flow)
        DaprType: The Dapr Agents type (e.g., AssistantAgent, Callable, Workflow)
    """

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        """Initialize the converter.

        Args:
            tool_registry: Optional dictionary mapping tool names to their
                          callable implementations.
        """
        self._tool_registry = tool_registry or {}

    @property
    def tool_registry(self) -> ToolRegistry:
        """Get the tool registry."""
        return self._tool_registry

    @tool_registry.setter
    def tool_registry(self, registry: ToolRegistry) -> None:
        """Set the tool registry."""
        self._tool_registry = registry

    @abstractmethod
    def from_oas(self, component: OASType) -> DaprType:
        """Convert an OAS component to a Dapr Agents component.

        Args:
            component: The OAS component to convert

        Returns:
            The equivalent Dapr Agents component

        Raises:
            ConversionError: If the conversion fails
        """
        ...

    @abstractmethod
    def to_oas(self, component: DaprType) -> OASType:
        """Convert a Dapr Agents component to an OAS component.

        Args:
            component: The Dapr Agents component to convert

        Returns:
            The equivalent OAS component

        Raises:
            ConversionError: If the conversion fails
        """
        ...

    @abstractmethod
    def can_convert(self, component: Any) -> bool:
        """Check if this converter can handle the given component.

        Args:
            component: The component to check

        Returns:
            True if this converter can handle the component
        """
        ...

    def validate_oas_component(self, component: OASType) -> None:
        """Validate an OAS component before conversion.

        Args:
            component: The OAS component to validate

        Raises:
            ValidationError: If the component is invalid
        """
        if not hasattr(component, "id"):
            raise ValidationError("OAS component must have an 'id' attribute")
        if not hasattr(component, "name"):
            raise ValidationError("OAS component must have a 'name' attribute")

    def get_component_metadata(self, component: OASType) -> dict[str, Any]:
        """Extract metadata from an OAS component.

        Args:
            component: The OAS component

        Returns:
            Dictionary of metadata
        """
        metadata: dict[str, Any] = {}
        if hasattr(component, "metadata") and component.metadata:
            metadata = dict(component.metadata)
        if hasattr(component, "description") and component.description:
            metadata["description"] = component.description
        return metadata


class ConversionError(Exception):
    """Exception raised when a conversion fails."""

    def __init__(self, message: str, component: Any = None) -> None:
        """Initialize the error.

        Args:
            message: Error message
            component: The component that failed to convert
        """
        super().__init__(message)
        self.component = component


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, message: str, field: str | None = None) -> None:
        """Initialize the error.

        Args:
            message: Error message
            field: The field that failed validation
        """
        super().__init__(message)
        self.field = field


class ConverterRegistry:
    """Registry for managing multiple component converters."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._converters: list[ComponentConverter[Any, Any]] = []

    def register(self, converter: ComponentConverter[Any, Any]) -> None:
        """Register a converter.

        Args:
            converter: The converter to register
        """
        self._converters.append(converter)

    def get_converter(self, component: Any) -> ComponentConverter[Any, Any] | None:
        """Get a converter that can handle the given component.

        Args:
            component: The component to find a converter for

        Returns:
            A converter that can handle the component, or None
        """
        for converter in self._converters:
            if converter.can_convert(component):
                return converter
        return None

    def convert_from_oas(self, component: Component) -> Any:
        """Convert an OAS component using the appropriate converter.

        Args:
            component: The OAS component to convert

        Returns:
            The converted Dapr Agents component

        Raises:
            ConversionError: If no suitable converter is found
        """
        converter = self.get_converter(component)
        if converter is None:
            raise ConversionError(
                f"No converter found for component type: {type(component).__name__}",
                component,
            )
        return converter.from_oas(component)

    def convert_to_oas(self, component: Any) -> Component:
        """Convert a Dapr component using the appropriate converter.

        Args:
            component: The Dapr Agents component to convert

        Returns:
            The converted OAS component

        Raises:
            ConversionError: If no suitable converter is found
        """
        converter = self.get_converter(component)
        if converter is None:
            raise ConversionError(
                f"No converter found for component type: {type(component).__name__}",
                component,
            )
        result: Component = converter.to_oas(component)
        return result

