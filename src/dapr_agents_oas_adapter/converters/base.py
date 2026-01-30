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
    """Exception raised when a conversion fails.

    Provides detailed error information including:
    - The component that caused the error
    - A suggestion for how to fix the error
    - The underlying cause if available
    - Component name and type for easier debugging
    """

    def __init__(
        self,
        message: str,
        component: Any = None,
        *,
        suggestion: str | None = None,
        caused_by: Exception | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            message: Error message describing what went wrong
            component: The component that failed to convert
            suggestion: Optional actionable suggestion for fixing the error
            caused_by: Optional underlying exception that caused this error
        """
        self.component = component
        self.suggestion = suggestion
        self.caused_by = caused_by
        self._message = message

        # Build enhanced error message
        full_message = self._build_message(message)
        super().__init__(full_message)

    def _build_message(self, message: str) -> str:
        """Build a detailed error message with context.

        Args:
            message: The base error message

        Returns:
            Enhanced message with component info and suggestion
        """
        parts = [message]

        # Add component context
        component_info = self._get_component_info()
        if component_info:
            parts.append(f"Component: {component_info}")

        # Add suggestion
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")

        # Add cause info
        if self.caused_by:
            parts.append(f"Caused by: {type(self.caused_by).__name__}: {self.caused_by}")

        return " | ".join(parts)

    def _get_component_info(self) -> str:
        """Extract useful information from the component.

        Returns:
            String describing the component, or empty string
        """
        if self.component is None:
            return ""

        # Try to get name and type
        comp_name = None
        comp_type = None

        if isinstance(self.component, dict):
            comp_name = self.component.get("name")
            comp_type = self.component.get("component_type") or self.component.get("type")
        else:
            comp_name = getattr(self.component, "name", None)
            comp_type = getattr(self.component, "component_type", None)
            if comp_type is None:
                comp_type = type(self.component).__name__

        parts = []
        if comp_type:
            parts.append(f"type={comp_type}")
        if comp_name:
            parts.append(f"name={comp_name}")

        return ", ".join(parts) if parts else str(type(self.component).__name__)

    @property
    def base_message(self) -> str:
        """Get the original message without enhancements."""
        return self._message

    def with_suggestion(self, suggestion: str) -> "ConversionError":
        """Create a new error with an added suggestion.

        Args:
            suggestion: The suggestion to add

        Returns:
            New ConversionError with the suggestion
        """
        return ConversionError(
            self._message,
            self.component,
            suggestion=suggestion,
            caused_by=self.caused_by,
        )

    def with_cause(self, cause: Exception) -> "ConversionError":
        """Create a new error with an added cause.

        Args:
            cause: The underlying exception

        Returns:
            New ConversionError with the cause
        """
        return ConversionError(
            self._message,
            self.component,
            suggestion=self.suggestion,
            caused_by=cause,
        )


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

    def convert_from_oas(self, component: Any) -> Any:
        """Convert an OAS component using the appropriate converter.

        Args:
            component: The OAS component (or object) to convert

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
                suggestion="Register a converter for this component type",
            )
        return converter.from_oas(component)  # type: ignore[arg-type]

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
                suggestion="Register a converter for this component type",
            )
        result: Component = converter.to_oas(component)
        return result
