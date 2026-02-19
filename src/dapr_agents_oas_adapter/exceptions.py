"""Exception hierarchy for dapr-agents-oas-adapter."""

from typing import Any


class DaprAgentsOasAdapterError(Exception):
    """Base exception for all dapr-agents-oas-adapter errors.

    Consumers can catch this single exception type to handle
    all errors raised by this library.
    """


class ConversionError(DaprAgentsOasAdapterError):
    """Exception raised when a conversion fails.

    Provides detailed error information including:
    - The component that caused the error
    - A suggestion for how to fix the error
    - Component name and type for easier debugging
    """

    def __init__(
        self,
        message: str,
        component: Any = None,
        *,
        suggestion: str | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            message: Error message describing what went wrong
            component: The component that failed to convert
            suggestion: Optional actionable suggestion for fixing the error
        """
        self.component = component
        self.suggestion = suggestion
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
            New ConversionError with the suggestion (preserves any existing __cause__)
        """
        new_error = ConversionError(
            self._message,
            self.component,
            suggestion=suggestion,
        )
        if self.__cause__ is not None:
            new_error.__cause__ = self.__cause__
            new_error.__suppress_context__ = True
        return new_error

    def with_cause(self, cause: Exception) -> "ConversionError":
        """Create a new error with an added cause.

        Args:
            cause: The underlying exception

        Returns:
            New ConversionError with the cause set via __cause__
        """
        new_error = ConversionError(
            self._message,
            self.component,
            suggestion=self.suggestion,
        )
        new_error.__cause__ = cause
        new_error.__suppress_context__ = True
        return new_error


class ValidationError(DaprAgentsOasAdapterError):
    """Exception raised when validation fails."""

    def __init__(self, message: str, field_path: str | None = None) -> None:
        """Initialize the error.

        Args:
            message: Error message
            field_path: The field path that failed validation
        """
        super().__init__(message)
        self.field_path = field_path
