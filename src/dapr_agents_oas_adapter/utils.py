"""Utility functions for OAS <-> Dapr Agents conversion."""

import re
from typing import Any
from uuid import uuid4

from dapr_agents_oas_adapter.types import JSON_SCHEMA_TO_PYTHON, PYTHON_TO_JSON_SCHEMA


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix."""
    uid = str(uuid4())
    if prefix:
        return f"{prefix}_{uid[:8]}"
    return uid


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    pattern = re.compile(r"(?<!^)(?=[A-Z])")
    return pattern.sub("_", name).lower()


def json_schema_to_python_type(schema: dict[str, Any]) -> type:
    """Convert JSON Schema type to Python type."""
    schema_type = schema.get("type", "string")
    if isinstance(schema_type, list):
        # Handle union types - take the first non-null type
        for t in schema_type:
            if t != "null":
                schema_type = t
                break
        else:
            schema_type = "null"

    return JSON_SCHEMA_TO_PYTHON.get(schema_type, str)


def python_type_to_json_schema(py_type: type) -> str:
    """Convert Python type to JSON Schema type string."""
    return PYTHON_TO_JSON_SCHEMA.get(py_type, "string")


def build_json_schema_property(
    title: str,
    py_type: type = str,
    description: str | None = None,
    default: Any = None,
) -> dict[str, Any]:
    """Build a JSON Schema property definition."""
    prop: dict[str, Any] = {
        "title": title,
        "type": python_type_to_json_schema(py_type),
    }
    if description:
        prop["description"] = description
    if default is not None:
        prop["default"] = default
    return prop


def extract_template_variables(template: str) -> list[str]:
    """Extract variable names from a Jinja2-style template string.

    Args:
        template: Template string with {{ variable }} placeholders

    Returns:
        List of variable names found in the template
    """
    pattern = r"\{\{\s*(\w+)\s*\}\}"
    return re.findall(pattern, template)


def render_template(template: str, variables: dict[str, Any]) -> str:
    """Render a simple Jinja2-style template with provided variables.

    Args:
        template: Template string with {{ variable }} placeholders
        variables: Dictionary of variable name to value mappings

    Returns:
        Rendered template string
    """
    result = template
    for key, value in variables.items():
        pattern = r"\{\{\s*" + re.escape(key) + r"\s*\}\}"
        result = re.sub(pattern, str(value), result)
    return result


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def validate_component_id(component_id: str) -> bool:
    """Validate that a component ID is properly formatted."""
    if not component_id:
        return False
    # Allow alphanumeric, underscores, hyphens
    pattern = r"^[a-zA-Z0-9_-]+$"
    return bool(re.match(pattern, component_id))


def get_nested_value(data: dict[str, Any], path: str, default: Any = None) -> Any:
    """Get a nested value from a dictionary using dot notation.

    Args:
        data: Dictionary to extract value from
        path: Dot-separated path (e.g., "llm_config.model_id")
        default: Default value if path not found

    Returns:
        Value at path or default
    """
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def set_nested_value(data: dict[str, Any], path: str, value: Any) -> None:
    """Set a nested value in a dictionary using dot notation.

    Args:
        data: Dictionary to set value in
        path: Dot-separated path (e.g., "llm_config.model_id")
        value: Value to set
    """
    keys = path.split(".")
    current = data
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
