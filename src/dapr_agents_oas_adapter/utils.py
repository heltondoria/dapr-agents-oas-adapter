"""Utility functions for OAS <-> Dapr Agents conversion."""

import re
from random import Random
from typing import Any
from uuid import uuid4

from dapr_agents_oas_adapter.types import JSON_SCHEMA_TO_PYTHON, PYTHON_TO_JSON_SCHEMA


class IDGenerator:
    """Deterministic ID generator with optional seeding for reproducibility.

    Supports seeded mode for deterministic IDs (useful for testing) and
    default mode using UUID4 for production.

    Usage:
        # Production (random IDs)
        id_gen = IDGenerator()
        id1 = id_gen.generate("agent")  # agent_a1b2c3d4

        # Testing (deterministic IDs)
        id_gen = IDGenerator(seed=42)
        id1 = id_gen.generate("agent")  # Same every time with seed=42

        # Reset for new test
        id_gen.reset(seed=42)
    """

    _instance: "IDGenerator | None" = None

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the ID generator.

        Args:
            seed: Optional seed for deterministic ID generation.
                  If None, uses UUID4 for true randomness.
        """
        self._seed = seed
        # S311: Random is intentionally used for non-cryptographic ID generation
        self._random: Random | None = Random(seed) if seed is not None else None  # noqa: S311
        self._counter = 0

    @classmethod
    def get_instance(cls) -> "IDGenerator":
        """Get the global IDGenerator instance (singleton pattern)."""
        if cls._instance is None:
            cls._instance = IDGenerator()
        return cls._instance

    @classmethod
    def set_instance(cls, generator: "IDGenerator") -> None:
        """Set the global IDGenerator instance (for testing)."""
        cls._instance = generator

    @classmethod
    def reset_instance(cls, seed: int | None = None) -> "IDGenerator":
        """Reset the global instance with optional seed."""
        cls._instance = IDGenerator(seed=seed)
        return cls._instance

    def reset(self, seed: int | None = None) -> None:
        """Reset the generator with optional new seed.

        Args:
            seed: Optional seed for deterministic ID generation.
        """
        self._seed = seed
        # S311: Random is intentionally used for non-cryptographic ID generation
        self._random = Random(seed) if seed is not None else None  # noqa: S311
        self._counter = 0

    def generate(self, prefix: str = "") -> str:
        """Generate a unique identifier with optional prefix.

        Args:
            prefix: Optional prefix for the ID (e.g., "agent", "flow", "node")

        Returns:
            Generated ID string. Format depends on mode:
            - Seeded: "{prefix}_{8_hex_chars}" or "{counter}_{8_hex_chars}"
            - Random: "{prefix}_{8_uuid_chars}" or full UUID
        """
        if self._random is not None:
            # Deterministic mode: use seeded random
            hex_chars = f"{self._random.getrandbits(32):08x}"
            self._counter += 1
        else:
            # Random mode: use UUID4
            hex_chars = str(uuid4())[:8]

        if prefix:
            return f"{prefix}_{hex_chars}"
        return hex_chars if self._random is not None else str(uuid4())

    @property
    def is_seeded(self) -> bool:
        """Check if the generator is in seeded (deterministic) mode."""
        return self._random is not None

    @property
    def seed(self) -> int | None:
        """Get the current seed value."""
        return self._seed


def generate_id(prefix: str = "") -> str:
    """Generate a unique identifier with optional prefix.

    This function uses the global IDGenerator instance. For testing with
    deterministic IDs, use IDGenerator.reset_instance(seed=42) first.

    Args:
        prefix: Optional prefix for the ID

    Returns:
        Generated ID string
    """
    return IDGenerator.get_instance().generate(prefix)


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
