"""State schema builder for OAS <-> Dapr Agents conversion."""

from typing import Any

from dapr_agents_oas_adapter.types import PropertySchema
from dapr_agents_oas_adapter.utils import json_schema_to_python_type


class StateSchemaBuilder:
    """Builds state schemas from OAS input/output definitions."""

    @staticmethod
    def build_from_properties(properties: list[PropertySchema]) -> dict[str, type]:
        """Build a state schema dictionary from OAS properties.

        Args:
            properties: List of OAS property schemas (inputs or outputs)

        Returns:
            Dictionary mapping property names to Python types
        """
        schema: dict[str, type] = {}
        for prop in properties:
            title = prop.get("title", "")
            if title:
                py_type = json_schema_to_python_type(prop)
                schema[title] = py_type
        return schema

    @staticmethod
    def build_typed_dict_class(
        name: str,
        properties: list[PropertySchema],
    ) -> type:
        """Build a TypedDict class from OAS properties.

        Args:
            name: Name for the TypedDict class
            properties: List of OAS property schemas

        Returns:
            A dynamically created TypedDict class
        """
        from typing import TypedDict

        schema = StateSchemaBuilder.build_from_properties(properties)
        # Create TypedDict dynamically using the proper constructor
        return TypedDict(name, schema)  # type: ignore[operator]

    @staticmethod
    def extract_defaults(properties: list[PropertySchema]) -> dict[str, Any]:
        """Extract default values from OAS properties.

        Args:
            properties: List of OAS property schemas

        Returns:
            Dictionary mapping property names to their default values
        """
        defaults: dict[str, Any] = {}
        for prop in properties:
            title = prop.get("title", "")
            if title and "default" in prop:
                defaults[title] = prop["default"]
        return defaults

    @staticmethod
    def merge_input_output_schemas(
        inputs: list[PropertySchema],
        outputs: list[PropertySchema],
    ) -> dict[str, type]:
        """Merge input and output schemas into a single state schema.

        Args:
            inputs: List of input property schemas
            outputs: List of output property schemas

        Returns:
            Combined state schema dictionary
        """
        input_schema = StateSchemaBuilder.build_from_properties(inputs)
        output_schema = StateSchemaBuilder.build_from_properties(outputs)
        return {**input_schema, **output_schema}

    @staticmethod
    def to_dapr_state_config(
        properties: list[PropertySchema],
        store_name: str = "statestore",
    ) -> dict[str, Any]:
        """Convert OAS properties to Dapr state store configuration.

        Args:
            properties: List of OAS property schemas
            store_name: Name of the Dapr state store component

        Returns:
            Dapr state configuration dictionary
        """
        schema = StateSchemaBuilder.build_from_properties(properties)
        defaults = StateSchemaBuilder.extract_defaults(properties)

        return {
            "store_name": store_name,
            "schema": {k: v.__name__ for k, v in schema.items()},
            "defaults": defaults,
        }
