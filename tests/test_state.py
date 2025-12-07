"""Tests for StateSchemaBuilder module."""

from dapr_agents_oas_adapter.state import StateSchemaBuilder
from dapr_agents_oas_adapter.types import PropertySchema


class TestStateSchemaBuilder:
    """Tests for StateSchemaBuilder class."""

    def test_build_from_properties_with_string_type(self) -> None:
        """Test building schema from string properties."""
        properties: list[PropertySchema] = [
            {"title": "name", "type": "string"},
            {"title": "description", "type": "string"},
        ]
        result = StateSchemaBuilder.build_from_properties(properties)
        assert result == {"name": str, "description": str}

    def test_build_from_properties_with_number_types(self) -> None:
        """Test building schema from number properties."""
        properties: list[PropertySchema] = [
            {"title": "age", "type": "integer"},
            {"title": "price", "type": "number"},
        ]
        result = StateSchemaBuilder.build_from_properties(properties)
        assert result == {"age": int, "price": float}

    def test_build_from_properties_with_boolean_type(self) -> None:
        """Test building schema from boolean properties."""
        properties: list[PropertySchema] = [
            {"title": "is_active", "type": "boolean"},
        ]
        result = StateSchemaBuilder.build_from_properties(properties)
        assert result == {"is_active": bool}

    def test_build_from_properties_with_array_type(self) -> None:
        """Test building schema from array properties."""
        properties: list[PropertySchema] = [
            {"title": "items", "type": "array"},
        ]
        result = StateSchemaBuilder.build_from_properties(properties)
        assert result == {"items": list}

    def test_build_from_properties_with_object_type(self) -> None:
        """Test building schema from object properties."""
        properties: list[PropertySchema] = [
            {"title": "data", "type": "object"},
        ]
        result = StateSchemaBuilder.build_from_properties(properties)
        assert result == {"data": dict}

    def test_build_from_properties_empty_list(self) -> None:
        """Test building schema from empty properties list."""
        result = StateSchemaBuilder.build_from_properties([])
        assert result == {}

    def test_build_from_properties_skips_empty_title(self) -> None:
        """Test that properties without title are skipped."""
        properties: list[PropertySchema] = [
            {"title": "", "type": "string"},
            {"type": "string"},  # No title at all
            {"title": "valid", "type": "string"},
        ]
        result = StateSchemaBuilder.build_from_properties(properties)
        assert result == {"valid": str}

    def test_build_from_properties_mixed_types(self) -> None:
        """Test building schema from mixed property types."""
        properties: list[PropertySchema] = [
            {"title": "name", "type": "string"},
            {"title": "count", "type": "integer"},
            {"title": "score", "type": "number"},
            {"title": "active", "type": "boolean"},
            {"title": "tags", "type": "array"},
            {"title": "metadata", "type": "object"},
        ]
        result = StateSchemaBuilder.build_from_properties(properties)
        assert result == {
            "name": str,
            "count": int,
            "score": float,
            "active": bool,
            "tags": list,
            "metadata": dict,
        }

    def test_build_typed_dict_class_creates_valid_class(self) -> None:
        """Test that build_typed_dict_class creates a valid TypedDict."""
        properties: list[PropertySchema] = [
            {"title": "name", "type": "string"},
            {"title": "age", "type": "integer"},
        ]
        result = StateSchemaBuilder.build_typed_dict_class("TestState", properties)

        # Check it's a type
        assert isinstance(result, type)
        # Check name
        assert result.__name__ == "TestState"
        # Check annotations
        assert hasattr(result, "__annotations__")
        assert result.__annotations__ == {"name": str, "age": int}

    def test_build_typed_dict_class_empty_properties(self) -> None:
        """Test build_typed_dict_class with empty properties."""
        result = StateSchemaBuilder.build_typed_dict_class("EmptyState", [])
        assert isinstance(result, type)
        assert result.__name__ == "EmptyState"
        assert result.__annotations__ == {}

    def test_extract_defaults_with_defaults(self) -> None:
        """Test extracting default values from properties."""
        properties: list[PropertySchema] = [
            {"title": "name", "type": "string", "default": "default_name"},
            {"title": "count", "type": "integer", "default": 10},
            {"title": "active", "type": "boolean", "default": True},
        ]
        result = StateSchemaBuilder.extract_defaults(properties)
        assert result == {
            "name": "default_name",
            "count": 10,
            "active": True,
        }

    def test_extract_defaults_without_defaults(self) -> None:
        """Test extracting defaults when no defaults are present."""
        properties: list[PropertySchema] = [
            {"title": "name", "type": "string"},
            {"title": "count", "type": "integer"},
        ]
        result = StateSchemaBuilder.extract_defaults(properties)
        assert result == {}

    def test_extract_defaults_mixed_properties(self) -> None:
        """Test extracting defaults from mix of properties with and without defaults."""
        properties: list[PropertySchema] = [
            {"title": "name", "type": "string", "default": "test"},
            {"title": "count", "type": "integer"},  # No default
            {"title": "active", "type": "boolean", "default": False},
        ]
        result = StateSchemaBuilder.extract_defaults(properties)
        assert result == {"name": "test", "active": False}

    def test_extract_defaults_empty_list(self) -> None:
        """Test extracting defaults from empty list."""
        result = StateSchemaBuilder.extract_defaults([])
        assert result == {}

    def test_extract_defaults_skips_empty_title(self) -> None:
        """Test that properties without title are skipped."""
        properties: list[PropertySchema] = [
            {"title": "", "type": "string", "default": "skip"},
            {"title": "valid", "type": "string", "default": "keep"},
        ]
        result = StateSchemaBuilder.extract_defaults(properties)
        assert result == {"valid": "keep"}

    def test_extract_defaults_with_none_default(self) -> None:
        """Test extracting None as a default value."""
        properties: list[PropertySchema] = [
            {"title": "optional", "type": "string", "default": None},
        ]
        result = StateSchemaBuilder.extract_defaults(properties)
        assert result == {"optional": None}

    def test_merge_input_output_schemas(self) -> None:
        """Test merging input and output schemas."""
        inputs: list[PropertySchema] = [
            {"title": "query", "type": "string"},
            {"title": "limit", "type": "integer"},
        ]
        outputs: list[PropertySchema] = [
            {"title": "results", "type": "array"},
            {"title": "count", "type": "integer"},
        ]
        result = StateSchemaBuilder.merge_input_output_schemas(inputs, outputs)
        assert result == {
            "query": str,
            "limit": int,
            "results": list,
            "count": int,
        }

    def test_merge_input_output_schemas_empty_inputs(self) -> None:
        """Test merging with empty inputs."""
        outputs: list[PropertySchema] = [
            {"title": "result", "type": "string"},
        ]
        result = StateSchemaBuilder.merge_input_output_schemas([], outputs)
        assert result == {"result": str}

    def test_merge_input_output_schemas_empty_outputs(self) -> None:
        """Test merging with empty outputs."""
        inputs: list[PropertySchema] = [
            {"title": "input", "type": "string"},
        ]
        result = StateSchemaBuilder.merge_input_output_schemas(inputs, [])
        assert result == {"input": str}

    def test_merge_input_output_schemas_both_empty(self) -> None:
        """Test merging with both empty."""
        result = StateSchemaBuilder.merge_input_output_schemas([], [])
        assert result == {}

    def test_merge_input_output_schemas_overlapping_keys(self) -> None:
        """Test merging with overlapping keys (output should override)."""
        inputs: list[PropertySchema] = [
            {"title": "data", "type": "string"},
        ]
        outputs: list[PropertySchema] = [
            {"title": "data", "type": "object"},  # Same key, different type
        ]
        result = StateSchemaBuilder.merge_input_output_schemas(inputs, outputs)
        # Output schema should override input schema
        assert result == {"data": dict}

    def test_to_dapr_state_config_basic(self) -> None:
        """Test creating Dapr state configuration."""
        properties: list[PropertySchema] = [
            {"title": "name", "type": "string", "default": "test"},
            {"title": "count", "type": "integer"},
        ]
        result = StateSchemaBuilder.to_dapr_state_config(properties)
        assert result == {
            "store_name": "statestore",
            "schema": {"name": "str", "count": "int"},
            "defaults": {"name": "test"},
        }

    def test_to_dapr_state_config_custom_store_name(self) -> None:
        """Test creating Dapr state configuration with custom store name."""
        properties: list[PropertySchema] = [
            {"title": "value", "type": "number"},
        ]
        result = StateSchemaBuilder.to_dapr_state_config(properties, store_name="custom_store")
        assert result["store_name"] == "custom_store"
        assert result["schema"] == {"value": "float"}

    def test_to_dapr_state_config_empty_properties(self) -> None:
        """Test creating Dapr state configuration with empty properties."""
        result = StateSchemaBuilder.to_dapr_state_config([])
        assert result == {
            "store_name": "statestore",
            "schema": {},
            "defaults": {},
        }

    def test_to_dapr_state_config_all_types(self) -> None:
        """Test creating Dapr state configuration with all types."""
        properties: list[PropertySchema] = [
            {"title": "str_field", "type": "string"},
            {"title": "int_field", "type": "integer"},
            {"title": "float_field", "type": "number"},
            {"title": "bool_field", "type": "boolean"},
            {"title": "list_field", "type": "array"},
            {"title": "dict_field", "type": "object"},
        ]
        result = StateSchemaBuilder.to_dapr_state_config(properties)
        assert result["schema"] == {
            "str_field": "str",
            "int_field": "int",
            "float_field": "float",
            "bool_field": "bool",
            "list_field": "list",
            "dict_field": "dict",
        }

    def test_to_dapr_state_config_with_all_defaults(self) -> None:
        """Test creating Dapr state configuration with all defaults."""
        properties: list[PropertySchema] = [
            {"title": "name", "type": "string", "default": "default_name"},
            {"title": "count", "type": "integer", "default": 0},
            {"title": "enabled", "type": "boolean", "default": True},
        ]
        result = StateSchemaBuilder.to_dapr_state_config(properties)
        assert result["defaults"] == {
            "name": "default_name",
            "count": 0,
            "enabled": True,
        }
