"""Tests for utils module."""

from dapr_agents_oas_adapter.utils import (
    build_json_schema_property,
    camel_to_snake,
    extract_template_variables,
    generate_id,
    get_nested_value,
    json_schema_to_python_type,
    merge_dicts,
    python_type_to_json_schema,
    render_template,
    set_nested_value,
    snake_to_camel,
    validate_component_id,
)


class TestGenerateId:
    """Tests for generate_id function."""

    def test_without_prefix(self) -> None:
        """Test ID generation without prefix."""
        id1 = generate_id()
        id2 = generate_id()
        assert id1 != id2
        assert len(id1) == 36  # UUID format

    def test_with_prefix(self) -> None:
        """Test ID generation with prefix."""
        id1 = generate_id("agent")
        assert id1.startswith("agent_")
        assert len(id1) == 14  # prefix + underscore + 8 chars


class TestCaseConversion:
    """Tests for case conversion functions."""

    def test_snake_to_camel(self) -> None:
        """Test snake_case to camelCase conversion."""
        assert snake_to_camel("hello_world") == "helloWorld"
        assert snake_to_camel("my_variable_name") == "myVariableName"
        assert snake_to_camel("simple") == "simple"

    def test_camel_to_snake(self) -> None:
        """Test camelCase to snake_case conversion."""
        assert camel_to_snake("helloWorld") == "hello_world"
        assert camel_to_snake("myVariableName") == "my_variable_name"
        assert camel_to_snake("simple") == "simple"


class TestJsonSchemaTypeConversion:
    """Tests for JSON Schema type conversion functions."""

    def test_json_schema_to_python_type(self) -> None:
        """Test JSON Schema to Python type conversion."""
        assert json_schema_to_python_type({"type": "string"}) is str
        assert json_schema_to_python_type({"type": "integer"}) is int
        assert json_schema_to_python_type({"type": "number"}) is float
        assert json_schema_to_python_type({"type": "boolean"}) is bool
        assert json_schema_to_python_type({"type": "array"}) is list
        assert json_schema_to_python_type({"type": "object"}) is dict

    def test_json_schema_union_type(self) -> None:
        """Test JSON Schema union type conversion."""
        assert json_schema_to_python_type({"type": ["string", "null"]}) is str
        assert json_schema_to_python_type({"type": ["null", "integer"]}) is int

    def test_python_type_to_json_schema(self) -> None:
        """Test Python type to JSON Schema conversion."""
        assert python_type_to_json_schema(str) == "string"
        assert python_type_to_json_schema(int) == "integer"
        assert python_type_to_json_schema(float) == "number"
        assert python_type_to_json_schema(bool) == "boolean"
        assert python_type_to_json_schema(list) == "array"
        assert python_type_to_json_schema(dict) == "object"


class TestBuildJsonSchemaProperty:
    """Tests for build_json_schema_property function."""

    def test_basic_property(self) -> None:
        """Test building basic JSON Schema property."""
        prop = build_json_schema_property("name", str)
        assert prop["title"] == "name"
        assert prop["type"] == "string"

    def test_property_with_description(self) -> None:
        """Test building property with description."""
        prop = build_json_schema_property("age", int, description="User's age")
        assert prop["title"] == "age"
        assert prop["type"] == "integer"
        assert prop["description"] == "User's age"

    def test_property_with_default(self) -> None:
        """Test building property with default value."""
        prop = build_json_schema_property("count", int, default=0)
        assert prop["title"] == "count"
        assert prop["type"] == "integer"
        assert prop["default"] == 0


class TestTemplateVariables:
    """Tests for template variable functions."""

    def test_extract_template_variables(self) -> None:
        """Test extracting variables from template."""
        template = "Hello {{ name }}, you are {{ age }} years old."
        variables = extract_template_variables(template)
        assert "name" in variables
        assert "age" in variables
        assert len(variables) == 2

    def test_extract_variables_no_spaces(self) -> None:
        """Test extracting variables without spaces."""
        template = "Value: {{value}}"
        variables = extract_template_variables(template)
        assert "value" in variables

    def test_render_template(self) -> None:
        """Test rendering template with variables."""
        template = "Hello {{ name }}!"
        result = render_template(template, {"name": "World"})
        assert result == "Hello World!"

    def test_render_template_multiple_vars(self) -> None:
        """Test rendering template with multiple variables."""
        template = "{{ greeting }} {{ name }}, age {{ age }}."
        result = render_template(template, {"greeting": "Hi", "name": "Alice", "age": "30"})
        assert result == "Hi Alice, age 30."


class TestMergeDicts:
    """Tests for merge_dicts function."""

    def test_simple_merge(self) -> None:
        """Test simple dictionary merge."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self) -> None:
        """Test deep dictionary merge."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 5, "z": 6}}
        result = merge_dicts(base, override)
        assert result == {"a": {"x": 1, "y": 5, "z": 6}, "b": 3}


class TestValidateComponentId:
    """Tests for validate_component_id function."""

    def test_valid_ids(self) -> None:
        """Test valid component IDs."""
        assert validate_component_id("agent_123") is True
        assert validate_component_id("my-component") is True
        assert validate_component_id("Component1") is True

    def test_invalid_ids(self) -> None:
        """Test invalid component IDs."""
        assert validate_component_id("") is False
        assert validate_component_id("has spaces") is False
        assert validate_component_id("has.dots") is False


class TestNestedValues:
    """Tests for nested value functions."""

    def test_get_nested_value(self) -> None:
        """Test getting nested values."""
        data = {"a": {"b": {"c": 123}}}
        assert get_nested_value(data, "a.b.c") == 123
        assert get_nested_value(data, "a.b") == {"c": 123}
        assert get_nested_value(data, "x.y.z", "default") == "default"

    def test_set_nested_value(self) -> None:
        """Test setting nested values."""
        data: dict = {}
        set_nested_value(data, "a.b.c", 123)
        assert data == {"a": {"b": {"c": 123}}}

    def test_set_nested_value_existing(self) -> None:
        """Test setting nested value in existing structure."""
        data = {"a": {"b": {"c": 1}}}
        set_nested_value(data, "a.b.d", 2)
        assert data == {"a": {"b": {"c": 1, "d": 2}}}
