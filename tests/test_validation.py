"""Tests for validation module (RF-005 acceptance criteria)."""

import pytest

from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)
from dapr_agents_oas_adapter.validation import (
    OASSchemaValidationError,
    OASSchemaValidator,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    WorkflowValidationError,
    WorkflowValidator,
    validate_oas_dict,
    validate_workflow,
    validate_workflow_dict,
)


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_issue_string_format_error(self) -> None:
        """Test error issue string formatting."""
        issue = ValidationIssue(
            message="Missing required field",
            severity=ValidationSeverity.ERROR,
            field="name",
            suggestion="Provide a name",
        )
        result = str(issue)
        assert "[ERROR]" in result
        assert "name:" in result
        assert "Missing required field" in result
        assert "Suggestion: Provide a name" in result

    def test_issue_string_format_warning(self) -> None:
        """Test warning issue string formatting."""
        issue = ValidationIssue(
            message="Task not connected",
            severity=ValidationSeverity.WARNING,
        )
        result = str(issue)
        assert "[WARNING]" in result
        assert "Task not connected" in result

    def test_issue_string_format_minimal(self) -> None:
        """Test minimal issue string formatting."""
        issue = ValidationIssue(message="Simple message")
        result = str(issue)
        assert "[ERROR]" in result
        assert "Simple message" in result


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_empty_result_is_valid(self) -> None:
        """Empty result should be valid."""
        result = ValidationResult()
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_result_with_error_is_invalid(self) -> None:
        """Result with error should be invalid."""
        result = ValidationResult()
        result.add_error("Test error")
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_result_with_warning_is_valid(self) -> None:
        """Result with only warning should be valid."""
        result = ValidationResult()
        result.add_warning("Test warning")
        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_add_error_with_all_fields(self) -> None:
        """Test adding error with all optional fields."""
        result = ValidationResult()
        result.add_error("Error message", field="test_field", suggestion="Fix it")
        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.message == "Error message"
        assert issue.field == "test_field"
        assert issue.suggestion == "Fix it"
        assert issue.severity == ValidationSeverity.ERROR

    def test_add_warning_with_all_fields(self) -> None:
        """Test adding warning with all optional fields."""
        result = ValidationResult()
        result.add_warning("Warning message", field="test_field", suggestion="Consider it")
        assert len(result.issues) == 1
        issue = result.issues[0]
        assert issue.severity == ValidationSeverity.WARNING

    def test_merge_results(self) -> None:
        """Test merging two validation results."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        result1.add_warning("Warning 1")

        result2 = ValidationResult()
        result2.add_error("Error 2")

        result1.merge(result2)
        assert len(result1.issues) == 3
        assert len(result1.errors) == 2
        assert len(result1.warnings) == 1

    def test_raise_if_invalid_with_errors(self) -> None:
        """Test raise_if_invalid raises on errors."""
        result = ValidationResult()
        result.add_error("Critical error")

        with pytest.raises(WorkflowValidationError) as exc_info:
            result.raise_if_invalid()

        assert len(exc_info.value.issues) == 1
        assert "Critical error" in str(exc_info.value)

    def test_raise_if_invalid_with_warnings_only(self) -> None:
        """Test raise_if_invalid does not raise on warnings only."""
        result = ValidationResult()
        result.add_warning("Just a warning")
        result.raise_if_invalid()  # Should not raise


class TestWorkflowValidator:
    """Tests for WorkflowValidator class."""

    def test_validate_valid_workflow(self) -> None:
        """Test validation of a valid workflow."""
        workflow = WorkflowDefinition(
            name="test_workflow",
            tasks=[
                WorkflowTaskDefinition(name="start", task_type="start"),
                WorkflowTaskDefinition(name="process", task_type="llm"),
                WorkflowTaskDefinition(name="end", task_type="end"),
            ],
            edges=[
                WorkflowEdgeDefinition(from_node="start", to_node="process"),
                WorkflowEdgeDefinition(from_node="process", to_node="end"),
            ],
            start_node="start",
            end_nodes=["end"],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is True

    def test_validate_missing_name(self) -> None:
        """Test validation catches missing workflow name."""
        workflow = WorkflowDefinition(
            name="",  # Empty name
            tasks=[WorkflowTaskDefinition(name="task1", task_type="llm")],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("name" in str(e).lower() for e in result.errors)

    def test_validate_empty_tasks_warning(self) -> None:
        """Test validation warns about empty tasks."""
        workflow = WorkflowDefinition(name="empty_workflow", tasks=[])
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        # Empty tasks is a warning, not error
        assert any("no tasks" in str(w).lower() for w in result.warnings)

    def test_validate_duplicate_task_names(self) -> None:
        """Test validation catches duplicate task names."""
        workflow = WorkflowDefinition(
            name="dup_workflow",
            tasks=[
                WorkflowTaskDefinition(name="same_name", task_type="llm"),
                WorkflowTaskDefinition(name="same_name", task_type="tool"),
            ],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("duplicate" in str(e).lower() for e in result.errors)

    def test_validate_task_missing_name(self) -> None:
        """Test validation catches task missing name."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="", task_type="llm")],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("task name" in str(e).lower() for e in result.errors)

    def test_validate_task_missing_type(self) -> None:
        """Test validation catches task missing type."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="task1", task_type="")],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("task type" in str(e).lower() for e in result.errors)

    def test_validate_unknown_task_type_warning(self) -> None:
        """Test validation warns about unknown task type."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="task1", task_type="unknown_type")],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert any("unknown task type" in str(w).lower() for w in result.warnings)

    def test_validate_tool_task_missing_tool_name(self) -> None:
        """Test validation warns when tool task missing tool_name."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="tool_task", task_type="tool", config={})],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert any("tool_name" in str(w).lower() for w in result.warnings)

    def test_validate_flow_task_missing_flow_id(self) -> None:
        """Test validation warns when flow task missing flow_id."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="flow_task", task_type="flow", config={})],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert any("flow_id" in str(w).lower() for w in result.warnings)

    def test_validate_edge_missing_from_node(self) -> None:
        """Test validation catches edge missing from_node."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="task1", task_type="llm")],
            edges=[WorkflowEdgeDefinition(from_node="", to_node="task1")],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("from_node" in str(e).lower() for e in result.errors)

    def test_validate_edge_missing_to_node(self) -> None:
        """Test validation catches edge missing to_node."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="task1", task_type="llm")],
            edges=[WorkflowEdgeDefinition(from_node="task1", to_node="")],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("to_node" in str(e).lower() for e in result.errors)

    def test_validate_self_referencing_edge(self) -> None:
        """Test validation catches self-referencing edge."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="task1", task_type="llm")],
            edges=[WorkflowEdgeDefinition(from_node="task1", to_node="task1")],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("self-referencing" in str(e).lower() for e in result.errors)

    def test_validate_invalid_start_node_reference(self) -> None:
        """Test validation catches invalid start_node reference."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="task1", task_type="llm")],
            start_node="nonexistent",
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("start node" in str(e).lower() for e in result.errors)

    def test_validate_invalid_end_node_reference(self) -> None:
        """Test validation catches invalid end_nodes reference."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="task1", task_type="llm")],
            end_nodes=["nonexistent"],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("end node" in str(e).lower() for e in result.errors)

    def test_validate_edge_references_unknown_from_node(self) -> None:
        """Test validation catches edge referencing unknown from_node."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="task1", task_type="llm")],
            edges=[WorkflowEdgeDefinition(from_node="unknown", to_node="task1")],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("unknown source node" in str(e).lower() for e in result.errors)

    def test_validate_edge_references_unknown_to_node(self) -> None:
        """Test validation catches edge referencing unknown to_node."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[WorkflowTaskDefinition(name="task1", task_type="llm")],
            edges=[WorkflowEdgeDefinition(from_node="task1", to_node="unknown")],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("unknown target node" in str(e).lower() for e in result.errors)

    def test_validate_orphan_task_warning(self) -> None:
        """Test validation warns about orphan tasks."""
        workflow = WorkflowDefinition(
            name="workflow",
            tasks=[
                WorkflowTaskDefinition(name="connected", task_type="llm"),
                WorkflowTaskDefinition(name="orphan", task_type="tool"),
            ],
            edges=[WorkflowEdgeDefinition(from_node="connected", to_node="connected")],
            start_node="connected",
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        # Orphan warning, but self-referencing edge is an error
        # Check for the orphan warning specifically
        warnings_str = " ".join(str(w) for w in result.warnings)
        assert "orphan" in warnings_str.lower() or "not connected" in warnings_str.lower()

    def test_validate_cycle_detection(self) -> None:
        """Test validation detects cycles in workflow."""
        workflow = WorkflowDefinition(
            name="cyclic_workflow",
            tasks=[
                WorkflowTaskDefinition(name="a", task_type="llm"),
                WorkflowTaskDefinition(name="b", task_type="llm"),
                WorkflowTaskDefinition(name="c", task_type="llm"),
            ],
            edges=[
                WorkflowEdgeDefinition(from_node="a", to_node="b"),
                WorkflowEdgeDefinition(from_node="b", to_node="c"),
                WorkflowEdgeDefinition(from_node="c", to_node="a"),  # Creates cycle
            ],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert result.is_valid is False
        assert any("cycle" in str(e).lower() for e in result.errors)

    def test_validate_no_cycle_in_dag(self) -> None:
        """Test validation passes for DAG (no cycles)."""
        workflow = WorkflowDefinition(
            name="dag_workflow",
            tasks=[
                WorkflowTaskDefinition(name="a", task_type="llm"),
                WorkflowTaskDefinition(name="b", task_type="llm"),
                WorkflowTaskDefinition(name="c", task_type="llm"),
            ],
            edges=[
                WorkflowEdgeDefinition(from_node="a", to_node="b"),
                WorkflowEdgeDefinition(from_node="a", to_node="c"),
                WorkflowEdgeDefinition(from_node="b", to_node="c"),
            ],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert not any("cycle" in str(e).lower() for e in result.errors)

    def test_validate_duplicate_branch_warning(self) -> None:
        """Test validation warns about duplicate branches from same node."""
        workflow = WorkflowDefinition(
            name="branching_workflow",
            tasks=[
                WorkflowTaskDefinition(name="decision", task_type="llm"),
                WorkflowTaskDefinition(name="path1", task_type="llm"),
                WorkflowTaskDefinition(name="path2", task_type="llm"),
            ],
            edges=[
                WorkflowEdgeDefinition(from_node="decision", to_node="path1", from_branch="yes"),
                # Duplicate branch value
                WorkflowEdgeDefinition(from_node="decision", to_node="path2", from_branch="yes"),
            ],
        )
        validator = WorkflowValidator()
        result = validator.validate(workflow)
        assert any("duplicate branch" in str(w).lower() for w in result.warnings)


class TestConvenienceFunctions:
    """Tests for convenience validation functions."""

    def test_validate_workflow_function(self) -> None:
        """Test validate_workflow convenience function."""
        workflow = WorkflowDefinition(
            name="test",
            tasks=[WorkflowTaskDefinition(name="t1", task_type="llm")],
        )
        result = validate_workflow(workflow)
        assert isinstance(result, ValidationResult)

    def test_validate_workflow_raise_on_error(self) -> None:
        """Test validate_workflow with raise_on_error."""
        workflow = WorkflowDefinition(
            name="",  # Invalid
            tasks=[],
        )
        with pytest.raises(WorkflowValidationError):
            validate_workflow(workflow, raise_on_error=True)

    def test_validate_workflow_dict_function(self) -> None:
        """Test validate_workflow_dict convenience function."""
        workflow_dict = {
            "name": "test",
            "tasks": [{"name": "t1", "task_type": "llm"}],
        }
        result = validate_workflow_dict(workflow_dict)
        assert isinstance(result, ValidationResult)

    def test_validate_workflow_dict_invalid_structure(self) -> None:
        """Test validate_workflow_dict with invalid structure."""
        workflow_dict = {"invalid": "structure"}
        result = validate_workflow_dict(workflow_dict)
        assert result.is_valid is False
        assert any("parse" in str(e).lower() or "failed" in str(e).lower() for e in result.errors)

    def test_validate_workflow_dict_raise_on_error(self) -> None:
        """Test validate_workflow_dict with raise_on_error."""
        workflow_dict = {"invalid": "structure"}
        with pytest.raises(WorkflowValidationError):
            validate_workflow_dict(workflow_dict, raise_on_error=True)

    def test_validate_workflow_dict_chains_cause(self) -> None:
        """Test that raise_if_invalid chains the original parse exception."""
        workflow_dict = {"invalid": "structure"}
        with pytest.raises(WorkflowValidationError) as exc_info:
            validate_workflow_dict(workflow_dict, raise_on_error=True)
        assert exc_info.value.__cause__ is not None

    def test_raise_if_invalid_chains_cause_from_add_error(self) -> None:
        """Test that raise_if_invalid sets __cause__ from first error's cause."""
        original = ValueError("original error")
        result = ValidationResult()
        result.add_error("wrapper message", cause=original)
        with pytest.raises(WorkflowValidationError) as exc_info:
            result.raise_if_invalid()
        assert exc_info.value.__cause__ is original

    def test_raise_if_invalid_no_cause_when_absent(self) -> None:
        """Test that __cause__ is None when no cause is provided."""
        result = ValidationResult()
        result.add_error("error without cause")
        with pytest.raises(WorkflowValidationError) as exc_info:
            result.raise_if_invalid()
        assert exc_info.value.__cause__ is None


class TestWorkflowValidationError:
    """Tests for WorkflowValidationError exception."""

    def test_error_contains_issues(self) -> None:
        """Test error contains all issues."""
        issues = [
            ValidationIssue(message="Error 1"),
            ValidationIssue(message="Error 2"),
        ]
        error = WorkflowValidationError(issues)
        assert len(error.issues) == 2
        assert "2 error(s)" in str(error)
        assert "Error 1" in str(error)
        assert "Error 2" in str(error)


# =============================================================================
# OAS Schema Validation Tests (RF-010)
# =============================================================================


class TestOASSchemaValidator:
    """Tests for OASSchemaValidator class."""

    def test_validate_valid_agent(self) -> None:
        """Test validation of a valid agent component."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Agent",
            "name": "test_agent",
            "description": "A test agent",
            "role": "assistant",
            "goal": "Help users",
        }
        result = validator.validate_agent(data)
        assert result.is_valid is True

    def test_validate_agent_missing_name(self) -> None:
        """Test validation catches missing agent name."""
        validator = OASSchemaValidator()
        data = {"component_type": "Agent", "description": "No name"}
        result = validator.validate_agent(data)
        assert result.is_valid is False
        assert any("name" in str(e).lower() for e in result.errors)

    def test_validate_agent_empty_name(self) -> None:
        """Test validation catches empty agent name."""
        validator = OASSchemaValidator()
        data = {"component_type": "Agent", "name": ""}
        result = validator.validate_agent(data)
        assert result.is_valid is False
        assert any("name" in str(e).lower() for e in result.errors)

    def test_validate_agent_unknown_fields_warning(self) -> None:
        """Test validation warns about unknown fields in agent."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Agent",
            "name": "test",
            "unknown_field": "some_value",
        }
        result = validator.validate_agent(data)
        assert result.is_valid is True  # Valid, just warns
        assert any("unknown field" in str(w).lower() for w in result.warnings)

    def test_validate_agent_with_tools(self) -> None:
        """Test validation of agent with tools."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Agent",
            "name": "test",
            "tools": [{"name": "search_tool"}, "another_tool"],
        }
        result = validator.validate_agent(data)
        assert result.is_valid is True

    def test_validate_agent_invalid_tools_format(self) -> None:
        """Test validation catches invalid tools format."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Agent",
            "name": "test",
            "tools": "not_a_list",  # Should be a list
        }
        result = validator.validate_agent(data)
        assert result.is_valid is False
        assert any("tools must be an array" in str(e).lower() for e in result.errors)

    def test_validate_agent_tool_missing_name(self) -> None:
        """Test validation catches tool missing name."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Agent",
            "name": "test",
            "tools": [{"description": "No name tool"}],  # Missing name
        }
        result = validator.validate_agent(data)
        assert result.is_valid is False
        assert any("missing 'name'" in str(e).lower() for e in result.errors)

    def test_validate_agent_invalid_tool_type(self) -> None:
        """Test validation catches invalid tool type."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Agent",
            "name": "test",
            "tools": [123],  # Should be string or dict
        }
        result = validator.validate_agent(data)
        assert result.is_valid is False
        assert any("must be a string or object" in str(e).lower() for e in result.errors)

    def test_validate_agent_with_llm_config(self) -> None:
        """Test validation of agent with LLM config."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Agent",
            "name": "test",
            "llm_config": {"type": "openai", "model": "gpt-4"},
        }
        result = validator.validate_agent(data)
        assert result.is_valid is True

    def test_validate_agent_invalid_llm_config_format(self) -> None:
        """Test validation catches invalid llm_config format."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Agent",
            "name": "test",
            "llm_config": "not_a_dict",
        }
        result = validator.validate_agent(data)
        assert result.is_valid is False
        assert any("llm_config must be an object" in str(e).lower() for e in result.errors)

    def test_validate_agent_unknown_llm_type_warning(self) -> None:
        """Test validation warns about unknown LLM type."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Agent",
            "name": "test",
            "llm_config": {"type": "unknown_llm"},
        }
        result = validator.validate_agent(data)
        assert result.is_valid is True  # Valid, just warns
        assert any("unknown llm type" in str(w).lower() for w in result.warnings)

    def test_validate_valid_flow(self) -> None:
        """Test validation of a valid flow component."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test_flow",
            "description": "A test flow",
            "nodes": [
                {"id": "start", "type": "start"},
                {"id": "process", "type": "llm"},
                {"id": "end", "type": "end"},
            ],
            "edges": [
                {"from": "start", "to": "process"},
                {"from": "process", "to": "end"},
            ],
        }
        result = validator.validate_flow(data)
        assert result.is_valid is True

    def test_validate_flow_missing_name(self) -> None:
        """Test validation catches missing flow name."""
        validator = OASSchemaValidator()
        data = {"component_type": "Flow", "description": "No name"}
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("name" in str(e).lower() for e in result.errors)

    def test_validate_flow_unknown_fields_warning(self) -> None:
        """Test validation warns about unknown fields in flow."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "unknown_field": "value",
        }
        result = validator.validate_flow(data)
        assert result.is_valid is True
        assert any("unknown field" in str(w).lower() for w in result.warnings)

    def test_validate_flow_invalid_nodes_format(self) -> None:
        """Test validation catches invalid nodes format."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "nodes": "not_a_list",
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("nodes must be an array" in str(e).lower() for e in result.errors)

    def test_validate_flow_node_not_object(self) -> None:
        """Test validation catches node that is not an object."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "nodes": ["not_an_object"],
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("must be an object" in str(e).lower() for e in result.errors)

    def test_validate_flow_node_missing_id(self) -> None:
        """Test validation catches node missing id."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "nodes": [{"type": "llm"}],  # Missing id
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("missing 'id'" in str(e).lower() for e in result.errors)

    def test_validate_flow_duplicate_node_ids(self) -> None:
        """Test validation catches duplicate node ids."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "nodes": [
                {"id": "same_id", "type": "llm"},
                {"id": "same_id", "type": "tool"},
            ],
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("duplicate node id" in str(e).lower() for e in result.errors)

    def test_validate_flow_unknown_node_type_warning(self) -> None:
        """Test validation warns about unknown node type."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "nodes": [{"id": "node1", "type": "unknown_type"}],
        }
        result = validator.validate_flow(data)
        assert result.is_valid is True
        assert any("unknown node type" in str(w).lower() for w in result.warnings)

    def test_validate_flow_invalid_edges_format(self) -> None:
        """Test validation catches invalid edges format."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "edges": "not_a_list",
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("edges must be an array" in str(e).lower() for e in result.errors)

    def test_validate_flow_edge_not_object(self) -> None:
        """Test validation catches edge that is not an object."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "edges": ["not_an_object"],
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("must be an object" in str(e).lower() for e in result.errors)

    def test_validate_flow_edge_missing_from(self) -> None:
        """Test validation catches edge missing 'from' field."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "nodes": [{"id": "node1", "type": "llm"}],
            "edges": [{"to": "node1"}],  # Missing 'from'
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("missing 'from'" in str(e).lower() for e in result.errors)

    def test_validate_flow_edge_missing_to(self) -> None:
        """Test validation catches edge missing 'to' field."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "nodes": [{"id": "node1", "type": "llm"}],
            "edges": [{"from": "node1"}],  # Missing 'to'
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("missing 'to'" in str(e).lower() for e in result.errors)

    def test_validate_flow_edge_references_unknown_from_node(self) -> None:
        """Test validation catches edge referencing unknown from node."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "nodes": [{"id": "node1", "type": "llm"}],
            "edges": [{"from": "unknown", "to": "node1"}],
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("non-existent node" in str(e).lower() for e in result.errors)

    def test_validate_flow_edge_references_unknown_to_node(self) -> None:
        """Test validation catches edge referencing unknown to node."""
        validator = OASSchemaValidator()
        data = {
            "component_type": "Flow",
            "name": "test",
            "nodes": [{"id": "node1", "type": "llm"}],
            "edges": [{"from": "node1", "to": "unknown"}],
        }
        result = validator.validate_flow(data)
        assert result.is_valid is False
        assert any("non-existent node" in str(e).lower() for e in result.errors)


class TestOASSchemaValidatorComponent:
    """Tests for validate_component method."""

    def test_validate_component_agent(self) -> None:
        """Test validate_component routes to validate_agent."""
        validator = OASSchemaValidator()
        data = {"component_type": "Agent", "name": "test"}
        result = validator.validate_component(data)
        assert result.is_valid is True

    def test_validate_component_flow(self) -> None:
        """Test validate_component routes to validate_flow."""
        validator = OASSchemaValidator()
        data = {"component_type": "Flow", "name": "test"}
        result = validator.validate_component(data)
        assert result.is_valid is True

    def test_validate_component_unknown_type(self) -> None:
        """Test validate_component with unknown component type."""
        validator = OASSchemaValidator()
        data = {"component_type": "Unknown", "name": "test"}
        result = validator.validate_component(data)
        assert result.is_valid is False
        assert any("unknown or missing component_type" in str(e).lower() for e in result.errors)

    def test_validate_component_missing_type(self) -> None:
        """Test validate_component with missing component type."""
        validator = OASSchemaValidator()
        data = {"name": "test"}  # No component_type
        result = validator.validate_component(data)
        assert result.is_valid is False
        assert any("unknown or missing component_type" in str(e).lower() for e in result.errors)

    def test_validate_component_raise_on_error(self) -> None:
        """Test validate_component with raise_on_error."""
        validator = OASSchemaValidator()
        data = {"component_type": "Unknown"}
        with pytest.raises(OASSchemaValidationError) as exc_info:
            validator.validate_component(data, raise_on_error=True)
        assert len(exc_info.value.issues) > 0

    def test_validate_agent_raise_on_error(self) -> None:
        """Test validate_agent with raise_on_error."""
        validator = OASSchemaValidator()
        data = {"component_type": "Agent"}  # Missing name
        with pytest.raises(OASSchemaValidationError):
            validator.validate_agent(data, raise_on_error=True)

    def test_validate_flow_raise_on_error(self) -> None:
        """Test validate_flow with raise_on_error."""
        validator = OASSchemaValidator()
        data = {"component_type": "Flow"}  # Missing name
        with pytest.raises(OASSchemaValidationError):
            validator.validate_flow(data, raise_on_error=True)


class TestValidateOASDictFunction:
    """Tests for validate_oas_dict convenience function."""

    def test_validate_oas_dict_valid_agent(self) -> None:
        """Test validate_oas_dict with valid agent."""
        data = {"component_type": "Agent", "name": "test"}
        result = validate_oas_dict(data)
        assert result.is_valid is True

    def test_validate_oas_dict_valid_flow(self) -> None:
        """Test validate_oas_dict with valid flow."""
        data = {"component_type": "Flow", "name": "test"}
        result = validate_oas_dict(data)
        assert result.is_valid is True

    def test_validate_oas_dict_invalid(self) -> None:
        """Test validate_oas_dict with invalid data."""
        data = {"component_type": "Agent"}  # Missing name
        result = validate_oas_dict(data)
        assert result.is_valid is False

    def test_validate_oas_dict_raise_on_error(self) -> None:
        """Test validate_oas_dict with raise_on_error."""
        data = {"component_type": "Unknown"}
        with pytest.raises(OASSchemaValidationError):
            validate_oas_dict(data, raise_on_error=True)


class TestOASSchemaValidationError:
    """Tests for OASSchemaValidationError exception."""

    def test_error_contains_issues(self) -> None:
        """Test error contains all issues."""
        issues = [
            ValidationIssue(message="OAS Error 1"),
            ValidationIssue(message="OAS Error 2"),
        ]
        error = OASSchemaValidationError(issues)
        assert len(error.issues) == 2
        assert "2 error(s)" in str(error)
        assert "OAS Error 1" in str(error)
        assert "OAS Error 2" in str(error)

    def test_error_message_format(self) -> None:
        """Test error message formatting."""
        issues = [ValidationIssue(message="Test error", field="test_field")]
        error = OASSchemaValidationError(issues)
        assert "OAS schema validation failed" in str(error)
