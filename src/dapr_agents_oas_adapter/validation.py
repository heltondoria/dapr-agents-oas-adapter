"""Comprehensive validation for workflow definitions and OAS components.

This module provides validation utilities to catch errors early and provide
actionable error messages for invalid configurations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A single validation issue found during validation."""

    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    field: str | None = None
    suggestion: str | None = None

    def __str__(self) -> str:
        """Format the issue as a string."""
        parts = [f"[{self.severity.value.upper()}]"]
        if self.field:
            parts.append(f"{self.field}:")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"(Suggestion: {self.suggestion})")
        return " ".join(parts)


@dataclass
class ValidationResult:
    """Result of a validation operation."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return not any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    def add_error(
        self,
        message: str,
        field: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add an error-level issue."""
        self.issues.append(
            ValidationIssue(
                message=message,
                severity=ValidationSeverity.ERROR,
                field=field,
                suggestion=suggestion,
            )
        )

    def add_warning(
        self,
        message: str,
        field: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Add a warning-level issue."""
        self.issues.append(
            ValidationIssue(
                message=message,
                severity=ValidationSeverity.WARNING,
                field=field,
                suggestion=suggestion,
            )
        )

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        self.issues.extend(other.issues)

    def raise_if_invalid(self) -> None:
        """Raise WorkflowValidationError if there are any errors."""
        if not self.is_valid:
            raise WorkflowValidationError(self.errors)


class WorkflowValidationError(Exception):
    """Exception raised when workflow validation fails."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        """Initialize with list of validation issues."""
        self.issues = issues
        messages = [str(issue) for issue in issues]
        super().__init__(
            f"Workflow validation failed with {len(issues)} error(s):\n" + "\n".join(messages)
        )


class WorkflowValidator:
    """Comprehensive validator for WorkflowDefinition objects.

    Performs three levels of validation:
    1. Structure validation: Required fields, proper types
    2. Reference validation: Node references exist, no orphans
    3. Edge validation: Connected graph, no cycles in control flow, branching consistency

    Usage:
        validator = WorkflowValidator()
        result = validator.validate(workflow_def)
        if not result.is_valid:
            for error in result.errors:
                print(error)
        # Or raise on first error:
        result.raise_if_invalid()
    """

    def validate(self, workflow: WorkflowDefinition) -> ValidationResult:
        """Validate a workflow definition comprehensively.

        Args:
            workflow: The workflow definition to validate

        Returns:
            ValidationResult with all issues found
        """
        result = ValidationResult()

        # Run all validation phases
        result.merge(self._validate_structure(workflow))
        result.merge(self._validate_references(workflow))
        result.merge(self._validate_edges(workflow))

        return result

    def _validate_structure(self, workflow: WorkflowDefinition) -> ValidationResult:
        """Validate workflow structure (required fields, types)."""
        result = ValidationResult()

        # Workflow name is required
        if not workflow.name:
            result.add_error(
                "Workflow name is required",
                field="name",
                suggestion="Provide a descriptive name for the workflow",
            )

        # Validate tasks
        if not workflow.tasks:
            result.add_warning(
                "Workflow has no tasks defined",
                field="tasks",
                suggestion="Add at least one task to the workflow",
            )

        task_names: set[str] = set()
        for i, task in enumerate(workflow.tasks):
            task_result = self._validate_task(task, i)
            result.merge(task_result)

            # Check for duplicate task names
            if task.name in task_names:
                result.add_error(
                    f"Duplicate task name: '{task.name}'",
                    field=f"tasks[{i}].name",
                    suggestion="Use unique names for each task",
                )
            task_names.add(task.name)

        # Validate edges
        for i, edge in enumerate(workflow.edges):
            edge_result = self._validate_edge_structure(edge, i)
            result.merge(edge_result)

        return result

    def _validate_task(self, task: WorkflowTaskDefinition, index: int) -> ValidationResult:
        """Validate a single task definition."""
        result = ValidationResult()
        prefix = f"tasks[{index}]"

        if not task.name:
            result.add_error(
                "Task name is required",
                field=f"{prefix}.name",
                suggestion="Provide a unique name for the task",
            )

        if not task.task_type:
            result.add_error(
                "Task type is required",
                field=f"{prefix}.task_type",
                suggestion="Specify task type: 'llm', 'tool', 'agent', 'flow', or 'map'",
            )

        valid_types = {"llm", "tool", "agent", "flow", "map", "start", "end"}
        if task.task_type and task.task_type not in valid_types:
            result.add_warning(
                f"Unknown task type: '{task.task_type}'",
                field=f"{prefix}.task_type",
                suggestion=f"Valid types are: {', '.join(sorted(valid_types))}",
            )

        # Type-specific validation
        if task.task_type == "tool" and not task.config.get("tool_name"):
            result.add_warning(
                "Tool task missing 'tool_name' in config",
                field=f"{prefix}.config.tool_name",
                suggestion="Specify the tool to execute",
            )

        if task.task_type == "flow" and not task.config.get("flow_id"):
            result.add_warning(
                "Flow task missing 'flow_id' in config",
                field=f"{prefix}.config.flow_id",
                suggestion="Specify the subflow to execute",
            )

        return result

    def _validate_edge_structure(
        self, edge: WorkflowEdgeDefinition, index: int
    ) -> ValidationResult:
        """Validate a single edge structure."""
        result = ValidationResult()
        prefix = f"edges[{index}]"

        if not edge.from_node:
            result.add_error(
                "Edge 'from_node' is required",
                field=f"{prefix}.from_node",
                suggestion="Specify the source node for this edge",
            )

        if not edge.to_node:
            result.add_error(
                "Edge 'to_node' is required",
                field=f"{prefix}.to_node",
                suggestion="Specify the target node for this edge",
            )

        if edge.from_node and edge.to_node and edge.from_node == edge.to_node:
            result.add_error(
                f"Self-referencing edge: '{edge.from_node}' -> '{edge.to_node}'",
                field=prefix,
                suggestion="Remove the self-loop or use different nodes",
            )

        return result

    def _validate_references(self, workflow: WorkflowDefinition) -> ValidationResult:
        """Validate that all node references exist."""
        result = ValidationResult()

        # Build set of valid task names
        task_names = {task.name for task in workflow.tasks}

        # Validate start_node reference
        if workflow.start_node and workflow.start_node not in task_names:
            result.add_error(
                f"Start node '{workflow.start_node}' not found in tasks",
                field="start_node",
                suggestion=f"Valid task names are: {', '.join(sorted(task_names)) or '(none)'}",
            )

        # Validate end_nodes references
        for i, end_node in enumerate(workflow.end_nodes):
            if end_node not in task_names:
                result.add_error(
                    f"End node '{end_node}' not found in tasks",
                    field=f"end_nodes[{i}]",
                    suggestion=f"Valid task names are: {', '.join(sorted(task_names)) or '(none)'}",
                )

        # Validate edge node references
        for i, edge in enumerate(workflow.edges):
            if edge.from_node and edge.from_node not in task_names:
                result.add_error(
                    f"Edge references unknown source node: '{edge.from_node}'",
                    field=f"edges[{i}].from_node",
                    suggestion=f"Valid task names are: {', '.join(sorted(task_names)) or '(none)'}",
                )
            if edge.to_node and edge.to_node not in task_names:
                result.add_error(
                    f"Edge references unknown target node: '{edge.to_node}'",
                    field=f"edges[{i}].to_node",
                    suggestion=f"Valid task names are: {', '.join(sorted(task_names)) or '(none)'}",
                )

        # Check for orphan tasks (no incoming or outgoing edges)
        if workflow.edges:
            connected_nodes = set()
            for edge in workflow.edges:
                if edge.from_node:
                    connected_nodes.add(edge.from_node)
                if edge.to_node:
                    connected_nodes.add(edge.to_node)

            for task in workflow.tasks:
                # Check for orphaned tasks (not connected and not start/end)
                is_orphan = (
                    task.name not in connected_nodes
                    and task.name != workflow.start_node
                    and task.name not in workflow.end_nodes
                )
                if is_orphan:
                    result.add_warning(
                        f"Task '{task.name}' is not connected by any edge",
                        field=f"tasks.{task.name}",
                        suggestion="Add edges to connect this task or remove it",
                    )

        return result

    def _validate_edges(self, workflow: WorkflowDefinition) -> ValidationResult:
        """Validate edge connectivity and branching consistency."""
        result = ValidationResult()

        if not workflow.edges:
            return result

        # Build adjacency for cycle detection
        graph: dict[str, list[str]] = {}
        for task in workflow.tasks:
            graph[task.name] = []

        for edge in workflow.edges:
            if edge.from_node in graph and edge.to_node:
                graph[edge.from_node].append(edge.to_node)

        # Check for cycles (simple DFS-based detection)
        cycles = self._detect_cycles(graph)
        for cycle in cycles:
            result.add_error(
                f"Cycle detected in workflow: {' -> '.join(cycle)}",
                field="edges",
                suggestion="Remove or restructure edges to eliminate the cycle",
            )

        # Check branching consistency
        branch_edges: dict[str, list[str]] = {}  # node -> list of branches
        for edge in workflow.edges:
            if edge.from_branch:
                if edge.from_node not in branch_edges:
                    branch_edges[edge.from_node] = []
                branch_edges[edge.from_node].append(edge.from_branch)

        # Warn about duplicate branch values from same node
        for node, branches in branch_edges.items():
            seen_branches: set[str] = set()
            for branch in branches:
                if branch in seen_branches:
                    result.add_warning(
                        f"Duplicate branch '{branch}' from node '{node}'",
                        field=f"edges (from_node={node})",
                        suggestion="Use unique branch values from each decision node",
                    )
                seen_branches.add(branch)

        return result

    def _detect_cycles(self, graph: dict[str, list[str]]) -> list[list[str]]:
        """Detect cycles in the graph using DFS.

        Args:
            graph: Adjacency list representation

        Returns:
            List of cycles found (each cycle is a list of node names)
        """
        cycles: list[list[str]] = []
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycles.append([*path[cycle_start:], neighbor])
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles


def validate_workflow(
    workflow: WorkflowDefinition, *, raise_on_error: bool = False
) -> ValidationResult:
    """Convenience function to validate a workflow.

    Args:
        workflow: The workflow definition to validate
        raise_on_error: If True, raise WorkflowValidationError on validation errors

    Returns:
        ValidationResult with all issues found

    Raises:
        WorkflowValidationError: If raise_on_error=True and validation fails
    """
    validator = WorkflowValidator()
    result = validator.validate(workflow)
    if raise_on_error:
        result.raise_if_invalid()
    return result


def validate_workflow_dict(
    workflow_dict: dict[str, Any], *, raise_on_error: bool = False
) -> ValidationResult:
    """Validate a workflow from a dictionary representation.

    Args:
        workflow_dict: Dictionary representation of a workflow
        raise_on_error: If True, raise WorkflowValidationError on validation errors

    Returns:
        ValidationResult with all issues found

    Raises:
        WorkflowValidationError: If raise_on_error=True and validation fails
    """
    try:
        workflow = WorkflowDefinition.model_validate(workflow_dict)
    except Exception as e:
        result = ValidationResult()
        result.add_error(
            f"Failed to parse workflow: {e}", suggestion="Check the workflow structure"
        )
        if raise_on_error:
            result.raise_if_invalid()
        return result

    return validate_workflow(workflow, raise_on_error=raise_on_error)


# =============================================================================
# OAS Schema Validation
# =============================================================================


class OASSchemaValidationError(Exception):
    """Exception raised when OAS schema validation fails."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        """Initialize with list of validation issues."""
        self.issues = issues
        messages = [str(issue) for issue in issues]
        super().__init__(
            f"OAS schema validation failed with {len(issues)} error(s):\n" + "\n".join(messages)
        )


class OASSchemaValidator:
    """Validates OAS (Open Agent Spec) dictionaries before conversion.

    Performs validation to ensure the input data has the correct structure
    before attempting to convert it to Dapr formats.

    Usage:
        validator = OASSchemaValidator()
        result = validator.validate_agent(agent_dict)
        if not result.is_valid:
            for error in result.errors:
                print(error)

        # Or with strict mode:
        validator.validate_agent(agent_dict, raise_on_error=True)
    """

    # Required fields for Agent component
    AGENT_REQUIRED_FIELDS: ClassVar[set[str]] = {"name"}
    AGENT_OPTIONAL_FIELDS: ClassVar[set[str]] = {
        "component_type",
        "description",
        "prompt",
        "system_prompt",
        "tools",
        "llm_config",
        "metadata",
        "role",
        "goal",
        "instructions",
    }

    # Required fields for Flow component
    FLOW_REQUIRED_FIELDS: ClassVar[set[str]] = {"name"}
    FLOW_OPTIONAL_FIELDS: ClassVar[set[str]] = {
        "component_type",
        "description",
        "nodes",
        "edges",
        "start",
        "end",
        "metadata",
    }

    # Valid node types
    VALID_NODE_TYPES: ClassVar[set[str]] = {
        "start",
        "end",
        "llm",
        "tool",
        "agent",
        "flow",
        "map",
    }

    def validate_component(
        self,
        data: dict[str, Any],
        *,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """Validate any OAS component based on its type.

        Args:
            data: The component dictionary to validate
            raise_on_error: If True, raise OASSchemaValidationError on errors

        Returns:
            ValidationResult with all issues found

        Raises:
            OASSchemaValidationError: If raise_on_error=True and validation fails
        """
        component_type = data.get("component_type", "")

        if component_type == "Agent":
            return self.validate_agent(data, raise_on_error=raise_on_error)
        elif component_type == "Flow":
            return self.validate_flow(data, raise_on_error=raise_on_error)
        else:
            result = ValidationResult()
            result.add_error(
                f"Unknown or missing component_type: '{component_type}'",
                field="component_type",
                suggestion="Set component_type to 'Agent' or 'Flow'",
            )
            if raise_on_error:
                raise OASSchemaValidationError(result.errors)
            return result

    def validate_agent(
        self,
        data: dict[str, Any],
        *,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """Validate an Agent component dictionary.

        Args:
            data: The agent dictionary to validate
            raise_on_error: If True, raise OASSchemaValidationError on errors

        Returns:
            ValidationResult with all issues found

        Raises:
            OASSchemaValidationError: If raise_on_error=True and validation fails
        """
        result = ValidationResult()

        # Check required fields
        for field_name in self.AGENT_REQUIRED_FIELDS:
            if field_name not in data or not data[field_name]:
                result.add_error(
                    f"Required field '{field_name}' is missing or empty",
                    field=field_name,
                    suggestion=f"Provide a value for '{field_name}'",
                )

        # Warn about unknown fields
        known_fields = self.AGENT_REQUIRED_FIELDS | self.AGENT_OPTIONAL_FIELDS
        for field_name in data:
            if field_name not in known_fields:
                result.add_warning(
                    f"Unknown field '{field_name}' in Agent component",
                    field=field_name,
                    suggestion="This field may be ignored during conversion",
                )

        # Validate tools if present
        if "tools" in data:
            tools_result = self._validate_tools(data["tools"])
            result.merge(tools_result)

        # Validate llm_config if present
        if "llm_config" in data:
            llm_result = self._validate_llm_config(data["llm_config"])
            result.merge(llm_result)

        if raise_on_error and not result.is_valid:
            raise OASSchemaValidationError(result.errors)

        return result

    def validate_flow(
        self,
        data: dict[str, Any],
        *,
        raise_on_error: bool = False,
    ) -> ValidationResult:
        """Validate a Flow component dictionary.

        Args:
            data: The flow dictionary to validate
            raise_on_error: If True, raise OASSchemaValidationError on errors

        Returns:
            ValidationResult with all issues found

        Raises:
            OASSchemaValidationError: If raise_on_error=True and validation fails
        """
        result = ValidationResult()

        # Check required fields
        for field_name in self.FLOW_REQUIRED_FIELDS:
            if field_name not in data or not data[field_name]:
                result.add_error(
                    f"Required field '{field_name}' is missing or empty",
                    field=field_name,
                    suggestion=f"Provide a value for '{field_name}'",
                )

        # Warn about unknown fields
        known_fields = self.FLOW_REQUIRED_FIELDS | self.FLOW_OPTIONAL_FIELDS
        for field_name in data:
            if field_name not in known_fields:
                result.add_warning(
                    f"Unknown field '{field_name}' in Flow component",
                    field=field_name,
                    suggestion="This field may be ignored during conversion",
                )

        # Validate nodes if present
        if "nodes" in data:
            nodes_result = self._validate_nodes(data["nodes"])
            result.merge(nodes_result)

        # Validate edges if present
        if "edges" in data:
            edges_result = self._validate_oas_edges(data["edges"], data.get("nodes", []))
            result.merge(edges_result)

        if raise_on_error and not result.is_valid:
            raise OASSchemaValidationError(result.errors)

        return result

    def _validate_tools(self, tools: Any) -> ValidationResult:
        """Validate tools array in an Agent component."""
        result = ValidationResult()

        if not isinstance(tools, list):
            result.add_error(
                "tools must be an array",
                field="tools",
                suggestion="Provide tools as a list of tool definitions",
            )
            return result

        for i, tool in enumerate(tools):
            if isinstance(tool, dict):
                if "name" not in tool:
                    result.add_error(
                        f"Tool at index {i} is missing 'name' field",
                        field=f"tools[{i}].name",
                        suggestion="Each tool must have a name",
                    )
            elif not isinstance(tool, str):
                result.add_error(
                    f"Tool at index {i} must be a string or object",
                    field=f"tools[{i}]",
                    suggestion="Tools can be strings (tool names) or objects (full definitions)",
                )

        return result

    def _validate_llm_config(self, llm_config: Any) -> ValidationResult:
        """Validate LLM configuration in an Agent component."""
        result = ValidationResult()

        if not isinstance(llm_config, dict):
            result.add_error(
                "llm_config must be an object",
                field="llm_config",
                suggestion="Provide llm_config as a dictionary",
            )
            return result

        # Check for common LLM config fields
        valid_types = {"openai", "ollama", "vllm", "oci", "OpenAI", "Ollama", "VLLM", "OCI"}
        if "type" in llm_config and llm_config["type"] not in valid_types:
            result.add_warning(
                f"Unknown LLM type: '{llm_config['type']}'",
                field="llm_config.type",
                suggestion=f"Known types are: {', '.join(sorted(valid_types))}",
            )

        return result

    def _validate_nodes(self, nodes: Any) -> ValidationResult:
        """Validate nodes array in a Flow component."""
        result = ValidationResult()

        if not isinstance(nodes, list):
            result.add_error(
                "nodes must be an array",
                field="nodes",
                suggestion="Provide nodes as a list of node definitions",
            )
            return result

        node_ids: set[str] = set()

        for i, node in enumerate(nodes):
            if not isinstance(node, dict):
                result.add_error(
                    f"Node at index {i} must be an object",
                    field=f"nodes[{i}]",
                    suggestion="Each node must be a dictionary with at least 'id' field",
                )
                continue

            # Check for id field
            node_id = node.get("id")
            if not node_id:
                result.add_error(
                    f"Node at index {i} is missing 'id' field",
                    field=f"nodes[{i}].id",
                    suggestion="Each node must have a unique id",
                )
            else:
                if node_id in node_ids:
                    result.add_error(
                        f"Duplicate node id: '{node_id}'",
                        field=f"nodes[{i}].id",
                        suggestion="Use unique ids for each node",
                    )
                node_ids.add(node_id)

            # Validate node type if present
            node_type = node.get("type")
            if node_type and node_type not in self.VALID_NODE_TYPES:
                result.add_warning(
                    f"Unknown node type: '{node_type}'",
                    field=f"nodes[{i}].type",
                    suggestion=f"Known types: {', '.join(sorted(self.VALID_NODE_TYPES))}",
                )

        return result

    def _validate_oas_edges(self, edges: Any, nodes: list[Any]) -> ValidationResult:
        """Validate edges array in a Flow component."""
        result = ValidationResult()

        if not isinstance(edges, list):
            result.add_error(
                "edges must be an array",
                field="edges",
                suggestion="Provide edges as a list of edge definitions",
            )
            return result

        # Build set of valid node ids
        node_ids: set[str] = set()
        for n in nodes:
            if isinstance(n, dict):
                node_id = n.get("id")
                if isinstance(node_id, str):
                    node_ids.add(node_id)

        for i, edge in enumerate(edges):
            if not isinstance(edge, dict):
                result.add_error(
                    f"Edge at index {i} must be an object",
                    field=f"edges[{i}]",
                    suggestion="Each edge must be a dictionary with 'from' and 'to' fields",
                )
                continue

            # Check from field
            from_node = edge.get("from")
            if not from_node:
                result.add_error(
                    f"Edge at index {i} is missing 'from' field",
                    field=f"edges[{i}].from",
                    suggestion="Each edge must have a 'from' node reference",
                )
            elif node_ids and from_node not in node_ids:
                result.add_error(
                    f"Edge references non-existent node: '{from_node}'",
                    field=f"edges[{i}].from",
                    suggestion=f"Valid node ids: {', '.join(sorted(node_ids))}",
                )

            # Check to field
            to_node = edge.get("to")
            if not to_node:
                result.add_error(
                    f"Edge at index {i} is missing 'to' field",
                    field=f"edges[{i}].to",
                    suggestion="Each edge must have a 'to' node reference",
                )
            elif node_ids and to_node not in node_ids:
                result.add_error(
                    f"Edge references non-existent node: '{to_node}'",
                    field=f"edges[{i}].to",
                    suggestion=f"Valid node ids: {', '.join(sorted(node_ids))}",
                )

        return result


def validate_oas_dict(
    data: dict[str, Any],
    *,
    raise_on_error: bool = False,
) -> ValidationResult:
    """Convenience function to validate an OAS component dictionary.

    Args:
        data: The OAS component dictionary to validate
        raise_on_error: If True, raise OASSchemaValidationError on errors

    Returns:
        ValidationResult with all issues found

    Raises:
        OASSchemaValidationError: If raise_on_error=True and validation fails
    """
    validator = OASSchemaValidator()
    return validator.validate_component(data, raise_on_error=raise_on_error)
