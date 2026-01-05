"""Flow converter for OAS <-> Dapr Agents workflows."""

from collections.abc import Callable
from typing import Any

from pyagentspec import Component, Property  # noqa: F401

# Import flow components from correct submodules
from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.node import Node
from pyagentspec.flows.nodes import EndNode, StartNode

from dapr_agents_oas_adapter.converters.base import (
    ComponentConverter,
    ConversionError,
)
from dapr_agents_oas_adapter.converters.node import NodeConverter
from dapr_agents_oas_adapter.types import (
    NamedCallable,
    ToolRegistry,
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)
from dapr_agents_oas_adapter.utils import generate_id


class FlowConverter(ComponentConverter[Flow, WorkflowDefinition]):
    """Converter for OAS Flow <-> Dapr Workflow definition.

    Handles conversion of OAS Flows with their nodes and edges
    to Dapr workflow definitions that can be executed using
    Dapr's workflow engine.
    """

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        """Initialize the converter."""
        super().__init__(tool_registry)
        self._node_converter = NodeConverter(tool_registry)

    def from_oas(self, component: Flow) -> WorkflowDefinition:
        """Convert an OAS Flow to a Dapr WorkflowDefinition.

        Args:
            component: The OAS Flow to convert

        Returns:
            WorkflowDefinition with equivalent structure

        Raises:
            ConversionError: If the flow cannot be converted
        """
        self.validate_oas_component(component)

        # Extract nodes
        tasks = self._convert_nodes(component)

        # Extract edges
        edges = self._convert_edges(component)

        # Find start and end nodes
        start_node = self._find_start_node(component)
        end_nodes = self._find_end_nodes(component)

        # Extract inputs/outputs from flow
        inputs = self._extract_properties(getattr(component, "inputs", []))
        outputs = self._extract_properties(getattr(component, "outputs", []))

        return WorkflowDefinition(
            name=component.name,
            description=component.description,
            tasks=tasks,
            edges=edges,
            start_node=start_node,
            end_nodes=end_nodes,
            inputs=inputs,
            outputs=outputs,
        )

    def to_oas(self, component: WorkflowDefinition) -> Flow:
        """Convert a Dapr WorkflowDefinition to an OAS Flow.

        Args:
            component: The Dapr WorkflowDefinition to convert

        Returns:
            OAS Flow with equivalent structure
        """
        flow_id = generate_id("flow")

        # Convert tasks to nodes
        nodes: list[Node] = []
        node_map: dict[str, Node] = {}

        for task in component.tasks:
            node = self._node_converter.to_oas(task)
            nodes.append(node)
            node_map[task.name] = node

        # Convert edges
        control_edges: list[ControlFlowEdge] = []
        data_edges: list[DataFlowEdge] = []

        for edge in component.edges:
            from_node = node_map.get(edge.from_node)
            to_node = node_map.get(edge.to_node)

            if from_node and to_node:
                # Create control flow edge
                control_edge = ControlFlowEdge(
                    id=generate_id("edge"),
                    name=f"{edge.from_node}_to_{edge.to_node}",
                    from_node=from_node,
                    to_node=to_node,
                    from_branch=edge.from_branch,
                )
                control_edges.append(control_edge)

                # Create data flow edges for mappings
                for source_output, dest_input in edge.data_mapping.items():
                    data_edge = DataFlowEdge(
                        id=generate_id("data_edge"),
                        name=f"{source_output}_to_{dest_input}",
                        source_node=from_node,
                        source_output=source_output,
                        destination_node=to_node,
                        destination_input=dest_input,
                    )
                    data_edges.append(data_edge)

        # Find start node - create a default if none exists
        start_node_obj = node_map.get(component.start_node) if component.start_node else None
        if not start_node_obj and nodes:
            # Default to first node
            start_node_obj = nodes[0]
        if not start_node_obj:
            # Create a minimal start node if no nodes exist
            start_node_obj = StartNode(
                id=generate_id("start"),
                name="start",
            )
            nodes.insert(0, start_node_obj)

        # Convert inputs/outputs to Property objects
        flow_inputs = self._dicts_to_properties(component.inputs)
        flow_outputs = self._dicts_to_properties(component.outputs)

        return Flow(
            id=flow_id,
            name=component.name,
            description=component.description,
            start_node=start_node_obj,
            nodes=nodes,
            control_flow_connections=control_edges,
            data_flow_connections=data_edges,
            inputs=flow_inputs if flow_inputs else None,
            outputs=flow_outputs if flow_outputs else None,
        )

    def can_convert(self, component: Any) -> bool:
        """Check if this converter can handle the given component."""
        if isinstance(component, Flow):
            return True
        if isinstance(component, WorkflowDefinition):
            return True
        if isinstance(component, dict):
            comp_type = component.get("component_type", "")
            return comp_type == "Flow"
        return False

    def from_dict(self, flow_dict: dict[str, Any]) -> WorkflowDefinition:
        """Convert a dictionary representation to WorkflowDefinition."""
        # Get referenced components for resolving $component_ref
        referenced_components = flow_dict.get("$referenced_components", {})

        # Convert nodes - resolve component references if needed
        tasks: list[WorkflowTaskDefinition] = []
        nodes = flow_dict.get("nodes", [])
        node_id_map: dict[str, str] = {}  # Map node IDs to names
        unresolved_node_ids: set[str] = set()  # Track unresolved references

        for node in nodes:
            # Resolve $component_ref if present
            if isinstance(node, dict) and "$component_ref" in node:
                node_id = node["$component_ref"]
                resolved_node = referenced_components.get(node_id, {})
                if not resolved_node:
                    # Track unresolved reference to skip related edges
                    unresolved_node_ids.add(node_id)
                    continue
                node = resolved_node
            task = self._node_converter.from_dict(node)
            tasks.append(task)
            node_id_map[node.get("id", "")] = task.name

        # Convert edges
        edges: list[WorkflowEdgeDefinition] = []

        # Control flow edges
        for edge in flow_dict.get("control_flow_connections", []):
            from_ref = edge.get("from_node", {})
            to_ref = edge.get("to_node", {})

            from_id = from_ref.get("$component_ref", "") if isinstance(from_ref, dict) else ""
            to_id = to_ref.get("$component_ref", "") if isinstance(to_ref, dict) else ""

            # Skip edges that reference unresolved nodes
            if from_id in unresolved_node_ids or to_id in unresolved_node_ids:
                continue

            # Only create edge if both nodes are resolved
            from_name = node_id_map.get(from_id)
            to_name = node_id_map.get(to_id)
            if from_name is None or to_name is None:
                continue  # Skip edges referencing unknown nodes

            edges.append(
                WorkflowEdgeDefinition(
                    from_node=from_name,
                    to_node=to_name,
                    from_branch=edge.get("from_branch"),
                )
            )

        # Data flow edges
        data_mappings: dict[tuple[str, str], dict[str, str]] = {}
        for edge in flow_dict.get("data_flow_connections", []):
            source_ref = edge.get("source_node", {})
            dest_ref = edge.get("destination_node", {})

            source_id = source_ref.get("$component_ref", "") if isinstance(source_ref, dict) else ""
            dest_id = dest_ref.get("$component_ref", "") if isinstance(dest_ref, dict) else ""

            # Skip data flow edges that reference unresolved nodes
            if source_id in unresolved_node_ids or dest_id in unresolved_node_ids:
                continue

            # Only create data mapping if both nodes are resolved
            source_name = node_id_map.get(source_id)
            dest_name = node_id_map.get(dest_id)
            if source_name is None or dest_name is None:
                continue

            key = (source_name, dest_name)
            if key not in data_mappings:
                data_mappings[key] = {}

            data_mappings[key][edge.get("source_output", "")] = edge.get("destination_input", "")

        # Merge data mappings into edges
        for edge in edges:
            key = (edge.from_node, edge.to_node)
            if key in data_mappings:
                edge.data_mapping = data_mappings[key]

        # Find start node
        start_ref = flow_dict.get("start_node", {})
        start_id = start_ref.get("$component_ref", "") if isinstance(start_ref, dict) else ""
        start_node = node_id_map.get(start_id, "")

        # Find end nodes
        end_nodes: list[str] = []
        for task in tasks:
            if task.task_type == "end":
                end_nodes.append(task.name)

        return WorkflowDefinition(
            name=flow_dict.get("name", ""),
            description=flow_dict.get("description"),
            tasks=tasks,
            edges=edges,
            start_node=start_node,
            end_nodes=end_nodes,
            inputs=flow_dict.get("inputs", []),
            outputs=flow_dict.get("outputs", []),
        )

    def to_dict(self, workflow_def: WorkflowDefinition) -> dict[str, Any]:
        """Convert WorkflowDefinition to a dictionary representation."""
        # Build referenced components
        referenced: dict[str, Any] = {}
        nodes: list[dict[str, Any]] = []

        for task in workflow_def.tasks:
            node_dict = self._node_converter.to_dict(task)
            node_id = node_dict["id"]
            referenced[node_id] = node_dict
            nodes.append({"$component_ref": node_id})

        # Build control flow edges
        control_edges: list[dict[str, Any]] = []
        data_edges: list[dict[str, Any]] = []

        # Map task names to IDs
        name_to_id: dict[str, str] = {}
        for node_id, node_data in referenced.items():
            name_to_id[node_data["name"]] = node_id

        for edge in workflow_def.edges:
            from_id = name_to_id.get(edge.from_node, edge.from_node)
            to_id = name_to_id.get(edge.to_node, edge.to_node)

            control_edge: dict[str, Any] = {
                "component_type": "ControlFlowEdge",
                "id": generate_id("edge"),
                "name": f"{edge.from_node}_to_{edge.to_node}",
                "from_node": {"$component_ref": from_id},
                "to_node": {"$component_ref": to_id},
                "from_branch": edge.from_branch,
            }
            control_edges.append(control_edge)

            # Create data flow edges
            for source_output, dest_input in edge.data_mapping.items():
                data_edge: dict[str, Any] = {
                    "component_type": "DataFlowEdge",
                    "id": generate_id("data_edge"),
                    "name": f"{source_output}_to_{dest_input}",
                    "source_node": {"$component_ref": from_id},
                    "source_output": source_output,
                    "destination_node": {"$component_ref": to_id},
                    "destination_input": dest_input,
                }
                data_edges.append(data_edge)

        # Find start node ID
        start_node_id = name_to_id.get(workflow_def.start_node or "", "")

        return {
            "component_type": "Flow",
            "id": generate_id("flow"),
            "name": workflow_def.name,
            "description": workflow_def.description,
            "inputs": workflow_def.inputs,
            "outputs": workflow_def.outputs,
            "start_node": {"$component_ref": start_node_id} if start_node_id else None,
            "nodes": list(nodes),
            "control_flow_connections": control_edges,
            "data_flow_connections": data_edges,
            "$referenced_components": referenced,
            "agentspec_version": "25.4.1",
        }

    def create_dapr_workflow(
        self,
        workflow_def: WorkflowDefinition,
        task_implementations: dict[str, Callable[..., Any]] | None = None,
    ) -> NamedCallable:
        """Create a Dapr workflow function from a WorkflowDefinition.

        Args:
            workflow_def: The workflow definition
            task_implementations: Optional task implementations

        Returns:
            A workflow function that can be registered with Dapr

        Raises:
            ConversionError: If workflow creation fails
        """
        try:
            # Build the task execution order from edges
            execution_order = self._build_execution_order(workflow_def)

            # Create the workflow function
            def workflow_function(ctx: Any, input_params: dict[str, Any]) -> Any:
                """Generated Dapr workflow function."""
                results: dict[str, Any] = {"__input__": input_params}

                for task_name in execution_order:
                    task = next((t for t in workflow_def.tasks if t.name == task_name), None)
                    if not task:
                        continue

                    # Skip start/end nodes
                    if task.task_type in ("start", "end"):
                        continue

                    # Get task input from previous results
                    task_input = self._build_task_input(task, results, workflow_def.edges)

                    # Execute task
                    if task_implementations and task_name in task_implementations:
                        impl = task_implementations[task_name]
                        result = impl(**task_input)
                    else:
                        # Use ctx.call_activity for Dapr workflow
                        result = ctx.call_activity(task_name, input=task_input)

                    results[task_name] = result

                # Return final outputs
                return self._build_workflow_output(workflow_def, results)

            # Set function metadata
            workflow_function.__name__ = workflow_def.name
            workflow_function.__doc__ = workflow_def.description or f"Workflow: {workflow_def.name}"

            return workflow_function

        except Exception as e:
            raise ConversionError(
                f"Failed to create Dapr workflow: {e}",
                workflow_def,
            ) from e

    def generate_workflow_code(self, workflow_def: WorkflowDefinition) -> str:
        """Generate Python code for a Dapr workflow.

        Args:
            workflow_def: The workflow definition

        Returns:
            Python code string that can be executed
        """
        lines: list[str] = [
            "from dapr_agents.workflow import WorkflowApp, workflow, task",
            "from dapr_agents.types import DaprWorkflowContext",
            "",
            "",
        ]

        # Generate task functions
        for task in workflow_def.tasks:
            if task.task_type in ("start", "end"):
                continue

            lines.append(f"@task(name='{task.name}')")
            lines.append(f"def {task.name}_task(ctx, input_data: dict) -> dict:")
            lines.append(f'    """Task: {task.name}"""')

            if task.task_type == "llm":
                prompt = task.config.get("prompt_template", "")
                lines.append(f"    # LLM task with prompt: {prompt[:50]}...")
                lines.append("    # TODO: Implement LLM call")
                lines.append("    return {'result': 'TODO'}")
            elif task.task_type == "tool":
                tool_name = task.config.get("tool_name", task.name)
                lines.append(f"    # Tool task: {tool_name}")
                lines.append("    # TODO: Implement tool call")
                lines.append("    return {'result': 'TODO'}")
            else:
                lines.append("    # TODO: Implement task logic")
                lines.append("    return input_data")

            lines.append("")

        # Generate workflow function
        lines.append(f"@workflow(name='{workflow_def.name}')")
        func_name = f"{workflow_def.name}_workflow"
        lines.append(f"def {func_name}(ctx: DaprWorkflowContext, input_params: dict) -> dict:")
        lines.append(f'    """{workflow_def.description or workflow_def.name}"""')

        # Build execution order
        execution_order = self._build_execution_order(workflow_def)

        for task_name in execution_order:
            task_or_none = next((t for t in workflow_def.tasks if t.name == task_name), None)
            if not task_or_none or task_or_none.task_type in ("start", "end"):
                continue
            task = task_or_none

            lines.append(f"    {task_name}_result = yield ctx.call_activity(")
            lines.append(f"        {task_name}_task,")
            lines.append(f"        input={{'task': '{task_name}'}}")
            lines.append("    )")
            lines.append("")

        lines.append("    return {'status': 'completed'}")

        return "\n".join(lines)

    def _convert_nodes(self, flow: Flow) -> list[WorkflowTaskDefinition]:
        """Convert OAS flow nodes to workflow tasks."""
        tasks: list[WorkflowTaskDefinition] = []
        nodes = getattr(flow, "nodes", [])

        for node in nodes:
            if node:
                task = self._node_converter.from_oas(node)
                tasks.append(task)

        return tasks

    def _convert_edges(self, flow: Flow) -> list[WorkflowEdgeDefinition]:
        """Convert OAS flow edges to workflow edge definitions."""
        edges: list[WorkflowEdgeDefinition] = []

        # Process control flow edges
        control_edges = getattr(flow, "control_flow_connections", [])
        for edge in control_edges:
            if edge:
                from_node = getattr(edge, "from_node", None)
                to_node = getattr(edge, "to_node", None)

                edges.append(
                    WorkflowEdgeDefinition(
                        from_node=getattr(from_node, "name", "") if from_node else "",
                        to_node=getattr(to_node, "name", "") if to_node else "",
                        from_branch=getattr(edge, "from_branch", None),
                    )
                )

        # Process data flow edges and merge into control flow edges
        data_edges = getattr(flow, "data_flow_connections", [])
        for data_edge in data_edges:
            if not data_edge:
                continue

            source_node = getattr(data_edge, "source_node", None)
            dest_node = getattr(data_edge, "destination_node", None)
            source_name = getattr(source_node, "name", "") if source_node else ""
            dest_name = getattr(dest_node, "name", "") if dest_node else ""

            # Find matching control edge or create new one
            matching = next(
                (e for e in edges if e.from_node == source_name and e.to_node == dest_name),
                None,
            )

            if matching:
                matching.data_mapping[getattr(data_edge, "source_output", "")] = getattr(
                    data_edge, "destination_input", ""
                )
            else:
                edges.append(
                    WorkflowEdgeDefinition(
                        from_node=source_name,
                        to_node=dest_name,
                        data_mapping={
                            getattr(data_edge, "source_output", ""): getattr(
                                data_edge, "destination_input", ""
                            )
                        },
                    )
                )

        return edges

    def _find_start_node(self, flow: Flow) -> str | None:
        """Find the start node name in a flow."""
        start_node = getattr(flow, "start_node", None)
        if start_node:
            return getattr(start_node, "name", None)

        # Look for StartNode in nodes
        nodes = getattr(flow, "nodes", [])
        for node in nodes:
            if isinstance(node, StartNode):
                return node.name

        return None

    def _find_end_nodes(self, flow: Flow) -> list[str]:
        """Find all end node names in a flow."""
        end_nodes: list[str] = []
        nodes = getattr(flow, "nodes", [])

        for node in nodes:
            if isinstance(node, EndNode):
                end_nodes.append(node.name)

        return end_nodes

    def _extract_properties(self, props: list[Any]) -> list[dict[str, Any]]:
        """Extract property schemas."""
        result: list[dict[str, Any]] = []
        for prop in props:
            if isinstance(prop, dict):
                result.append(prop)
            elif hasattr(prop, "model_dump"):
                result.append(prop.model_dump())
        return result

    def _dicts_to_properties(self, props: list[dict[str, Any]]) -> list[Property]:
        """Convert dictionary property schemas to Property objects."""
        result: list[Property] = []
        for prop in props:
            if isinstance(prop, dict):
                result.append(
                    Property(
                        title=prop.get("title", ""),
                        type=prop.get("type", "string"),
                        description=prop.get("description"),
                        default=prop.get("default"),
                    )
                )
            elif isinstance(prop, Property):
                result.append(prop)
        return result

    def _build_execution_order(self, workflow_def: WorkflowDefinition) -> list[str]:
        """Build topological execution order from edges."""
        # Simple topological sort
        in_degree: dict[str, int] = {t.name: 0 for t in workflow_def.tasks}
        adjacency: dict[str, list[str]] = {t.name: [] for t in workflow_def.tasks}

        for edge in workflow_def.edges:
            if edge.to_node in in_degree:
                in_degree[edge.to_node] += 1
            if edge.from_node in adjacency:
                adjacency[edge.from_node].append(edge.to_node)

        # Start with nodes that have no incoming edges
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order: list[str] = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for neighbor in adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def _build_task_input(
        self,
        task: WorkflowTaskDefinition,
        results: dict[str, Any],
        edges: list[WorkflowEdgeDefinition],
    ) -> dict[str, Any]:
        """Build input for a task from previous results."""
        task_input: dict[str, Any] = {}

        for edge in edges:
            if edge.to_node == task.name:
                source_result = results.get(edge.from_node, {})
                for source_key, dest_key in edge.data_mapping.items():
                    if isinstance(source_result, dict) and source_key in source_result:
                        task_input[dest_key] = source_result[source_key]

        return task_input

    def _build_workflow_output(
        self,
        workflow_def: WorkflowDefinition,
        results: dict[str, Any],
    ) -> dict[str, Any]:
        """Build final workflow output from task results."""
        output: dict[str, Any] = {}

        # Collect outputs from end nodes
        for end_node in workflow_def.end_nodes:
            if end_node in results:
                result = results[end_node]
                if isinstance(result, dict):
                    output.update(result)
                else:
                    output[end_node] = result

        # If no outputs from end nodes, return explicit empty result
        # instead of unpredictable dictionary value
        if not output:
            return {"status": "completed", "output": None}

        return output
