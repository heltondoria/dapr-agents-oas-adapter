"""Flow converter for OAS <-> Dapr Agents workflows."""

import json
from collections.abc import Callable
from datetime import timedelta
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
            flow_id=getattr(component, "id", None),
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
        flow_id = component.flow_id or generate_id("flow")

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

    def from_dict(
        self, flow_dict: dict[str, Any], *, _visited_flows: set[str] | None = None
    ) -> WorkflowDefinition:
        """Convert a dictionary representation to WorkflowDefinition."""
        visited_flows = _visited_flows or set()
        flow_id = flow_dict.get("id")
        if flow_id:
            if flow_id in visited_flows:
                return WorkflowDefinition(name=flow_dict.get("name", ""), flow_id=flow_id)
            visited_flows.add(flow_id)
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

        subflows = self._extract_subflows(referenced_components, visited_flows)

        return WorkflowDefinition(
            name=flow_dict.get("name", ""),
            description=flow_dict.get("description"),
            flow_id=flow_dict.get("id"),
            tasks=tasks,
            edges=edges,
            start_node=start_node,
            end_nodes=end_nodes,
            inputs=flow_dict.get("inputs", []),
            outputs=flow_dict.get("outputs", []),
            subflows=subflows,
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

        for subflow_id, subflow_def in workflow_def.subflows.items():
            subflow_dict = self.to_dict(subflow_def)
            referenced_key = subflow_def.flow_id or subflow_id
            referenced[referenced_key] = subflow_dict

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

        flow_id = workflow_def.flow_id or generate_id("flow")
        return {
            "component_type": "Flow",
            "id": flow_id,
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
        *,
        _visited_workflows: set[str] | None = None,
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
            import dapr.ext.workflow as wf
        except Exception as e:
            raise ConversionError(f"Failed to import Dapr workflow SDK: {e}", workflow_def) from e

        try:
            visited_workflows = _visited_workflows or set()
            flow_key = workflow_def.flow_id or workflow_def.name
            visited_workflows.add(flow_key)

            def _make_activity_stub(name: str) -> Callable[[Any, Any], Any]:
                """Create a callable placeholder used only to reference an activity by name."""

                def _activity(_: Any, __: Any = None) -> Any:  # pragma: no cover
                    raise RuntimeError("This stub should never be executed directly.")

                _activity.__name__ = name
                _activity.__qualname__ = name
                return _activity

            activity_stubs: dict[str, Callable[[Any, Any], Any]] = {}

            def _get_activity_stub(name: str) -> Callable[[Any, Any], Any]:
                if name not in activity_stubs:
                    activity_stubs[name] = _make_activity_stub(name)
                return activity_stubs[name]

            tasks_by_name: dict[str, WorkflowTaskDefinition] = {
                task.name: task for task in workflow_def.tasks
            }
            outgoing_edges: dict[str, list[WorkflowEdgeDefinition]] = {}
            incoming_edges: dict[str, list[WorkflowEdgeDefinition]] = {}
            for edge in workflow_def.edges:
                outgoing_edges.setdefault(edge.from_node, []).append(edge)
                incoming_edges.setdefault(edge.to_node, []).append(edge)

            def _is_default_branch(branch: str | None) -> bool:
                if branch is None:
                    return True
                return branch.strip().lower() in ("", "default", "next")

            def _extract_branch_value(task: WorkflowTaskDefinition, result: Any) -> str | None:
                key = task.config.get("branch_output_key")
                if key and isinstance(result, dict) and key in result:
                    value = result.get(key)
                    return str(value) if value is not None else None
                if isinstance(result, dict):
                    for candidate in ("branch", "branch_name", "__branch__"):
                        if candidate in result:
                            value = result.get(candidate)
                            return str(value) if value is not None else None
                if isinstance(result, str):
                    return result
                return None

            def _select_next_edges(
                task: WorkflowTaskDefinition, result: Any
            ) -> list[WorkflowEdgeDefinition]:
                edges = outgoing_edges.get(task.name, [])
                if not edges:
                    return []
                has_branches = any(edge.from_branch for edge in edges)
                if not has_branches:
                    return edges
                branch_value = _extract_branch_value(task, result)
                if branch_value is not None:
                    matching = [e for e in edges if e.from_branch == branch_value]
                    if matching:
                        return matching
                default_edges = [e for e in edges if _is_default_branch(e.from_branch)]
                return default_edges if default_edges else []

            def _build_retry_policy(config: dict[str, Any]) -> Any | None:
                if not hasattr(wf, "RetryPolicy"):
                    return None
                retry_config = config.get("retry_policy")
                if retry_config is None:
                    return None
                if hasattr(retry_config, "max_attempts"):
                    max_attempts = getattr(retry_config, "max_attempts", 1)
                    initial_backoff = getattr(retry_config, "initial_backoff_seconds", 5)
                    max_backoff = getattr(retry_config, "max_backoff_seconds", 30)
                    multiplier = getattr(retry_config, "backoff_multiplier", 1.5)
                    retry_timeout = getattr(retry_config, "retry_timeout", None)
                elif isinstance(retry_config, dict):
                    max_attempts = retry_config.get("max_attempts", 1)
                    initial_backoff = retry_config.get("initial_backoff_seconds", 5)
                    max_backoff = retry_config.get("max_backoff_seconds", 30)
                    multiplier = retry_config.get("backoff_multiplier", 1.5)
                    retry_timeout = retry_config.get("retry_timeout")
                else:
                    return None
                return wf.RetryPolicy(
                    max_number_of_attempts=max_attempts,
                    first_retry_interval=timedelta(seconds=initial_backoff),
                    max_retry_interval=timedelta(seconds=max_backoff),
                    backoff_coefficient=multiplier,
                    retry_timeout=timedelta(seconds=retry_timeout)
                    if retry_timeout
                    else None,
                )

            def _call_activity(
                ctx: Any, activity: Callable[[Any, Any], Any], input_data: dict[str, Any], retry: Any
            ) -> Any:
                kwargs: dict[str, Any] = {"input": input_data}
                if retry is not None:
                    kwargs["retry_policy"] = retry
                try:
                    return ctx.call_activity(activity, **kwargs)
                except TypeError:
                    kwargs.pop("retry_policy", None)
                    return ctx.call_activity(activity, **kwargs)

            def _call_child_workflow(
                ctx: Any, workflow_name: str, input_data: dict[str, Any], retry: Any
            ) -> Any:
                if not hasattr(ctx, "call_child_workflow"):
                    raise RuntimeError("call_child_workflow is not available in this SDK.")
                kwargs: dict[str, Any] = {"input": input_data}
                if retry is not None:
                    kwargs["retry_policy"] = retry
                try:
                    return ctx.call_child_workflow(workflow_name, **kwargs)
                except TypeError:
                    kwargs.pop("retry_policy", None)
                    return ctx.call_child_workflow(workflow_name, **kwargs)

            def _await_with_timeout(
                ctx: Any, task_obj: Any, timeout_seconds: int | None
            ) -> Any:
                if not timeout_seconds or not hasattr(ctx, "create_timer"):
                    return (yield task_obj)
                timeout_task = ctx.create_timer(timedelta(seconds=timeout_seconds))
                winner = yield wf.when_any([task_obj, timeout_task])
                if winner == timeout_task:
                    raise TimeoutError("Task execution timed out.")
                return (yield task_obj)

            def _resolve_flow_name(task: WorkflowTaskDefinition, key: str) -> str:
                config_value = task.config.get(key) or task.config.get("flow_name")
                if config_value:
                    return str(config_value)
                config_value = task.config.get("flow_id") or task.config.get("inner_flow_id")
                if config_value:
                    return str(config_value)
                return task.name

            def _extract_map_items(task: WorkflowTaskDefinition, task_input: dict[str, Any]) -> list[Any]:
                map_key = task.config.get("map_input_key") or "items"
                if map_key in task_input:
                    items = task_input.get(map_key)
                elif len(task.inputs) == 1 and task.inputs[0] in task_input:
                    items = task_input.get(task.inputs[0])
                else:
                    items = task_input.get("items")
                if not isinstance(items, list):
                    raise ValueError(f"MapNode '{task.name}' expects a list for '{map_key}'.")
                return items

            def _build_map_item_input(
                task: WorkflowTaskDefinition, task_input: dict[str, Any], item: Any
            ) -> dict[str, Any]:
                map_key = task.config.get("map_input_key") or "items"
                item_key = task.config.get("map_item_key") or "item"
                base = {k: v for k, v in task_input.items() if k != map_key}
                if isinstance(item, dict):
                    return {**base, **item}
                return {**base, item_key: item}

            def _get_compensation_activity(task: WorkflowTaskDefinition) -> str | None:
                for key in ("compensation_activity", "compensating_activity", "compensation_task"):
                    if key in task.config:
                        return str(task.config[key])
                on_error = task.config.get("on_error")
                if isinstance(on_error, dict):
                    for key in ("compensation_activity", "compensation_task", "activity"):
                        if key in on_error:
                            return str(on_error[key])
                return None

            def _execute_compensations(
                ctx: Any,
                executed: list[str],
                results: dict[str, Any],
                error: Exception,
            ) -> Any:
                for task_name in reversed(executed):
                    task = tasks_by_name.get(task_name)
                    if not task:
                        continue
                    compensation = _get_compensation_activity(task)
                    if not compensation:
                        continue
                    payload = {
                        "task": task_name,
                        "error": str(error),
                        "result": results.get(task_name),
                    }
                    extra = task.config.get("compensation_input")
                    if isinstance(extra, dict):
                        payload.update(extra)
                    try:
                        yield _call_activity(ctx, _get_activity_stub(compensation), payload, None)
                    except Exception:
                        continue

            def _execute_task(
                ctx: Any, task: WorkflowTaskDefinition, task_input: dict[str, Any]
            ) -> Any:
                timeout_seconds = task.config.get("timeout_seconds")
                retry_policy = _build_retry_policy(task.config)
                if task_implementations and task.name in task_implementations:
                    impl = task_implementations[task.name]
                    return impl(**task_input)

                if task.task_type == "flow":
                    workflow_name = _resolve_flow_name(task, "flow_name")
                    if hasattr(ctx, "call_child_workflow"):
                        task_obj = _call_child_workflow(ctx, workflow_name, task_input, retry_policy)
                        return (yield from _await_with_timeout(ctx, task_obj, timeout_seconds))
                    activity = _get_activity_stub(task.name)
                    task_obj = _call_activity(ctx, activity, task_input, retry_policy)
                    return (yield from _await_with_timeout(ctx, task_obj, timeout_seconds))

                if task.task_type == "map":
                    items = _extract_map_items(task, task_input)
                    parallel = bool(task.config.get("parallel", True))
                    workflow_name = _resolve_flow_name(task, "inner_flow_id")
                    if hasattr(ctx, "call_child_workflow"):
                        tasks = [
                            _call_child_workflow(
                                ctx,
                                workflow_name,
                                _build_map_item_input(task, task_input, item),
                                retry_policy,
                            )
                            for item in items
                        ]
                        if parallel:
                            if timeout_seconds:
                                all_task = wf.when_all(tasks)
                                return (yield from _await_with_timeout(ctx, all_task, timeout_seconds))
                            return (yield wf.when_all(tasks))
                        results_list = []
                        for t in tasks:
                            results_list.append((yield from _await_with_timeout(ctx, t, timeout_seconds)))
                        return results_list
                    activity = _get_activity_stub(task.name)
                    task_obj = _call_activity(ctx, activity, task_input, retry_policy)
                    return (yield from _await_with_timeout(ctx, task_obj, timeout_seconds))

                activity = _get_activity_stub(task.name)
                task_obj = _call_activity(ctx, activity, task_input, retry_policy)
                return (yield from _await_with_timeout(ctx, task_obj, timeout_seconds))

            child_workflows: list[NamedCallable] = []
            for subflow_def in workflow_def.subflows.values():
                subflow_key = subflow_def.flow_id or subflow_def.name
                if subflow_key in visited_workflows:
                    continue
                child_workflows.append(
                    self.create_dapr_workflow(
                        subflow_def,
                        task_implementations,
                        _visited_workflows=visited_workflows,
                    )
                )

            def workflow_function(ctx: Any, input_params: dict[str, Any]) -> Any:
                """Generated Dapr workflow function."""
                results: dict[str, Any] = {"__input__": input_params}
                executed: list[str] = []
                executed_set: set[str] = set()
                pending_queue: list[str] = []
                pending_inputs: dict[str, list[WorkflowEdgeDefinition]] = {}

                def _enqueue_edge(edge: WorkflowEdgeDefinition) -> None:
                    if edge.to_node in executed_set:
                        return
                    pending_inputs.setdefault(edge.to_node, []).append(edge)
                    if edge.to_node not in pending_queue:
                        pending_queue.append(edge.to_node)

                # Seed with start node outputs.
                if workflow_def.start_node:
                    results[workflow_def.start_node] = input_params
                    for edge in outgoing_edges.get(workflow_def.start_node, []):
                        _enqueue_edge(edge)
                else:
                    # If no explicit start node, schedule nodes with no incoming edges.
                    for task_name in tasks_by_name:
                        if task_name not in incoming_edges:
                            pending_queue.append(task_name)

                try:
                    while pending_queue:
                        task_name = pending_queue.pop(0)
                        if task_name in executed_set:
                            continue
                        task = tasks_by_name.get(task_name)
                        if not task:
                            continue
                        edges_for_task = pending_inputs.pop(task_name, [])
                        task_input = self._build_task_input(task, results, edges_for_task)

                        if task.task_type == "start" or task.task_type == "end":
                            results[task_name] = task_input
                        else:
                            result = yield from _execute_task(ctx, task, task_input)
                            results[task_name] = result

                        executed.append(task_name)
                        executed_set.add(task_name)

                        next_edges = _select_next_edges(task, results.get(task_name))
                        for edge in next_edges:
                            _enqueue_edge(edge)
                except Exception as exc:
                    yield from _execute_compensations(ctx, executed, results, exc)
                    raise

                return self._build_workflow_output(workflow_def, results)

            workflow_function.__name__ = workflow_def.name
            workflow_function.__doc__ = workflow_def.description or f"Workflow: {workflow_def.name}"
            workflow_function.child_workflows = child_workflows

            return workflow_function

        except Exception as e:
            raise ConversionError(f"Failed to create Dapr workflow: {e}", workflow_def) from e

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

    def _extract_subflows(
        self,
        referenced_components: dict[str, Any],
        visited_flows: set[str],
    ) -> dict[str, WorkflowDefinition]:
        """Extract referenced subflows into WorkflowDefinition objects."""
        subflows: dict[str, WorkflowDefinition] = {}
        for comp_id, component in referenced_components.items():
            if not isinstance(component, dict):
                continue
            if component.get("component_type") != "Flow":
                continue
            if comp_id in visited_flows:
                continue
            try:
                subflow_def = self.from_dict(component, _visited_flows=visited_flows)
            except Exception:
                continue
            subflows[comp_id] = subflow_def
        return subflows

    def _dicts_to_properties(self, props: list[dict[str, Any]]) -> list[Property]:
        """Convert dictionary property schemas to Property objects."""
        result: list[Property] = []
        for prop in props:
            if isinstance(prop, dict):
                description = prop.get("description")
                result.append(
                    Property(
                        title=prop.get("title", ""),
                        type=prop.get("type", "string"),
                        # `pyagentspec.Property` validates JSON schema; do not pass None.
                        description=str(description) if description is not None else "",
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
        """Build input for a task from selected edges."""
        task_input: dict[str, Any] = {}

        for edge in edges:
            if edge.to_node == task.name:
                source_result = results.get(edge.from_node, {})
                if source_result is None:
                    source_result = {}
                if isinstance(source_result, str):
                    parsed = self._try_parse_json_dict(source_result)
                    if parsed is not None:
                        source_result = parsed
                if not isinstance(source_result, dict):
                    # Normalize scalar results (e.g., str) so they can be mapped.
                    source_result = {"result": source_result}
                for source_key, dest_key in edge.data_mapping.items():
                    if isinstance(source_result, dict) and source_key in source_result:
                        task_input[dest_key] = source_result[source_key]

        return task_input

    @staticmethod
    def _try_parse_json_dict(value: str) -> dict[str, Any] | None:
        """Best-effort parse of a JSON object (dict) serialized as a string."""
        raw = value.strip()
        if not raw:
            return None
        if raw[0] != "{":
            return None
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

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
