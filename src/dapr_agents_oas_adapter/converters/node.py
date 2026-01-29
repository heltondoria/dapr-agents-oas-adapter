"""Node converters for OAS <-> Dapr Agents workflow tasks."""

from typing import Any

from pyagentspec import Component, Property  # noqa: F401

# Import flow components from correct submodules
from pyagentspec.flows.node import Node
from pyagentspec.flows.nodes import (
    AgentNode,
    EndNode,
    FlowNode,
    LlmNode,
    MapNode,
    StartNode,
    ToolNode,
)

from dapr_agents_oas_adapter.converters.base import (
    ComponentConverter,
)
from dapr_agents_oas_adapter.types import (
    ToolRegistry,
    WorkflowTaskDefinition,
)
from dapr_agents_oas_adapter.utils import generate_id


class NodeConverter(ComponentConverter[Node, WorkflowTaskDefinition]):
    """Converter for OAS Node <-> Dapr Workflow task definition.

    Handles conversion of various node types (LlmNode, ToolNode, AgentNode, etc.)
    to Dapr workflow task definitions.
    """

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        """Initialize the converter."""
        super().__init__(tool_registry)

    def from_oas(self, component: Node) -> WorkflowTaskDefinition:
        """Convert an OAS Node to a Dapr WorkflowTaskDefinition.

        Args:
            component: The OAS Node to convert

        Returns:
            WorkflowTaskDefinition with equivalent settings
        """
        self.validate_oas_component(component)

        node_type = type(component).__name__
        task_type = self._get_task_type(node_type)
        config = self._extract_node_config(component)
        inputs = self._extract_input_names(component)
        outputs = self._extract_output_names(component)

        return WorkflowTaskDefinition(
            name=component.name,
            task_type=task_type,
            config=config,
            inputs=inputs,
            outputs=outputs,
        )

    def to_oas(self, component: WorkflowTaskDefinition) -> Node:
        """Convert a Dapr WorkflowTaskDefinition to an OAS Node.

        Args:
            component: The Dapr WorkflowTaskDefinition to convert

        Returns:
            OAS Node with equivalent settings
        """
        from pyagentspec.llms import VllmConfig
        from pyagentspec.tools import ServerTool

        node_id = generate_id("node")
        node_class = self._get_node_class(component.task_type)

        # Build inputs/outputs as Property objects
        inputs = self._dicts_to_properties(component.inputs)
        outputs = self._dicts_to_properties(component.outputs)

        if node_class == StartNode:
            return StartNode(
                id=node_id,
                name=component.name,
                inputs=inputs if inputs else None,
                outputs=inputs if inputs else None,  # StartNode passes inputs as outputs
            )
        elif node_class == EndNode:
            return EndNode(
                id=node_id,
                name=component.name,
                inputs=inputs if inputs else None,
                outputs=outputs if outputs else None,
            )
        elif node_class == LlmNode:
            # Get or create default LLM config
            llm_config = component.config.get("llm_config")
            if llm_config is None:
                llm_config = VllmConfig(
                    id=generate_id("llm"),
                    name="default_llm",
                    model_id="gpt-4",
                    url="https://api.openai.com/v1",
                )
            return LlmNode(
                id=node_id,
                name=component.name,
                # Let `pyagentspec` infer inputs from the prompt template placeholders.
                # This avoids strict validation mismatches when callers provide only titles.
                inputs=None,
                outputs=outputs if outputs else None,
                prompt_template=component.config.get("prompt_template", ""),
                llm_config=llm_config,
            )
        elif node_class == ToolNode:
            # Get or create default tool
            tool = component.config.get("tool")
            if tool is None:
                tool = ServerTool(
                    id=generate_id("tool"),
                    name=component.name,
                    description=f"Tool: {component.name}",
                )
            return ToolNode(
                id=node_id,
                name=component.name,
                inputs=inputs if inputs else None,
                outputs=outputs if outputs else None,
                tool=tool,
            )
        else:
            # Default to a generic node representation (LlmNode)
            default_llm = VllmConfig(
                id=generate_id("llm"),
                name="default_llm",
                model_id="gpt-4",
                url="https://api.openai.com/v1",
            )
            return LlmNode(
                id=node_id,
                name=component.name,
                inputs=inputs if inputs else None,
                outputs=outputs if outputs else None,
                prompt_template=component.config.get("prompt_template", ""),
                llm_config=default_llm,
            )

    def can_convert(self, component: Any) -> bool:
        """Check if this converter can handle the given component."""
        if isinstance(component, Node):
            return True
        if isinstance(component, WorkflowTaskDefinition):
            return True
        if isinstance(component, dict):
            comp_type = component.get("component_type", "")
            return comp_type in (
                "StartNode",
                "EndNode",
                "LlmNode",
                "ToolNode",
                "AgentNode",
                "FlowNode",
                "MapNode",
            )
        return False

    def from_dict(self, node_dict: dict[str, Any]) -> WorkflowTaskDefinition:
        """Convert a dictionary representation to WorkflowTaskDefinition."""
        node_type = node_dict.get("component_type", "LlmNode")
        task_type = self._get_task_type(node_type)

        inputs = [p.get("title", "") for p in node_dict.get("inputs", [])]
        outputs = [p.get("title", "") for p in node_dict.get("outputs", [])]

        config: dict[str, Any] = {}
        if "prompt_template" in node_dict:
            config["prompt_template"] = node_dict["prompt_template"]
        if "llm_config" in node_dict:
            config["llm_config"] = node_dict["llm_config"]
        if "tool" in node_dict:
            config["tool"] = node_dict["tool"]
        if node_type == "FlowNode":
            flow_ref = node_dict.get("flow") or node_dict.get("subflow")
            if isinstance(flow_ref, dict):
                config["flow_id"] = flow_ref.get("$component_ref", "")
                config["flow_name"] = flow_ref.get("name", "")
            elif flow_ref:
                config["flow_id"] = str(flow_ref)
        if node_type == "MapNode":
            if "parallel" in node_dict:
                config["parallel"] = bool(node_dict.get("parallel"))
            flow_ref = (
                node_dict.get("inner_flow") or node_dict.get("subflow") or node_dict.get("flow")
            )
            if isinstance(flow_ref, dict):
                config["inner_flow_id"] = flow_ref.get("$component_ref", "")
            elif flow_ref:
                config["inner_flow_id"] = str(flow_ref)
            if "map_input_key" in node_dict:
                config["map_input_key"] = node_dict.get("map_input_key")
            if "map_item_key" in node_dict:
                config["map_item_key"] = node_dict.get("map_item_key")

        metadata = node_dict.get("metadata") or {}
        self._merge_runtime_metadata(config, metadata)

        return WorkflowTaskDefinition(
            name=node_dict.get("name", ""),
            task_type=task_type,
            config=config,
            inputs=inputs,
            outputs=outputs,
        )

    def to_dict(self, task_def: WorkflowTaskDefinition) -> dict[str, Any]:
        """Convert WorkflowTaskDefinition to a dictionary representation."""
        node_type = self._get_node_type(task_def.task_type)

        result: dict[str, Any] = {
            "component_type": node_type,
            "id": generate_id("node"),
            "name": task_def.name,
            "inputs": [{"title": name, "type": "string"} for name in task_def.inputs],
            "outputs": [{"title": name, "type": "string"} for name in task_def.outputs],
        }

        # Add type-specific fields
        if task_def.task_type == "llm":
            result["prompt_template"] = task_def.config.get("prompt_template", "")
            if "llm_config" in task_def.config:
                result["llm_config"] = task_def.config["llm_config"]
        elif task_def.task_type == "tool":
            if "tool" in task_def.config:
                result["tool"] = task_def.config["tool"]
        elif task_def.task_type == "flow":
            flow_id = task_def.config.get("flow_id") or task_def.config.get("subflow_id")
            if flow_id:
                result["subflow"] = {"$component_ref": flow_id}
        elif task_def.task_type == "map":
            result["parallel"] = bool(task_def.config.get("parallel", True))
            inner_flow_id = task_def.config.get("inner_flow_id") or task_def.config.get(
                "subflow_id"
            )
            if inner_flow_id:
                result["subflow"] = {"$component_ref": inner_flow_id}

        runtime_metadata = {}
        for key in (
            "retry_policy",
            "timeout_seconds",
            "compensation_activity",
            "compensation_input",
            "branch_output_key",
            "map_input_key",
            "map_item_key",
            "on_error",
        ):
            if key in task_def.config:
                runtime_metadata[key] = task_def.config[key]
        if runtime_metadata:
            result["metadata"] = {"dapr": runtime_metadata}

        return result

    def create_workflow_activity(self, task_def: WorkflowTaskDefinition) -> dict[str, Any]:
        """Create a Dapr workflow activity configuration from a task definition.

        Args:
            task_def: The task definition

        Returns:
            Dictionary with activity configuration for Dapr workflow
        """
        activity_config: dict[str, Any] = {
            "name": task_def.name,
            "type": task_def.task_type,
        }

        if task_def.task_type == "llm":
            activity_config["prompt"] = task_def.config.get("prompt_template", "")
            activity_config["llm_config"] = task_def.config.get("llm_config", {})
        elif task_def.task_type == "tool":
            activity_config["tool_name"] = task_def.config.get("tool_name", task_def.name)
        elif task_def.task_type == "agent":
            activity_config["agent_config"] = task_def.config.get("agent_config", {})

        return activity_config

    def _get_task_type(self, node_type: str) -> str:
        """Map OAS node type to Dapr task type."""
        mapping = {
            "StartNode": "start",
            "EndNode": "end",
            "LlmNode": "llm",
            "ToolNode": "tool",
            "AgentNode": "agent",
            "FlowNode": "flow",
            "MapNode": "map",
        }
        return mapping.get(node_type, "llm")

    def _get_node_type(self, task_type: str) -> str:
        """Map Dapr task type to OAS node type."""
        mapping = {
            "start": "StartNode",
            "end": "EndNode",
            "llm": "LlmNode",
            "tool": "ToolNode",
            "agent": "AgentNode",
            "flow": "FlowNode",
            "map": "MapNode",
        }
        return mapping.get(task_type, "LlmNode")

    def _get_node_class(self, task_type: str) -> type[Node]:
        """Get the OAS Node class for a task type."""
        mapping: dict[str, type[Node]] = {
            "start": StartNode,
            "end": EndNode,
            "llm": LlmNode,
            "tool": ToolNode,
            "agent": AgentNode,
            "flow": FlowNode,
            "map": MapNode,
        }
        return mapping.get(task_type, LlmNode)

    def _extract_node_config(self, node: Node) -> dict[str, Any]:
        """Extract configuration from an OAS Node."""
        config: dict[str, Any] = {}

        if isinstance(node, LlmNode):
            config["prompt_template"] = getattr(node, "prompt_template", "")
            llm_config = getattr(node, "llm_config", None)
            if llm_config:
                config["llm_config"] = self._serialize_llm_config(llm_config)

        elif isinstance(node, ToolNode):
            tool = getattr(node, "tool", None)
            if tool:
                config["tool"] = self._serialize_tool(tool)
                config["tool_name"] = getattr(tool, "name", "")

        elif isinstance(node, AgentNode):
            agent = getattr(node, "agent", None)
            if agent:
                config["agent_config"] = self._serialize_agent(agent)

        elif isinstance(node, FlowNode):
            flow = getattr(node, "flow", None) or getattr(node, "subflow", None)
            if flow:
                config["flow_id"] = getattr(flow, "id", "")
                config["flow_name"] = getattr(flow, "name", "")

        elif isinstance(node, MapNode):
            config["parallel"] = getattr(node, "parallel", True)
            inner_flow = (
                getattr(node, "inner_flow", None)
                or getattr(node, "subflow", None)
                or getattr(node, "flow", None)
            )
            if inner_flow:
                config["inner_flow_id"] = getattr(inner_flow, "id", "")

        metadata = getattr(node, "metadata", None) or {}
        self._merge_runtime_metadata(config, metadata)

        return config

    def _extract_input_names(self, node: Node) -> list[str]:
        """Extract input property names from a node."""
        inputs = getattr(node, "inputs", [])
        return [self._extract_name_from_property(p) for p in inputs]

    def _extract_output_names(self, node: Node) -> list[str]:
        """Extract output property names from a node."""
        outputs = getattr(node, "outputs", [])
        return [self._extract_name_from_property(p) for p in outputs]

    def _extract_name_from_property(self, prop: Any) -> str:
        """Extract name from a property (dict, object, or string)."""
        if isinstance(prop, dict):
            title = prop.get("title")
            if title is not None:
                return str(title)
            name = prop.get("name")
            if name is not None:
                return str(name)
            return ""
        if isinstance(prop, str):
            return prop
        # Object with title/name attribute
        title = getattr(prop, "title", None)
        if title is not None:
            return str(title)
        name = getattr(prop, "name", None)
        if name is not None:
            return str(name)
        return str(prop)

    def _serialize_llm_config(self, llm_config: Any) -> dict[str, Any]:
        """Serialize an LLM config to dictionary.

        Uses pyagentspec's serializer to handle Component objects properly.
        """
        return self._serialize_component(llm_config)

    def _serialize_tool(self, tool: Any) -> dict[str, Any]:
        """Serialize a tool to dictionary.

        Uses pyagentspec's serializer to handle Component objects properly.
        """
        return self._serialize_component(tool)

    def _serialize_agent(self, agent: Any) -> dict[str, Any]:
        """Serialize an agent to dictionary.

        Uses pyagentspec's serializer to handle Component objects properly.
        """
        return self._serialize_component(agent)

    def _serialize_component(self, component: Any) -> dict[str, Any]:
        """Serialize a pyagentspec Component to dictionary.

        Uses the proper serialization context required by pyagentspec.
        Falls back to model_dump() or __dict__ to preserve all attributes.
        """
        try:
            from pyagentspec.serialization import AgentSpecSerializer

            serializer = AgentSpecSerializer()
            # Use to_json then parse back to dict to get proper serialization
            import json

            json_str = serializer.to_json(component)
            return json.loads(json_str)
        except Exception:
            # Fallback 1: try model_dump() for Pydantic models
            model_dump = getattr(component, "model_dump", None)
            if callable(model_dump):
                result = model_dump()
                if isinstance(result, dict):
                    return result

            # Fallback 2: try __dict__ for plain objects
            if hasattr(component, "__dict__"):
                return dict(component.__dict__)

            return {}

    def _dicts_to_properties(self, names: list[str]) -> list[Property]:
        """Convert a list of names to Property objects."""
        return [
            Property(
                title=name,
                type="string",
            )
            for name in names
        ]

    @staticmethod
    def _merge_runtime_metadata(config: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Merge runtime hints from metadata into config."""
        if not isinstance(metadata, dict):
            return
        runtime = metadata.get("dapr") or metadata.get("x-dapr") or {}
        if not isinstance(runtime, dict):
            return
        for key in (
            "retry_policy",
            "timeout_seconds",
            "compensation_activity",
            "compensation_input",
            "branch_output_key",
            "map_input_key",
            "map_item_key",
            "flow_name",
            "on_error",
        ):
            if key in runtime and key not in config:
                config[key] = runtime[key]
