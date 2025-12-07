"""Node converters for OAS <-> Dapr Agents workflow tasks."""

from typing import Any

from pyagentspec import Component

# Import flow components with fallback for different pyagentspec versions
try:
    from pyagentspec.flows import (
        AgentNode,
        EndNode,
        FlowNode,
        LlmNode,
        MapNode,
        Node,
        StartNode,
        ToolNode,
    )
except ImportError:
    # Fallback for older versions - use Component as base
    Node = Component
    StartNode = Component
    EndNode = Component
    LlmNode = Component
    ToolNode = Component
    AgentNode = Component
    FlowNode = Component
    MapNode = Component

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
        node_id = generate_id("node")
        node_class = self._get_node_class(component.task_type)

        # Build inputs/outputs from names
        inputs = [{"title": name, "type": "string"} for name in component.inputs]
        outputs = [{"title": name, "type": "string"} for name in component.outputs]

        if node_class == StartNode:
            return StartNode(
                id=node_id,
                name=component.name,
                inputs=inputs,
                outputs=inputs,  # StartNode passes inputs as outputs
            )
        elif node_class == EndNode:
            return EndNode(
                id=node_id,
                name=component.name,
                inputs=inputs,
                outputs=outputs,
            )
        elif node_class == LlmNode:
            return LlmNode(
                id=node_id,
                name=component.name,
                inputs=inputs,
                outputs=outputs,
                prompt_template=component.config.get("prompt_template", ""),
                llm_config=component.config.get("llm_config"),
            )
        elif node_class == ToolNode:
            return ToolNode(
                id=node_id,
                name=component.name,
                inputs=inputs,
                outputs=outputs,
                tool=component.config.get("tool"),
            )
        else:
            # Default to a generic node representation
            return LlmNode(
                id=node_id,
                name=component.name,
                inputs=inputs,
                outputs=outputs,
                prompt_template=component.config.get("prompt_template", ""),
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
            flow = getattr(node, "flow", None)
            if flow:
                config["flow_id"] = getattr(flow, "id", "")
                config["flow_name"] = getattr(flow, "name", "")

        elif isinstance(node, MapNode):
            config["parallel"] = getattr(node, "parallel", True)
            inner_flow = getattr(node, "inner_flow", None)
            if inner_flow:
                config["inner_flow_id"] = getattr(inner_flow, "id", "")

        return config

    def _extract_input_names(self, node: Node) -> list[str]:
        """Extract input property names from a node."""
        inputs = getattr(node, "inputs", [])
        return [p.get("title", "") if isinstance(p, dict) else str(p) for p in inputs]

    def _extract_output_names(self, node: Node) -> list[str]:
        """Extract output property names from a node."""
        outputs = getattr(node, "outputs", [])
        return [p.get("title", "") if isinstance(p, dict) else str(p) for p in outputs]

    def _serialize_llm_config(self, llm_config: Any) -> dict[str, Any]:
        """Serialize an LLM config to dictionary."""
        if hasattr(llm_config, "model_dump"):
            return llm_config.model_dump()
        elif hasattr(llm_config, "__dict__"):
            return dict(llm_config.__dict__)
        return {}

    def _serialize_tool(self, tool: Any) -> dict[str, Any]:
        """Serialize a tool to dictionary."""
        if hasattr(tool, "model_dump"):
            return tool.model_dump()
        elif hasattr(tool, "__dict__"):
            return dict(tool.__dict__)
        return {}

    def _serialize_agent(self, agent: Any) -> dict[str, Any]:
        """Serialize an agent to dictionary."""
        if hasattr(agent, "model_dump"):
            return agent.model_dump()
        elif hasattr(agent, "__dict__"):
            return dict(agent.__dict__)
        return {}
