"""Exporter for converting Dapr Agents components to OAS specifications."""

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pyagentspec import Component
from pyagentspec.serialization import AgentSpecSerializer

from dapr_agents_oas_adapter.converters.agent import AgentConverter
from dapr_agents_oas_adapter.converters.base import ConversionError
from dapr_agents_oas_adapter.converters.flow import FlowConverter
from dapr_agents_oas_adapter.converters.tool import ToolConverter
from dapr_agents_oas_adapter.logging import get_logger, log_operation
from dapr_agents_oas_adapter.types import (
    DaprAgentConfig,
    WorkflowDefinition,
)


class DaprAgentSpecExporter:
    """Exporter for converting Dapr Agents components to OAS specifications.

    This class provides methods to export Dapr Agents and workflows
    to Open Agent Spec (OAS) format in JSON or YAML.

    Example:
        ```python
        exporter = DaprAgentSpecExporter()

        # Export agent configuration to JSON
        json_spec = exporter.to_json(agent_config)

        # Export workflow to YAML file
        exporter.to_yaml_file(workflow_def, "workflow.yaml")

        # Export Dapr Agent instance
        oas_spec = exporter.from_dapr_agent(dapr_agent)
        ```
    """

    AGENTSPEC_VERSION = "25.4.1"

    def __init__(self) -> None:
        """Initialize the exporter."""
        self._serializer = AgentSpecSerializer()
        self._agent_converter = AgentConverter()
        self._flow_converter = FlowConverter()
        self._tool_converter = ToolConverter()
        self._logger = get_logger("DaprAgentSpecExporter")

    def to_json(
        self,
        component: DaprAgentConfig | WorkflowDefinition,
        indent: int = 2,
    ) -> str:
        """Export a Dapr component to OAS JSON format.

        Args:
            component: The Dapr component to export
            indent: JSON indentation level

        Returns:
            JSON string containing the OAS specification

        Raises:
            ConversionError: If the component cannot be exported
        """
        oas_component = self.to_component(component)
        # Use serializer to get base JSON, then re-format with indent
        base_json = self._serializer.to_json(oas_component)
        if indent is not None and indent != 0:
            # Reparse and format with specified indentation
            return json.dumps(json.loads(base_json), indent=indent, ensure_ascii=False)
        return base_json

    def to_yaml(self, component: DaprAgentConfig | WorkflowDefinition) -> str:
        """Export a Dapr component to OAS YAML format.

        Args:
            component: The Dapr component to export

        Returns:
            YAML string containing the OAS specification

        Raises:
            ConversionError: If the component cannot be exported
        """
        oas_component = self.to_component(component)
        return self._serializer.to_yaml(oas_component)

    def to_dict(self, component: Any) -> dict[str, Any]:
        """Export a Dapr component to OAS dictionary format.

        Args:
            component: The Dapr component to export (or object)

        Returns:
            Dictionary containing the OAS specification

        Raises:
            ConversionError: If the component cannot be exported
        """
        component_type = type(component).__name__
        component_name = getattr(component, "name", None)

        with log_operation(
            "export_to_dict",
            self._logger,
            component_type=component_type,
            component_name=component_name,
        ):
            if isinstance(component, DaprAgentConfig):
                result = self._agent_converter.to_dict(component)
                self._logger.debug("agent_exported", agent_name=component.name)
            elif isinstance(component, WorkflowDefinition):
                result = self._flow_converter.to_dict(component)
                self._logger.debug(
                    "workflow_exported",
                    workflow_name=component.name,
                    task_count=len(component.tasks),
                )
            else:
                raise ConversionError(
                    f"Unsupported component type: {component_type}",
                    component,
                    suggestion="Only DaprAgentConfig and WorkflowDefinition are supported",
                )

            result["agentspec_version"] = self.AGENTSPEC_VERSION
            return result

    def to_component(self, component: DaprAgentConfig | WorkflowDefinition) -> Component:
        """Convert a Dapr component to an OAS Component object.

        Args:
            component: The Dapr component to convert

        Returns:
            PyAgentSpec Component object

        Raises:
            ConversionError: If the component type is not supported
        """
        if isinstance(component, DaprAgentConfig):
            return self._agent_converter.to_oas(component)
        elif isinstance(component, WorkflowDefinition):
            return self._flow_converter.to_oas(component)
        else:
            raise ConversionError(
                f"Unsupported component type: {type(component).__name__}",
                component,
                suggestion="Only DaprAgentConfig and WorkflowDefinition are supported",
            )

    def to_json_file(
        self,
        component: DaprAgentConfig | WorkflowDefinition,
        file_path: str | Path,
        indent: int = 2,
    ) -> None:
        """Export a Dapr component to an OAS JSON file.

        Args:
            component: The Dapr component to export
            file_path: Path to the output file
            indent: JSON indentation level

        Raises:
            ConversionError: If the component cannot be exported
        """
        json_content = self.to_json(component, indent)
        Path(file_path).write_text(json_content, encoding="utf-8")

    def to_yaml_file(
        self,
        component: DaprAgentConfig | WorkflowDefinition,
        file_path: str | Path,
    ) -> None:
        """Export a Dapr component to an OAS YAML file.

        Args:
            component: The Dapr component to export
            file_path: Path to the output file

        Raises:
            ConversionError: If the component cannot be exported
        """
        yaml_content = self.to_yaml(component)
        Path(file_path).write_text(yaml_content, encoding="utf-8")

    def from_dapr_agent(self, agent: Any) -> DaprAgentConfig:
        """Extract configuration from a Dapr Agent instance.

        Args:
            agent: A Dapr Agent instance (AssistantAgent, ReActAgent, etc.)

        Returns:
            DaprAgentConfig with the agent's configuration

        Raises:
            ConversionError: If the agent cannot be converted
        """
        agent_name = getattr(agent, "name", "unknown")
        agent_type = type(agent).__name__

        with log_operation(
            "extract_agent_config",
            self._logger,
            agent_name=agent_name,
            agent_type=agent_type,
        ):
            try:
                # Extract basic properties
                name = getattr(agent, "name", "")
                role = getattr(agent, "role", None)
                goal = getattr(agent, "goal", None)
                instructions = getattr(agent, "instructions", [])

                # Extract tools
                tools = getattr(agent, "tools", [])
                tool_names = [
                    getattr(t, "__name__", str(t)) if callable(t) else str(t) for t in tools
                ]

                # Extract tool definitions
                tool_definitions = []
                for tool in tools:
                    if callable(tool):
                        tool_def = self._tool_converter.from_callable(tool)
                        tool_definitions.append(self._tool_converter.to_dict(tool_def))

                # Extract Dapr-specific configuration
                message_bus_name = getattr(agent, "message_bus_name", "messagepubsub")
                state_store_name = getattr(agent, "state_store_name", "statestore")
                agents_registry_store_name = getattr(
                    agent, "agents_registry_store_name", "agentsregistry"
                )
                service_port = getattr(agent, "service_port", 8000)

                # Determine an agent type from the agent class
                extracted_agent_type = type(agent).__name__

                # Build system prompt from instructions
                system_prompt = self._build_system_prompt(role, goal, instructions)

                self._logger.debug(
                    "agent_config_extracted",
                    tool_count=len(tool_names),
                    has_instructions=len(instructions) > 0,
                )

                return DaprAgentConfig(
                    name=name,
                    role=role,
                    goal=goal,
                    instructions=list(instructions) if instructions else [],
                    system_prompt=system_prompt,
                    tools=tool_names,
                    message_bus_name=message_bus_name,
                    state_store_name=state_store_name,
                    agents_registry_store_name=agents_registry_store_name,
                    service_port=service_port,
                    agent_type=extracted_agent_type,
                    tool_definitions=tool_definitions,
                )

            except Exception as e:
                raise ConversionError(
                    "Failed to extract configuration from Dapr agent",
                    agent,
                    suggestion="Ensure the agent was created using dapr-agents library",
                    caused_by=e,
                ) from e

    def from_dapr_workflow(
        self,
        workflow_func: Callable[..., Any],
        task_funcs: list[Callable[..., Any]] | None = None,
    ) -> WorkflowDefinition:
        """Extract definition from a Dapr workflow function.

        Args:
            workflow_func: The workflow function decorated with @workflow
            task_funcs: Optional list of task functions used in the workflow

        Returns:
            WorkflowDefinition with the workflow's structure

        Raises:
            ConversionError: If the workflow cannot be converted
        """
        try:
            from dapr_agents_oas_adapter.types import (
                WorkflowEdgeDefinition,
                WorkflowTaskDefinition,
            )

            # Extract workflow name and description
            name = getattr(workflow_func, "__name__", "workflow")
            description = workflow_func.__doc__ or f"Workflow: {name}"

            # Extract tasks from task functions
            tasks: list[WorkflowTaskDefinition] = []

            # Add a start node
            tasks.append(
                WorkflowTaskDefinition(
                    name="start",
                    task_type="start",
                    inputs=["input"],
                    outputs=["input"],
                )
            )

            # Process task functions
            if task_funcs:
                for i, func in enumerate(task_funcs):
                    task_name = getattr(func, "__name__", f"task_{i}")
                    task_doc = func.__doc__ or ""

                    # Determine a task type from a function
                    task_type = self._infer_task_type(func)

                    tasks.append(
                        WorkflowTaskDefinition(
                            name=task_name,
                            task_type=task_type,
                            config={"description": task_doc},
                            inputs=self._extract_func_inputs(func),
                            outputs=["result"],
                        )
                    )

            # Add end node
            tasks.append(
                WorkflowTaskDefinition(
                    name="end",
                    task_type="end",
                    inputs=["result"],
                    outputs=["result"],
                )
            )

            # Build edges (simple sequential flow)
            edges: list[WorkflowEdgeDefinition] = []
            for i in range(len(tasks) - 1):
                edges.append(
                    WorkflowEdgeDefinition(
                        from_node=tasks[i].name,
                        to_node=tasks[i + 1].name,
                        data_mapping={
                            tasks[i].outputs[0] if tasks[i].outputs else "result": tasks[
                                i + 1
                            ].inputs[0]
                            if tasks[i + 1].inputs
                            else "input"
                        },
                    )
                )

            return WorkflowDefinition(
                name=name,
                description=description.strip(),
                tasks=tasks,
                edges=edges,
                start_node="start",
                end_nodes=["end"],
            )

        except Exception as e:
            raise ConversionError(
                "Failed to extract definition from Dapr workflow",
                workflow_func,
                suggestion="Ensure the workflow was decorated with @workflow from dapr-agents",
                caused_by=e,
            ) from e

    def export_agent_to_json(self, agent: Any) -> str:
        """Convenience method to export a Dapr Agent to JSON.

        Args:
            agent: A Dapr Agent instance

        Returns:
            JSON string with the OAS specification
        """
        config = self.from_dapr_agent(agent)
        return self.to_json(config)

    def export_agent_to_yaml(self, agent: Any) -> str:
        """Convenience method to export a Dapr Agent to YAML.

        Args:
            agent: A Dapr Agent instance

        Returns:
            YAML string with the OAS specification
        """
        config = self.from_dapr_agent(agent)
        return self.to_yaml(config)

    def export_workflow_to_json(
        self,
        workflow_func: Callable[..., Any],
        task_funcs: list[Callable[..., Any]] | None = None,
    ) -> str:
        """Convenience method to export a Dapr workflow to JSON.

        Args:
            workflow_func: The workflow function
            task_funcs: Optional list of task functions

        Returns:
            JSON string with the OAS specification
        """
        workflow_def = self.from_dapr_workflow(workflow_func, task_funcs)
        return self.to_json(workflow_def)

    def export_workflow_to_yaml(
        self,
        workflow_func: Callable[..., Any],
        task_funcs: list[Callable[..., Any]] | None = None,
    ) -> str:
        """Convenience method to export a Dapr workflow to YAML.

        Args:
            workflow_func: The workflow function
            task_funcs: Optional list of task functions

        Returns:
            YAML string with the OAS specification
        """
        workflow_def = self.from_dapr_workflow(workflow_func, task_funcs)
        return self.to_yaml(workflow_def)

    def _build_system_prompt(
        self,
        role: str | None,
        goal: str | None,
        instructions: list[str],
    ) -> str:
        """Build a system prompt from agent properties."""
        parts: list[str] = []

        if role:
            parts.append(f"You are {role}.")

        if goal:
            parts.append(f"Your goal is to {goal}.")

        if instructions:
            parts.append("\nInstructions:")
            for instruction in instructions:
                parts.append(f"- {instruction}")

        return "\n".join(parts)

    def _infer_task_type(self, func: Callable[..., Any]) -> str:
        """Infer the task type from function characteristics."""
        func_name = getattr(func, "__name__", "").lower()
        func_doc = (func.__doc__ or "").lower()

        if "llm" in func_name or "generate" in func_name or "llm" in func_doc:
            return "llm"
        elif "tool" in func_name or "tool" in func_doc:
            return "tool"
        elif "agent" in func_name or "agent" in func_doc:
            return "agent"
        else:
            return "llm"  # Default to LLM task

    def _extract_func_inputs(self, func: Callable[..., Any]) -> list[str]:
        """Extract input parameter names from a function."""
        sig = inspect.signature(func)
        inputs: list[str] = []

        for param_name, _param in sig.parameters.items():
            if param_name not in ("self", "cls", "ctx", "context"):
                inputs.append(param_name)

        return inputs if inputs else ["input"]
