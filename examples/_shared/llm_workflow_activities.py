"""Factory for LLM activities from a `WorkflowDefinition`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from dapr_agents.llm.dapr import DaprChatClient

    from dapr_agents_oas_adapter.types import WorkflowDefinition, WorkflowTaskDefinition


def _render_prompt(prompt_template: str, values: dict[str, Any]) -> str:
    try:
        # OAS templates use `{{var}}`. Convert to `{var}` for Python formatting.
        normalized = prompt_template.replace("{{", "{").replace("}}", "}")
        return normalized.format_map(values)
    except KeyError as exc:
        missing: str = str(exc).strip("'")
        raise KeyError(f"Missing prompt variable: {missing}") from exc


def _task_output_key(task: WorkflowTaskDefinition) -> str:
    return task.outputs[0] if task.outputs else "result"


def build_llm_activities_from_workflow(
    *,
    workflow_def: WorkflowDefinition,
    llm: DaprChatClient,
) -> dict[str, Callable[..., Any]]:
    """Create activity functions (callables) for all tasks where `task_type == 'llm'`.

    Each activity receives a payload (dict) and returns a dict whose key is the first entry
    in `outputs`. This simplifies data-mapping in the workflow runner.
    """
    activities: dict[str, Callable[..., Any]] = {}

    for task in workflow_def.tasks:
        if task.task_type != "llm":
            continue

        prompt_template: str = str(object=task.config.get("prompt_template", "")).strip()
        if not prompt_template:
            raise ValueError(f"LLM task is missing `prompt_template`: {task.name}")

        out_key: str = _task_output_key(task)

        def _make_activity(*, name: str, template: str, output_key: str) -> Callable[..., Any]:
            def _activity(ctx: Any, payload: dict[str, Any] | None = None) -> dict[str, Any]:
                values = payload or {}
                prompt = _render_prompt(template, values)
                response: Any = llm.generate(prompt)
                get_message = getattr(response, "get_message", None)
                message = get_message() if callable(get_message) else None
                content = getattr(message, "content", "") if message is not None else ""
                return {output_key: content}

            # Dapr's WorkflowRuntime uses the underlying Python function name as a registry key.
            # Ensure every generated activity has a stable, unique name to avoid collisions.
            _activity.__name__ = name
            _activity.__qualname__ = name
            return _activity

        activities[task.name] = _make_activity(
            name=task.name,
            template=prompt_template,
            output_key=out_key,
        )

    return activities
