"""Tests for dynamic workflow execution behavior."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from dapr_agents_oas_adapter.converters.flow import FlowConverter
from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)


@dataclass
class DummyTask:
    """Lightweight task placeholder for workflow tests."""

    kind: str
    name: str | None = None
    payload: Any | None = None
    retry_policy: Any | None = None
    tasks: list[Any] | None = None


class DummyWorkflowContext:
    """Minimal workflow context stub."""

    def __init__(self, *, force_timeout: bool = False) -> None:
        self.force_timeout = force_timeout
        self.is_replaying = False
        self.calls: list[tuple[str, Any, Any, Any]] = []

    def call_activity(
        self,
        activity: Any,
        input: dict | None = None,  # noqa: A002 - matches Dapr SDK signature
        retry_policy: Any = None,
    ) -> DummyTask:
        self.calls.append(("activity", activity.__name__, input, retry_policy))
        return DummyTask(
            kind="activity",
            name=activity.__name__,
            payload=input,
            retry_policy=retry_policy,
        )

    def call_child_workflow(
        self,
        workflow_name: str,
        input: dict | None = None,  # noqa: A002 - matches Dapr SDK signature
        retry_policy: Any = None,
    ) -> DummyTask:
        self.calls.append(("child", workflow_name, input, retry_policy))
        return DummyTask(
            kind="child",
            name=workflow_name,
            payload=input,
            retry_policy=retry_policy,
        )

    def create_timer(self, delta: Any) -> DummyTask:
        return DummyTask(kind="timer", payload=delta)


def _run_workflow(gen, handler):
    result = None
    while True:
        try:
            yielded = gen.send(result)
        except StopIteration as exc:
            return exc.value
        try:
            result = handler(yielded)
        except Exception as exc:
            try:
                yielded = gen.throw(exc)
            except StopIteration as stop:
                return stop.value
            result = handler(yielded)


@pytest.fixture
def workflow_stubs(monkeypatch):
    try:
        import dapr.ext.workflow as wf
    except Exception as exc:  # pragma: no cover - requires dapr SDK
        pytest.skip(f"Dapr workflow SDK not available: {exc}")

    def when_all(tasks):
        return DummyTask(kind="when_all", tasks=list(tasks))

    def when_any(tasks):
        return DummyTask(kind="when_any", tasks=list(tasks))

    class DummyRetryPolicy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(wf, "when_all", when_all)
    monkeypatch.setattr(wf, "when_any", when_any)
    monkeypatch.setattr(wf, "RetryPolicy", DummyRetryPolicy)
    return wf


def _make_handler(
    ctx: DummyWorkflowContext,
    *,
    activity_results: dict[str, Any] | None = None,
    child_results: dict[str, Any] | None = None,
    fail_on: set[str] | list[str] | None = None,
) -> Callable[[Any], Any]:
    activity_results = activity_results or {}
    child_results = child_results or {}
    fail_on = set(fail_on or [])

    def handle(task: Any) -> Any:
        if not isinstance(task, DummyTask):
            return task
        if task.kind == "activity":
            if task.name in fail_on:
                raise RuntimeError(f"Activity failed: {task.name}")
            return activity_results.get(task.name, {"result": task.payload})
        if task.kind == "child":
            return child_results.get(task.name, {"result": task.payload})
        if task.kind == "when_all":
            return [handle(t) for t in task.tasks or []]
        if task.kind == "when_any":
            if ctx.force_timeout:
                for t in task.tasks or []:
                    if isinstance(t, DummyTask) and t.kind == "timer":
                        return t
            for t in task.tasks or []:
                if isinstance(t, DummyTask) and t.kind != "timer":
                    return t
            return (task.tasks or [task])[0]
        if task.kind == "timer":
            return task
        return task

    return handle


def test_branching_selects_from_branch(workflow_stubs):
    converter = FlowConverter()
    workflow_def = WorkflowDefinition(
        name="branch_flow",
        tasks=[
            WorkflowTaskDefinition(name="start", task_type="start"),
            WorkflowTaskDefinition(name="decide", task_type="llm"),
            WorkflowTaskDefinition(name="end_yes", task_type="end"),
            WorkflowTaskDefinition(name="end_no", task_type="end"),
        ],
        edges=[
            WorkflowEdgeDefinition(from_node="start", to_node="decide"),
            WorkflowEdgeDefinition(from_node="decide", to_node="end_yes", from_branch="yes"),
            WorkflowEdgeDefinition(from_node="decide", to_node="end_no", from_branch="no"),
        ],
        start_node="start",
        end_nodes=["end_yes", "end_no"],
    )

    wf = converter.create_dapr_workflow(workflow_def)
    ctx = DummyWorkflowContext()
    handler = _make_handler(ctx, activity_results={"decide": {"branch": "yes"}})
    result = _run_workflow(wf(ctx, {"input": "value"}), handler)

    assert result.get("branch") == "yes"


def test_flownode_calls_child_workflow(workflow_stubs):
    converter = FlowConverter()
    child = WorkflowDefinition(
        name="child_flow",
        flow_id="flow_child",
        tasks=[WorkflowTaskDefinition(name="start", task_type="start")],
        edges=[],
        start_node="start",
        end_nodes=[],
    )
    workflow_def = WorkflowDefinition(
        name="parent_flow",
        tasks=[
            WorkflowTaskDefinition(name="start", task_type="start"),
            WorkflowTaskDefinition(
                name="child_step", task_type="flow", config={"flow_name": "child_flow"}
            ),
            WorkflowTaskDefinition(name="end", task_type="end"),
        ],
        edges=[
            WorkflowEdgeDefinition(
                from_node="start", to_node="child_step", data_mapping={"input": "input"}
            ),
            WorkflowEdgeDefinition(
                from_node="child_step", to_node="end", data_mapping={"result": "result"}
            ),
        ],
        start_node="start",
        end_nodes=["end"],
        subflows={"flow_child": child},
    )

    wf = converter.create_dapr_workflow(workflow_def)
    ctx = DummyWorkflowContext()
    handler = _make_handler(ctx, child_results={"child_flow": {"result": "ok"}})
    result = _run_workflow(wf(ctx, {"input": "value"}), handler)

    assert result.get("result") == "ok"
    assert ("child", "child_flow", {"input": "value"}, None) in ctx.calls


def test_mapnode_parallel_calls_children(workflow_stubs):
    converter = FlowConverter()
    workflow_def = WorkflowDefinition(
        name="map_flow",
        tasks=[
            WorkflowTaskDefinition(name="start", task_type="start"),
            WorkflowTaskDefinition(
                name="map_step",
                task_type="map",
                config={"inner_flow_id": "child_flow", "parallel": True},
                inputs=["items"],
            ),
            WorkflowTaskDefinition(name="end", task_type="end"),
        ],
        edges=[
            WorkflowEdgeDefinition(
                from_node="start",
                to_node="map_step",
                data_mapping={"items": "items"},
            ),
            WorkflowEdgeDefinition(
                from_node="map_step",
                to_node="end",
                data_mapping={"result": "results"},
            ),
        ],
        start_node="start",
        end_nodes=["end"],
    )

    wf = converter.create_dapr_workflow(workflow_def)
    ctx = DummyWorkflowContext()
    handler = _make_handler(
        ctx,
        child_results={
            "child_flow": {"result": "ok"},
        },
    )
    result = _run_workflow(
        wf(ctx, {"items": [{"item": 1}, {"item": 2}]}),
        handler,
    )

    assert "results" in result
    assert len(result["results"]) == 2


def test_retry_policy_passed_to_activity(workflow_stubs):
    converter = FlowConverter()
    workflow_def = WorkflowDefinition(
        name="retry_flow",
        tasks=[
            WorkflowTaskDefinition(
                name="step",
                task_type="llm",
                config={"retry_policy": {"max_attempts": 2}},
            )
        ],
        edges=[],
        start_node=None,
        end_nodes=[],
    )

    wf = converter.create_dapr_workflow(workflow_def)
    ctx = DummyWorkflowContext()
    handler = _make_handler(ctx, activity_results={"step": {"result": "ok"}})
    _run_workflow(wf(ctx, {}), handler)

    assert ctx.calls
    assert ctx.calls[0][3] is not None


def test_timeout_raises_error(workflow_stubs):
    converter = FlowConverter()
    workflow_def = WorkflowDefinition(
        name="timeout_flow",
        tasks=[
            WorkflowTaskDefinition(
                name="step",
                task_type="llm",
                config={"timeout_seconds": 1},
            )
        ],
        edges=[],
        start_node=None,
        end_nodes=[],
    )

    wf = converter.create_dapr_workflow(workflow_def)
    ctx = DummyWorkflowContext(force_timeout=True)
    handler = _make_handler(ctx, activity_results={"step": {"result": "ok"}})
    with pytest.raises(TimeoutError):
        _run_workflow(wf(ctx, {}), handler)


def test_compensation_runs_on_failure(workflow_stubs):
    converter = FlowConverter()
    workflow_def = WorkflowDefinition(
        name="compensate_flow",
        tasks=[
            WorkflowTaskDefinition(
                name="step_a",
                task_type="llm",
                config={"compensation_activity": "compensate_a"},
            ),
            WorkflowTaskDefinition(name="step_b", task_type="llm"),
        ],
        edges=[
            WorkflowEdgeDefinition(from_node="step_a", to_node="step_b"),
        ],
        start_node=None,
        end_nodes=[],
    )

    wf = converter.create_dapr_workflow(workflow_def)
    ctx = DummyWorkflowContext()
    handler = _make_handler(
        ctx,
        activity_results={"step_a": {"result": "ok"}},
        fail_on={"step_b"},
    )
    with pytest.raises(RuntimeError):
        _run_workflow(wf(ctx, {}), handler)

    assert any(call[1] == "compensate_a" for call in ctx.calls)


def test_join_node_waits_for_all_predecessors(workflow_stubs):
    """Test that join nodes wait for ALL incoming edges before executing."""
    # Workflow: start -> (task_a, task_b) -> join_node -> end
    converter = FlowConverter()
    workflow_def = WorkflowDefinition(
        name="join_flow",
        tasks=[
            WorkflowTaskDefinition(name="start", task_type="start"),
            WorkflowTaskDefinition(name="task_a", task_type="llm"),
            WorkflowTaskDefinition(name="task_b", task_type="llm"),
            WorkflowTaskDefinition(name="join_node", task_type="llm"),
            WorkflowTaskDefinition(name="end", task_type="end"),
        ],
        edges=[
            WorkflowEdgeDefinition(from_node="start", to_node="task_a"),
            WorkflowEdgeDefinition(from_node="start", to_node="task_b"),
            WorkflowEdgeDefinition(
                from_node="task_a",
                to_node="join_node",
                data_mapping={"a_result": "a_input"},
            ),
            WorkflowEdgeDefinition(
                from_node="task_b",
                to_node="join_node",
                data_mapping={"b_result": "b_input"},
            ),
            WorkflowEdgeDefinition(from_node="join_node", to_node="end"),
        ],
        start_node="start",
        end_nodes=["end"],
    )

    wf = converter.create_dapr_workflow(workflow_def)
    ctx = DummyWorkflowContext()
    handler = _make_handler(
        ctx,
        activity_results={
            "task_a": {"a_result": "from_a"},
            "task_b": {"b_result": "from_b"},
            "join_node": {"combined": "joined"},
        },
    )
    _run_workflow(wf(ctx, {"input": "value"}), handler)

    # Verify join_node was called with data from BOTH task_a and task_b
    join_call = next(
        (call for call in ctx.calls if call[1] == "join_node"),
        None,
    )
    assert join_call is not None
    join_input = join_call[2]
    assert "a_input" in join_input, "join_node should receive a_input from task_a"
    assert "b_input" in join_input, "join_node should receive b_input from task_b"
    assert join_input["a_input"] == "from_a"
    assert join_input["b_input"] == "from_b"


def test_diamond_pattern_fork_join(workflow_stubs):
    """Test diamond pattern: start -> (A, B) -> merge -> end."""
    converter = FlowConverter()
    workflow_def = WorkflowDefinition(
        name="diamond_flow",
        tasks=[
            WorkflowTaskDefinition(name="start", task_type="start"),
            WorkflowTaskDefinition(name="branch_a", task_type="llm"),
            WorkflowTaskDefinition(name="branch_b", task_type="llm"),
            WorkflowTaskDefinition(name="merge", task_type="llm"),
            WorkflowTaskDefinition(name="end", task_type="end"),
        ],
        edges=[
            WorkflowEdgeDefinition(from_node="start", to_node="branch_a"),
            WorkflowEdgeDefinition(from_node="start", to_node="branch_b"),
            WorkflowEdgeDefinition(
                from_node="branch_a",
                to_node="merge",
                data_mapping={"result": "a_result"},
            ),
            WorkflowEdgeDefinition(
                from_node="branch_b",
                to_node="merge",
                data_mapping={"result": "b_result"},
            ),
            WorkflowEdgeDefinition(from_node="merge", to_node="end"),
        ],
        start_node="start",
        end_nodes=["end"],
    )

    wf = converter.create_dapr_workflow(workflow_def)
    ctx = DummyWorkflowContext()
    handler = _make_handler(
        ctx,
        activity_results={
            "branch_a": {"result": "A_OUTPUT"},
            "branch_b": {"result": "B_OUTPUT"},
            "merge": {"merged": "success"},
        },
    )
    _run_workflow(wf(ctx, {"input": "start"}), handler)

    # Verify merge received both inputs
    merge_call = next(
        (call for call in ctx.calls if call[1] == "merge"),
        None,
    )
    assert merge_call is not None
    merge_input = merge_call[2]
    assert merge_input.get("a_result") == "A_OUTPUT"
    assert merge_input.get("b_result") == "B_OUTPUT"


def test_linear_workflow_unchanged(workflow_stubs):
    """Ensure linear workflows still work after join fix."""
    converter = FlowConverter()
    workflow_def = WorkflowDefinition(
        name="linear_flow",
        tasks=[
            WorkflowTaskDefinition(name="start", task_type="start"),
            WorkflowTaskDefinition(name="step1", task_type="llm"),
            WorkflowTaskDefinition(name="step2", task_type="llm"),
            WorkflowTaskDefinition(name="step3", task_type="llm"),
            WorkflowTaskDefinition(name="end", task_type="end"),
        ],
        edges=[
            WorkflowEdgeDefinition(
                from_node="start",
                to_node="step1",
                data_mapping={"input": "data"},
            ),
            WorkflowEdgeDefinition(
                from_node="step1",
                to_node="step2",
                data_mapping={"output": "input"},
            ),
            WorkflowEdgeDefinition(
                from_node="step2",
                to_node="step3",
                data_mapping={"output": "input"},
            ),
            WorkflowEdgeDefinition(
                from_node="step3",
                to_node="end",
                data_mapping={"output": "final"},
            ),
        ],
        start_node="start",
        end_nodes=["end"],
    )

    wf = converter.create_dapr_workflow(workflow_def)
    ctx = DummyWorkflowContext()
    handler = _make_handler(
        ctx,
        activity_results={
            "step1": {"output": "step1_done"},
            "step2": {"output": "step2_done"},
            "step3": {"output": "step3_done"},
        },
    )
    result = _run_workflow(wf(ctx, {"input": "initial"}), handler)

    # Verify all steps were called in order
    activity_calls = [call[1] for call in ctx.calls if call[0] == "activity"]
    assert activity_calls == ["step1", "step2", "step3"]
    assert result.get("final") == "step3_done"
