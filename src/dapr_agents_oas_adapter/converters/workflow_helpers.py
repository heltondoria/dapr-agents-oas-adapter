"""Helper classes for Dapr workflow execution.

This module contains extracted helper classes from FlowConverter.create_dapr_workflow()
to improve code organization, testability, and maintainability.
"""

from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Protocol


class WorkflowContext(Protocol):
    """Protocol for Dapr workflow context."""

    def call_activity(self, activity: Callable[..., Any], **kwargs: Any) -> Any:
        """Call an activity."""
        ...

    def call_child_workflow(self, workflow_name: str, **kwargs: Any) -> Any:
        """Call a child workflow."""
        ...

    def create_timer(self, _duration: timedelta) -> Any:
        """Create a timer that fires after the given duration."""
        ...


@dataclass
class TaskConfig:
    """Configuration for a workflow task."""

    name: str
    task_type: str
    config: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)


class RetryPolicyBuilder:
    """Builds retry policies from task configuration."""

    def __init__(self, wf_module: Any) -> None:
        """Initialize with the Dapr workflow module.

        Args:
            wf_module: The dapr.ext.workflow module
        """
        self._wf = wf_module

    def build(self, config: dict[str, Any]) -> Any | None:
        """Build a retry policy from configuration.

        Args:
            config: Task configuration containing retry_policy

        Returns:
            RetryPolicy instance or None if not configured
        """
        if not hasattr(self._wf, "RetryPolicy"):
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

        return self._wf.RetryPolicy(
            max_number_of_attempts=max_attempts,
            first_retry_interval=timedelta(seconds=initial_backoff),
            max_retry_interval=timedelta(seconds=max_backoff),
            backoff_coefficient=multiplier,
            retry_timeout=timedelta(seconds=retry_timeout) if retry_timeout else None,
        )


class BranchRouter:
    """Routes workflow execution based on branch conditions."""

    DEFAULT_BRANCHES = frozenset({"", "default", "next"})

    @staticmethod
    def is_default_branch(branch: str | None) -> bool:
        """Check if a branch value represents the default path.

        Args:
            branch: Branch value to check

        Returns:
            True if this is a default branch
        """
        if branch is None:
            return True
        return branch.strip().lower() in BranchRouter.DEFAULT_BRANCHES

    @staticmethod
    def extract_branch_value(task_config: dict[str, Any], result: Any) -> str | None:
        """Extract the branch value from a task result.

        Args:
            task_config: Task configuration
            result: Task execution result

        Returns:
            Branch value or None if not found
        """
        # Check configured branch output key
        key = task_config.get("branch_output_key")
        if key and isinstance(result, dict) and key in result:
            value = result.get(key)
            return str(value) if value is not None else None

        # Check common branch keys in result dict
        if isinstance(result, dict):
            for candidate in ("branch", "branch_name", "__branch__"):
                if candidate in result:
                    value = result.get(candidate)
                    return str(value) if value is not None else None

        # String result is the branch value
        if isinstance(result, str):
            return result

        return None

    def select_next_edges(
        self,
        task_name: str,
        task_config: dict[str, Any],
        result: Any,
        outgoing_edges: dict[str, list[Any]],
    ) -> list[Any]:
        """Select the next edges to follow based on task result.

        Args:
            task_name: Name of the current task
            task_config: Task configuration
            result: Task execution result
            outgoing_edges: Map of task names to their outgoing edges

        Returns:
            List of edges to follow
        """
        edges = outgoing_edges.get(task_name, [])
        if not edges:
            return []

        # Check if any edge has a branch condition
        has_branches = any(getattr(edge, "from_branch", None) for edge in edges)
        if not has_branches:
            return edges

        # Try to match a specific branch
        branch_value = self.extract_branch_value(task_config, result)
        if branch_value is not None:
            matching = [e for e in edges if getattr(e, "from_branch", None) == branch_value]
            if matching:
                return matching

        # Fall back to default branches
        default_edges = [
            e for e in edges if self.is_default_branch(getattr(e, "from_branch", None))
        ]
        return default_edges if default_edges else []


class ActivityStubManager:
    """Manages activity stub creation for workflow execution."""

    def __init__(self) -> None:
        """Initialize the stub manager."""
        self._stubs: dict[str, Callable[[Any, Any], Any]] = {}

    def get_stub(self, name: str) -> Callable[[Any, Any], Any]:
        """Get or create an activity stub for the given name.

        Args:
            name: Activity name

        Returns:
            Callable stub for the activity
        """
        if name not in self._stubs:
            self._stubs[name] = self._make_stub(name)
        return self._stubs[name]

    @staticmethod
    def _make_stub(name: str) -> Callable[[Any, Any], Any]:
        """Create a callable placeholder for an activity.

        Args:
            name: Activity name

        Returns:
            Stub function
        """

        def _activity(_ctx: Any, _input: Any = None) -> Any:  # pragma: no cover
            raise RuntimeError("This stub should never be executed directly.")

        _activity.__name__ = name
        _activity.__qualname__ = name
        return _activity


class MapTaskHelper:
    """Helper for map task execution."""

    @staticmethod
    def extract_items(
        task_name: str,
        task_config: dict[str, Any],
        task_inputs: list[str],
        task_input: dict[str, Any],
    ) -> list[Any]:
        """Extract items for map iteration.

        Args:
            task_name: Name of the map task
            task_config: Task configuration
            task_inputs: List of task input names
            task_input: Input data for the task

        Returns:
            List of items to iterate over

        Raises:
            ValueError: If items is not a list
        """
        map_key = task_config.get("map_input_key") or "items"

        if map_key in task_input:
            items = task_input.get(map_key)
        elif len(task_inputs) == 1 and task_inputs[0] in task_input:
            items = task_input.get(task_inputs[0])
        else:
            items = task_input.get("items")

        if not isinstance(items, list):
            raise ValueError(f"MapNode '{task_name}' expects a list for '{map_key}'.")

        return items

    @staticmethod
    def build_item_input(
        task_config: dict[str, Any],
        task_input: dict[str, Any],
        item: Any,
    ) -> dict[str, Any]:
        """Build input for a single map item.

        Args:
            task_config: Task configuration
            task_input: Base input data
            item: Current item

        Returns:
            Input dict for this item
        """
        map_key = task_config.get("map_input_key") or "items"
        item_key = task_config.get("map_item_key") or "item"
        base = {k: v for k, v in task_input.items() if k != map_key}

        if isinstance(item, dict):
            return {**base, **item}
        return {**base, item_key: item}


class FlowNameResolver:
    """Resolves flow/workflow names from task configuration."""

    @staticmethod
    def resolve(task_name: str, task_config: dict[str, Any], key: str) -> str:
        """Resolve a flow name from task configuration.

        Args:
            task_name: Name of the task (fallback)
            task_config: Task configuration
            key: Primary key to look for (e.g., "flow_name", "inner_flow_id")

        Returns:
            Resolved flow name
        """
        config_value = task_config.get(key) or task_config.get("flow_name")
        if config_value:
            return str(config_value)

        config_value = task_config.get("flow_id") or task_config.get("inner_flow_id")
        if config_value:
            return str(config_value)

        return task_name


class CompensationHandler:
    """Handles compensation logic for saga pattern support."""

    COMPENSATION_KEYS = ("compensation_activity", "compensating_activity", "compensation_task")
    ON_ERROR_KEYS = ("compensation_activity", "compensation_task", "activity")

    @staticmethod
    def get_compensation_activity(task_config: dict[str, Any]) -> str | None:
        """Get the compensation activity name for a task.

        Args:
            task_config: Task configuration

        Returns:
            Compensation activity name or None
        """
        for key in CompensationHandler.COMPENSATION_KEYS:
            if key in task_config:
                return str(task_config[key])

        on_error = task_config.get("on_error")
        if isinstance(on_error, dict):
            for key in CompensationHandler.ON_ERROR_KEYS:
                if key in on_error:
                    return str(on_error[key])

        return None

    def execute_compensations(
        self,
        ctx: Any,
        executed_tasks: list[str],
        results: dict[str, Any],
        tasks_by_name: dict[str, Any],
        error: Exception,
        activity_caller: Callable[..., Any],
    ) -> Generator[Any, Any, None]:
        """Execute compensation activities for failed workflow.

        Args:
            ctx: Workflow context
            executed_tasks: List of executed task names (in order)
            results: Task results
            tasks_by_name: Map of task names to definitions
            error: The error that caused compensation
            activity_caller: Function to call activities

        Yields:
            Activity results
        """
        for task_name in reversed(executed_tasks):
            task = tasks_by_name.get(task_name)
            if not task:
                continue

            task_config = getattr(task, "config", {})
            compensation = self.get_compensation_activity(task_config)
            if not compensation:
                continue

            payload: dict[str, Any] = {
                "task": task_name,
                "error": str(error),
                "result": results.get(task_name),
            }
            extra = task_config.get("compensation_input")
            if isinstance(extra, dict):
                payload.update(extra)

            try:
                yield activity_caller(ctx, compensation, payload, None)
            except Exception:  # noqa: S112 - Intentionally swallow exceptions during compensation
                # Compensation must continue even if one step fails
                continue


class TaskExecutor:
    """Executes individual workflow tasks."""

    def __init__(
        self,
        wf_module: Any,
        retry_builder: RetryPolicyBuilder,
        stub_manager: ActivityStubManager,
        task_implementations: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        """Initialize the task executor.

        Args:
            wf_module: The dapr.ext.workflow module
            retry_builder: Retry policy builder
            stub_manager: Activity stub manager
            task_implementations: Optional custom task implementations
        """
        self._wf = wf_module
        self._retry_builder = retry_builder
        self._stub_manager = stub_manager
        self._task_implementations = task_implementations or {}

    def execute(
        self,
        ctx: Any,
        task: Any,
        task_input: dict[str, Any],
    ) -> Generator[Any, Any, Any]:
        """Execute a task and yield its result.

        Args:
            ctx: Workflow context
            task: Task definition
            task_input: Input data for the task

        Yields:
            Intermediate workflow states

        Returns:
            Task result
        """
        task_name = task.name
        task_type = task.task_type
        task_config = task.config
        task_inputs = getattr(task, "inputs", [])

        timeout_seconds = task_config.get("timeout_seconds")
        retry_policy = self._retry_builder.build(task_config)

        # Check for custom implementation - use inner generator to return directly
        if task_name in self._task_implementations:
            impl = self._task_implementations[task_name]
            return impl(**task_input)

        if task_type == "flow":
            return (
                yield from self._execute_flow(
                    ctx, task_name, task_config, task_input, retry_policy, timeout_seconds
                )
            )

        if task_type == "map":
            return (
                yield from self._execute_map(
                    ctx,
                    task_name,
                    task_config,
                    task_inputs,
                    task_input,
                    retry_policy,
                    timeout_seconds,
                )
            )

        # Default: call as activity
        activity = self._stub_manager.get_stub(task_name)
        task_obj = self._call_activity(ctx, activity, task_input, retry_policy)
        return (yield from self._await_with_timeout(ctx, task_obj, timeout_seconds))

    def _execute_flow(
        self,
        ctx: Any,
        task_name: str,
        task_config: dict[str, Any],
        task_input: dict[str, Any],
        retry_policy: Any,
        timeout_seconds: int | None,
    ) -> Generator[Any, Any, Any]:
        """Execute a flow task (child workflow)."""
        workflow_name = FlowNameResolver.resolve(task_name, task_config, "flow_name")

        if hasattr(ctx, "call_child_workflow"):
            task_obj = self._call_child_workflow(ctx, workflow_name, task_input, retry_policy)
            return (yield from self._await_with_timeout(ctx, task_obj, timeout_seconds))

        activity = self._stub_manager.get_stub(task_name)
        task_obj = self._call_activity(ctx, activity, task_input, retry_policy)
        return (yield from self._await_with_timeout(ctx, task_obj, timeout_seconds))

    def _execute_map(
        self,
        ctx: Any,
        task_name: str,
        task_config: dict[str, Any],
        task_inputs: list[str],
        task_input: dict[str, Any],
        retry_policy: Any,
        timeout_seconds: int | None,
    ) -> Generator[Any, Any, Any]:
        """Execute a map task (fan-out)."""
        items = MapTaskHelper.extract_items(task_name, task_config, task_inputs, task_input)
        parallel = bool(task_config.get("parallel", True))
        workflow_name = FlowNameResolver.resolve(task_name, task_config, "inner_flow_id")

        if hasattr(ctx, "call_child_workflow"):
            tasks = [
                self._call_child_workflow(
                    ctx,
                    workflow_name,
                    MapTaskHelper.build_item_input(task_config, task_input, item),
                    retry_policy,
                )
                for item in items
            ]
            if parallel:
                if timeout_seconds:
                    all_task = self._wf.when_all(tasks)
                    return (yield from self._await_with_timeout(ctx, all_task, timeout_seconds))
                return (yield self._wf.when_all(tasks))

            results_list = []
            for t in tasks:
                results_list.append((yield from self._await_with_timeout(ctx, t, timeout_seconds)))
            return results_list

        activity = self._stub_manager.get_stub(task_name)
        task_obj = self._call_activity(ctx, activity, task_input, retry_policy)
        return (yield from self._await_with_timeout(ctx, task_obj, timeout_seconds))

    def _call_activity(
        self,
        ctx: Any,
        activity: Callable[[Any, Any], Any],
        input_data: dict[str, Any],
        retry: Any,
    ) -> Any:
        """Call an activity with optional retry policy."""
        kwargs: dict[str, Any] = {"input": input_data}
        if retry is not None:
            kwargs["retry_policy"] = retry
        try:
            return ctx.call_activity(activity, **kwargs)
        except TypeError:
            kwargs.pop("retry_policy", None)
            return ctx.call_activity(activity, **kwargs)

    def _call_child_workflow(
        self,
        ctx: Any,
        workflow_name: str,
        input_data: dict[str, Any],
        retry: Any,
    ) -> Any:
        """Call a child workflow with optional retry policy."""
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
        self,
        ctx: Any,
        task_obj: Any,
        timeout_seconds: int | None,
    ) -> Generator[Any, Any, Any]:
        """Await a task with optional timeout."""
        if not timeout_seconds or not hasattr(ctx, "create_timer"):
            return (yield task_obj)

        timeout_task = ctx.create_timer(timedelta(seconds=timeout_seconds))
        winner = yield self._wf.when_any([task_obj, timeout_task])
        if winner == timeout_task:
            raise TimeoutError("Task execution timed out.")
        return (yield task_obj)
