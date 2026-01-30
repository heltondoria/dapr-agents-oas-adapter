"""Tests for workflow helper classes."""

import types
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from dapr_agents_oas_adapter.converters.workflow_helpers import (
    ActivityStubManager,
    BranchRouter,
    CompensationHandler,
    FlowNameResolver,
    MapTaskHelper,
    RetryPolicyBuilder,
    TaskExecutor,
)


class TestRetryPolicyBuilder:
    """Tests for RetryPolicyBuilder class."""

    def test_build_returns_none_when_no_retry_policy_class(self) -> None:
        """Test build returns None when wf module has no RetryPolicy."""
        wf_module = MagicMock(spec=[])  # No RetryPolicy attribute
        builder = RetryPolicyBuilder(wf_module)
        result = builder.build({"retry_policy": {"max_attempts": 3}})
        assert result is None

    def test_build_returns_none_when_no_retry_config(self) -> None:
        """Test build returns None when config has no retry_policy."""
        wf_module = MagicMock()
        wf_module.RetryPolicy = MagicMock()
        builder = RetryPolicyBuilder(wf_module)
        result = builder.build({})
        assert result is None

    def test_build_from_dict_config(self) -> None:
        """Test build from dictionary retry config."""
        wf_module = MagicMock()
        mock_policy = MagicMock()
        wf_module.RetryPolicy = MagicMock(return_value=mock_policy)

        builder = RetryPolicyBuilder(wf_module)
        config = {
            "retry_policy": {
                "max_attempts": 5,
                "initial_backoff_seconds": 2,
                "max_backoff_seconds": 60,
                "backoff_multiplier": 2.0,
                "retry_timeout": 300,
            }
        }
        result = builder.build(config)

        assert result == mock_policy
        wf_module.RetryPolicy.assert_called_once_with(
            max_number_of_attempts=5,
            first_retry_interval=timedelta(seconds=2),
            max_retry_interval=timedelta(seconds=60),
            backoff_coefficient=2.0,
            retry_timeout=timedelta(seconds=300),
        )

    def test_build_from_object_config(self) -> None:
        """Test build from object with attributes."""
        wf_module = MagicMock()
        mock_policy = MagicMock()
        wf_module.RetryPolicy = MagicMock(return_value=mock_policy)

        retry_obj = MagicMock()
        retry_obj.max_attempts = 3
        retry_obj.initial_backoff_seconds = 10
        retry_obj.max_backoff_seconds = 120
        retry_obj.backoff_multiplier = 1.5
        retry_obj.retry_timeout = None

        builder = RetryPolicyBuilder(wf_module)
        result = builder.build({"retry_policy": retry_obj})

        assert result == mock_policy
        wf_module.RetryPolicy.assert_called_once()

    def test_build_with_defaults(self) -> None:
        """Test build uses defaults for missing dict values."""
        wf_module = MagicMock()
        mock_policy = MagicMock()
        wf_module.RetryPolicy = MagicMock(return_value=mock_policy)

        builder = RetryPolicyBuilder(wf_module)
        result = builder.build({"retry_policy": {}})

        assert result == mock_policy
        wf_module.RetryPolicy.assert_called_once_with(
            max_number_of_attempts=1,
            first_retry_interval=timedelta(seconds=5),
            max_retry_interval=timedelta(seconds=30),
            backoff_coefficient=1.5,
            retry_timeout=None,
        )

    def test_build_returns_none_for_invalid_retry_config(self) -> None:
        """Test build returns None for unsupported retry config type."""
        wf_module = MagicMock()
        wf_module.RetryPolicy = MagicMock()
        builder = RetryPolicyBuilder(wf_module)
        result = builder.build({"retry_policy": "invalid"})
        assert result is None


class TestBranchRouter:
    """Tests for BranchRouter class."""

    def test_is_default_branch_none(self) -> None:
        """Test None is a default branch."""
        assert BranchRouter.is_default_branch(None) is True

    def test_is_default_branch_empty_string(self) -> None:
        """Test empty string is a default branch."""
        assert BranchRouter.is_default_branch("") is True

    def test_is_default_branch_default(self) -> None:
        """Test 'default' is a default branch."""
        assert BranchRouter.is_default_branch("default") is True
        assert BranchRouter.is_default_branch("DEFAULT") is True
        assert BranchRouter.is_default_branch("  Default  ") is True

    def test_is_default_branch_next(self) -> None:
        """Test 'next' is a default branch."""
        assert BranchRouter.is_default_branch("next") is True
        assert BranchRouter.is_default_branch("NEXT") is True

    def test_is_default_branch_custom(self) -> None:
        """Test custom branch names are not default."""
        assert BranchRouter.is_default_branch("yes") is False
        assert BranchRouter.is_default_branch("success") is False

    def test_extract_branch_value_from_config_key(self) -> None:
        """Test extracting branch from configured output key."""
        config = {"branch_output_key": "decision"}
        result = {"decision": "approved"}
        assert BranchRouter.extract_branch_value(config, result) == "approved"

    def test_extract_branch_value_from_common_keys(self) -> None:
        """Test extracting branch from common branch keys."""
        assert BranchRouter.extract_branch_value({}, {"branch": "yes"}) == "yes"
        assert BranchRouter.extract_branch_value({}, {"branch_name": "no"}) == "no"
        assert BranchRouter.extract_branch_value({}, {"__branch__": "maybe"}) == "maybe"

    def test_extract_branch_value_from_string_result(self) -> None:
        """Test string result is the branch value."""
        assert BranchRouter.extract_branch_value({}, "success") == "success"

    def test_extract_branch_value_returns_none(self) -> None:
        """Test returns None when no branch found."""
        assert BranchRouter.extract_branch_value({}, {"data": 123}) is None
        assert BranchRouter.extract_branch_value({}, 123) is None
        assert BranchRouter.extract_branch_value({}, [1, 2, 3]) is None

    def test_extract_branch_value_none_in_dict(self) -> None:
        """Test returns None when branch key exists but value is None."""
        config = {"branch_output_key": "decision"}
        result = {"decision": None}
        assert BranchRouter.extract_branch_value(config, result) is None

    def test_select_next_edges_no_edges(self) -> None:
        """Test returns empty list when no outgoing edges."""
        router = BranchRouter()
        result = router.select_next_edges("task1", {}, "result", {})
        assert result == []

    def test_select_next_edges_no_branches(self) -> None:
        """Test returns all edges when no branch conditions."""
        router = BranchRouter()
        edge1 = MagicMock(from_branch=None)
        edge2 = MagicMock(from_branch=None)
        outgoing = {"task1": [edge1, edge2]}

        result = router.select_next_edges("task1", {}, "result", outgoing)
        assert result == [edge1, edge2]

    def test_select_next_edges_matching_branch(self) -> None:
        """Test returns matching branch edge."""
        router = BranchRouter()
        edge_yes = MagicMock(from_branch="yes")
        edge_no = MagicMock(from_branch="no")
        outgoing = {"task1": [edge_yes, edge_no]}

        result = router.select_next_edges("task1", {}, {"branch": "yes"}, outgoing)
        assert result == [edge_yes]

    def test_select_next_edges_default_fallback(self) -> None:
        """Test falls back to default edges when no match."""
        router = BranchRouter()
        edge_yes = MagicMock(from_branch="yes")
        edge_default = MagicMock(from_branch="default")
        outgoing = {"task1": [edge_yes, edge_default]}

        result = router.select_next_edges("task1", {}, {"branch": "unknown"}, outgoing)
        assert result == [edge_default]


class TestActivityStubManager:
    """Tests for ActivityStubManager class."""

    def test_get_stub_creates_new(self) -> None:
        """Test get_stub creates a new stub."""
        manager = ActivityStubManager()
        stub = manager.get_stub("my_activity")

        assert callable(stub)
        assert isinstance(stub, types.FunctionType)
        assert stub.__name__ == "my_activity"
        assert stub.__qualname__ == "my_activity"

    def test_get_stub_returns_cached(self) -> None:
        """Test get_stub returns the same stub on subsequent calls."""
        manager = ActivityStubManager()
        stub1 = manager.get_stub("my_activity")
        stub2 = manager.get_stub("my_activity")

        assert stub1 is stub2

    def test_stub_raises_on_execution(self) -> None:
        """Test stub raises RuntimeError when called."""
        manager = ActivityStubManager()
        stub = manager.get_stub("my_activity")

        with pytest.raises(RuntimeError, match="should never be executed"):
            stub(None, None)


class TestMapTaskHelper:
    """Tests for MapTaskHelper class."""

    def test_extract_items_from_map_key(self) -> None:
        """Test extracting items from configured map key."""
        items = MapTaskHelper.extract_items(
            "map_task",
            {"map_input_key": "data"},
            [],
            {"data": [1, 2, 3]},
        )
        assert items == [1, 2, 3]

    def test_extract_items_from_default_key(self) -> None:
        """Test extracting items from default 'items' key."""
        items = MapTaskHelper.extract_items(
            "map_task",
            {},
            [],
            {"items": [1, 2, 3]},
        )
        assert items == [1, 2, 3]

    def test_extract_items_from_single_input(self) -> None:
        """Test extracting items from single input field."""
        items = MapTaskHelper.extract_items(
            "map_task",
            {},
            ["batch"],
            {"batch": [1, 2, 3]},
        )
        assert items == [1, 2, 3]

    def test_extract_items_raises_for_non_list(self) -> None:
        """Test raises ValueError when items is not a list."""
        with pytest.raises(ValueError, match="expects a list"):
            MapTaskHelper.extract_items(
                "map_task",
                {},
                [],
                {"items": "not a list"},
            )

    def test_build_item_input_dict_item(self) -> None:
        """Test building input for dict item."""
        result = MapTaskHelper.build_item_input(
            {},
            {"items": [1, 2], "context": "ctx"},
            {"id": 1, "name": "test"},
        )
        assert result == {"context": "ctx", "id": 1, "name": "test"}

    def test_build_item_input_scalar_item(self) -> None:
        """Test building input for scalar item."""
        result = MapTaskHelper.build_item_input(
            {"map_item_key": "value"},
            {"items": [1, 2], "context": "ctx"},
            42,
        )
        assert result == {"context": "ctx", "value": 42}


class TestFlowNameResolver:
    """Tests for FlowNameResolver class."""

    def test_resolve_from_primary_key(self) -> None:
        """Test resolving from primary key."""
        name = FlowNameResolver.resolve(
            "fallback",
            {"flow_name": "my_flow"},
            "flow_name",
        )
        assert name == "my_flow"

    def test_resolve_from_flow_id(self) -> None:
        """Test resolving from flow_id."""
        name = FlowNameResolver.resolve(
            "fallback",
            {"flow_id": "flow-123"},
            "flow_name",
        )
        assert name == "flow-123"

    def test_resolve_from_inner_flow_id(self) -> None:
        """Test resolving from inner_flow_id."""
        name = FlowNameResolver.resolve(
            "fallback",
            {"inner_flow_id": "inner-456"},
            "inner_flow_id",
        )
        assert name == "inner-456"

    def test_resolve_fallback_to_task_name(self) -> None:
        """Test falling back to task name."""
        name = FlowNameResolver.resolve(
            "fallback_task",
            {},
            "flow_name",
        )
        assert name == "fallback_task"


class TestCompensationHandler:
    """Tests for CompensationHandler class."""

    def test_get_compensation_activity_direct(self) -> None:
        """Test getting compensation from direct config keys."""
        assert (
            CompensationHandler.get_compensation_activity({"compensation_activity": "rollback"})
            == "rollback"
        )
        assert (
            CompensationHandler.get_compensation_activity({"compensating_activity": "undo"})
            == "undo"
        )
        assert (
            CompensationHandler.get_compensation_activity({"compensation_task": "cleanup"})
            == "cleanup"
        )

    def test_get_compensation_activity_from_on_error(self) -> None:
        """Test getting compensation from on_error config."""
        config = {"on_error": {"compensation_activity": "rollback"}}
        assert CompensationHandler.get_compensation_activity(config) == "rollback"

    def test_get_compensation_activity_returns_none(self) -> None:
        """Test returns None when no compensation configured."""
        assert CompensationHandler.get_compensation_activity({}) is None
        assert CompensationHandler.get_compensation_activity({"on_error": {}}) is None

    def test_execute_compensations_in_reverse_order(self) -> None:
        """Test compensations execute in reverse order."""
        handler = CompensationHandler()
        ctx = MagicMock()

        task1 = MagicMock()
        task1.config = {"compensation_activity": "comp1"}
        task2 = MagicMock()
        task2.config = {"compensation_activity": "comp2"}

        tasks_by_name = {"task1": task1, "task2": task2}
        executed = ["task1", "task2"]
        results: dict[str, Any] = {"task1": "r1", "task2": "r2"}
        error = Exception("test error")

        called_activities: list[str] = []

        def mock_caller(_ctx: Any, name: str, _payload: Any, _retry: Any) -> Any:
            called_activities.append(name)
            return f"compensated_{name}"

        gen = handler.execute_compensations(
            ctx, executed, results, tasks_by_name, error, mock_caller
        )
        list(gen)  # Consume generator

        assert called_activities == ["comp2", "comp1"]  # Reverse order

    def test_execute_compensations_skips_missing_tasks(self) -> None:
        """Test compensations skip tasks not in tasks_by_name."""
        handler = CompensationHandler()
        ctx = MagicMock()

        task1 = MagicMock()
        task1.config = {"compensation_activity": "comp1"}

        tasks_by_name = {"task1": task1}
        executed = ["task1", "missing_task"]
        results: dict[str, Any] = {}
        error = Exception("test error")

        called_activities: list[str] = []

        def mock_caller(_ctx: Any, name: str, _payload: Any, _retry: Any) -> Any:
            called_activities.append(name)

        gen = handler.execute_compensations(
            ctx, executed, results, tasks_by_name, error, mock_caller
        )
        list(gen)

        assert called_activities == ["comp1"]

    def test_execute_compensations_continues_on_error(self) -> None:
        """Test compensations continue even if one fails."""
        handler = CompensationHandler()
        ctx = MagicMock()

        task1 = MagicMock()
        task1.config = {"compensation_activity": "comp1"}
        task2 = MagicMock()
        task2.config = {"compensation_activity": "comp2"}

        tasks_by_name = {"task1": task1, "task2": task2}
        executed = ["task1", "task2"]
        results: dict[str, Any] = {}
        error = Exception("test error")

        call_count = 0

        def mock_caller(_ctx: Any, name: str, _payload: Any, _retry: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if name == "comp2":
                raise RuntimeError("Compensation failed")
            return "ok"

        gen = handler.execute_compensations(
            ctx, executed, results, tasks_by_name, error, mock_caller
        )
        list(gen)

        assert call_count == 2  # Both were attempted


class TestTaskExecutor:
    """Tests for TaskExecutor class."""

    def test_execute_with_custom_implementation(self) -> None:
        """Test execute uses custom implementation when provided."""
        wf_module = MagicMock()
        retry_builder = MagicMock()
        retry_builder.build.return_value = None
        stub_manager = ActivityStubManager()

        def custom_impl(**kwargs: Any) -> dict[str, Any]:
            return {"custom": "result", **kwargs}

        executor = TaskExecutor(wf_module, retry_builder, stub_manager, {"my_task": custom_impl})

        task = MagicMock()
        task.name = "my_task"
        task.task_type = "llm"
        task.config = {}
        task.inputs = []

        ctx = MagicMock()
        gen = executor.execute(ctx, task, {"input": "data"})

        # Generator should return the result
        try:
            next(gen)
            pytest.fail("Generator should have returned directly")
        except StopIteration as e:
            assert e.value == {"custom": "result", "input": "data"}

    def test_execute_activity_task(self) -> None:
        """Test execute calls activity for regular task."""
        wf_module = MagicMock()
        retry_builder = MagicMock()
        retry_builder.build.return_value = None
        stub_manager = ActivityStubManager()

        executor = TaskExecutor(wf_module, retry_builder, stub_manager)

        task = MagicMock()
        task.name = "my_task"
        task.task_type = "llm"
        task.config = {}
        task.inputs = []

        ctx = MagicMock()
        ctx.call_activity.return_value = {"result": "from_activity"}

        gen = executor.execute(ctx, task, {"input": "data"})

        # First yield should be the activity call
        yielded = next(gen)
        assert yielded == {"result": "from_activity"}

    def test_execute_flow_task_with_child_workflow(self) -> None:
        """Test execute calls child workflow for flow task."""
        wf_module = MagicMock()
        retry_builder = MagicMock()
        retry_builder.build.return_value = None
        stub_manager = ActivityStubManager()

        executor = TaskExecutor(wf_module, retry_builder, stub_manager)

        task = MagicMock()
        task.name = "my_flow"
        task.task_type = "flow"
        task.config = {"flow_name": "child_workflow"}
        task.inputs = []

        ctx = MagicMock()
        ctx.call_child_workflow.return_value = {"child": "result"}

        gen = executor.execute(ctx, task, {"input": "data"})

        # First yield should be the child workflow call
        yielded = next(gen)
        assert yielded == {"child": "result"}
        ctx.call_child_workflow.assert_called_once()

    def test_execute_map_task_parallel(self) -> None:
        """Test execute handles map task with parallel execution."""
        wf_module = MagicMock()
        wf_module.when_all.return_value = "all_tasks"
        retry_builder = MagicMock()
        retry_builder.build.return_value = None
        stub_manager = ActivityStubManager()

        executor = TaskExecutor(wf_module, retry_builder, stub_manager)

        task = MagicMock()
        task.name = "my_map"
        task.task_type = "map"
        task.config = {"parallel": True}
        task.inputs = []

        ctx = MagicMock()
        ctx.call_child_workflow.side_effect = [
            {"item1": "result1"},
            {"item2": "result2"},
        ]

        gen = executor.execute(ctx, task, {"items": [1, 2]})

        # Should yield when_all result
        yielded = next(gen)
        assert yielded == "all_tasks"
        wf_module.when_all.assert_called_once()
