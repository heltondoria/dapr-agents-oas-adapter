from __future__ import annotations

import sys
import time
from pathlib import Path

import dapr.ext.workflow as wf
from dapr_agents.llm.dapr import DaprChatClient

from dapr_agents_oas_adapter import DaprAgentSpecLoader
from dapr_agents_oas_adapter.types import WorkflowDefinition


def _ensure_repo_root_on_sys_path() -> None:
    """Ensure the repo root (the folder containing `pyproject.toml`) is on sys.path."""
    anchor = Path(__file__).resolve()
    for candidate in [anchor, *anchor.parents]:
        if (candidate / "pyproject.toml").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_ensure_repo_root_on_sys_path()


def _load_workflow_yaml() -> WorkflowDefinition:
    from examples._shared.paths import find_repo_root

    repo_root = find_repo_root(anchor_file=__file__)
    spec_path = (
        repo_root
        / "examples"
        / "to_oas"
        / "04-llm-based-workflows"
        / "exported"
        / "single_task_workflow.yaml"
    )
    loader = DaprAgentSpecLoader()
    loaded = loader.load_yaml_file(spec_path)
    if not isinstance(loaded, WorkflowDefinition):
        raise TypeError(f"Expected WorkflowDefinition, got: {type(loaded).__name__}")
    return loaded


def main() -> int:
    from examples._shared.llm_workflow_activities import build_llm_activities_from_workflow
    from examples._shared.optional_dotenv import try_load_dotenv

    try_load_dotenv()
    workflow_def = _load_workflow_yaml()

    runtime = wf.WorkflowRuntime()
    llm = DaprChatClient()
    llm.component_name = "openai"

    for activity in build_llm_activities_from_workflow(
        workflow_def=workflow_def,
        llm=llm,
    ).values():
        runtime.register_activity(activity)

    loader = DaprAgentSpecLoader()
    workflow_func = loader.create_workflow(workflow_def)
    runtime.register_workflow(workflow_func)

    runtime.start()
    time.sleep(5)

    client = wf.DaprWorkflowClient()
    instance_id = client.schedule_new_workflow(
        workflow=workflow_func,
        input={"name": "Grace Hopper"},
    )
    print(f"Workflow started: {instance_id}")

    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        print("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        print(f"Output:\n{state.serialized_output}")
    else:
        print(f"Workflow ended with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            print("Failure type:", fd.error_type)
            print("Failure message:", fd.message)
            print("Stack trace:\n", fd.stack_trace)

    runtime.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
