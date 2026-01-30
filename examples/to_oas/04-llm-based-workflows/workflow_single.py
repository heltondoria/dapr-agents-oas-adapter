from __future__ import annotations

import sys
import time
from pathlib import Path

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators import llm_activity


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

from examples._shared.optional_dotenv import try_load_dotenv  # noqa: E402

try_load_dotenv()

runtime = wf.WorkflowRuntime()
llm = DaprChatClient(component_name="openai")


@runtime.workflow(name="single_task_workflow")
def single_task_workflow(ctx: DaprWorkflowContext, name: str) -> object:
    """Simple workflow: a single LLM activity."""
    response = yield ctx.call_activity(describe_person, input={"name": name})
    return response


@runtime.activity(name="describe_person")
@llm_activity(prompt="Who was {name}?", llm=llm)
async def describe_person(ctx, name: str) -> str:
    # Implementado pelo decorator.
    raise NotImplementedError


def _run() -> None:
    runtime.start()
    time.sleep(5)

    client = wf.DaprWorkflowClient()
    instance_id = client.schedule_new_workflow(workflow=single_task_workflow, input="Grace Hopper")
    print(f"Workflow started: {instance_id}")

    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        print("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        print(f"Grace Hopper bio:\n{state.serialized_output}")
    else:
        print(f"Workflow ended with status: {state.runtime_status}")
        if state.failure_details:
            fd = state.failure_details
            print("Failure type:", fd.error_type)
            print("Failure message:", fd.message)
            print("Stack trace:\n", fd.stack_trace)
        else:
            print("Custom status:", state.serialized_custom_status)

    runtime.shutdown()


if __name__ == "__main__":
    _run()
