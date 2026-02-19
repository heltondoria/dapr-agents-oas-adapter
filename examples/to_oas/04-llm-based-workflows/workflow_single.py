from __future__ import annotations

import sys
import time
from pathlib import Path

import dapr.ext.workflow as wf
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

from typing import TYPE_CHECKING

from examples._shared.optional_dotenv import try_load_dotenv

if TYPE_CHECKING:
    from dapr.ext.workflow import DaprWorkflowContext

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

    state = client.wait_for_workflow_completion(instance_id)
    if not state or state.runtime_status.name != "COMPLETED" or state.failure_details:
        print(f"Workflow failed: {state}")
    else:
        print(f"Workflow completed: {state.serialized_output}")

    runtime.shutdown()


if __name__ == "__main__":
    _run()
