from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from pathlib import Path

import dapr.ext.workflow as wf
from dapr.clients import DaprClient
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators.routers import message_router
from dapr_agents.workflow.utils.registration import register_message_routes
from pydantic import BaseModel, Field

from dapr_agents_oas_adapter import DaprAgentSpecLoader
from dapr_agents_oas_adapter.types import WorkflowDefinition
from examples._shared.llm_workflow_activities import build_llm_activities_from_workflow
from examples._shared.optional_dotenv import try_load_dotenv
from examples._shared.paths import find_repo_root

try_load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class StartBlogMessage(BaseModel):
    topic: str = Field(min_length=1, description="Blog topic/title")


def _load_workflow_yaml() -> WorkflowDefinition:
    repo_root = find_repo_root(anchor_file=__file__)
    spec_path = (
        repo_root
        / "examples"
        / "to_oas"
        / "04-message-router-workflow"
        / "exported"
        / "blog_workflow.yaml"
    )
    loader = DaprAgentSpecLoader()
    loaded = loader.load_yaml_file(spec_path)
    if not isinstance(loaded, WorkflowDefinition):
        raise TypeError(f"Expected WorkflowDefinition, got: {type(loaded).__name__}")
    return loaded


async def _wait_for_shutdown() -> None:
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()

    def _set_stop(*_: object) -> None:
        stop.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _set_stop)
        loop.add_signal_handler(signal.SIGTERM, _set_stop)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda *_: _set_stop())
        signal.signal(signal.SIGTERM, lambda *_: _set_stop())

    await stop.wait()


async def main() -> None:
    workflow_def = _load_workflow_yaml()

    runtime = wf.WorkflowRuntime()
    llm = DaprChatClient(component_name="openai")

    for activity in build_llm_activities_from_workflow(
        workflow_def=workflow_def,
        llm=llm,
        runtime=runtime,
    ).values():
        runtime.register_activity(activity)

    loader = DaprAgentSpecLoader()
    inner_workflow = loader.create_workflow(workflow_def)

    @message_router(pubsub="messagepubsub", topic="blog.requests", message_model=StartBlogMessage)
    def blog_workflow(ctx: wf.DaprWorkflowContext, wf_input: dict) -> object:
        """Wrapper com message_router delegando para o workflow criado via OAS."""
        output = yield from inner_workflow(ctx, wf_input)
        if isinstance(output, dict) and "post" in output:
            return str(output["post"])
        return str(output)

    runtime.register_workflow(blog_workflow)
    runtime.start()

    try:
        with DaprClient() as client:
            closers = register_message_routes(targets=[blog_workflow], dapr_client=client)
            try:
                await _wait_for_shutdown()
            finally:
                for close in closers:
                    try:
                        close()
                    except Exception:
                        logger.exception("Error while closing subscription")
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
