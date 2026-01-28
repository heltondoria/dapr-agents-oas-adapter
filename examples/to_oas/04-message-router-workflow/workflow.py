from __future__ import annotations

import sys
from pathlib import Path

from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents.llm.dapr import DaprChatClient
from dapr_agents.workflow.decorators import llm_activity
from dapr_agents.workflow.decorators.routers import message_router
from pydantic import BaseModel, Field


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

from examples._shared.optional_dotenv import try_load_dotenv

try_load_dotenv()


class StartBlogMessage(BaseModel):
    topic: str = Field(min_length=1, description="Blog topic/title")


llm = DaprChatClient(component_name="openai")


@message_router(pubsub="messagepubsub", topic="blog.requests", message_model=StartBlogMessage)
def blog_workflow(ctx: DaprWorkflowContext, wf_input: dict) -> object:
    """Workflow triggered by Pub/Sub."""
    topic = wf_input["topic"]
    outline = yield ctx.call_activity(create_outline, input={"topic": topic})
    post = yield ctx.call_activity(write_post, input={"outline": outline})
    return post


@llm_activity(prompt="Create a short outline about {topic}. Output 3-5 bullet points.", llm=llm)
async def create_outline(ctx, topic: str) -> str:
    raise NotImplementedError


@llm_activity(prompt="Write a short blog post following this outline:\n{outline}", llm=llm)
async def write_post(ctx, outline: str) -> str:
    raise NotImplementedError
