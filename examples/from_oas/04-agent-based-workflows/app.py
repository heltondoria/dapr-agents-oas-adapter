from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import dapr.ext.workflow as wf
from dapr.ext.workflow import DaprWorkflowContext
from dapr_agents.workflow.decorators import agent_activity


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

from dapr_agents_oas_adapter import DaprAgentSpecLoader  # noqa: E402
from dapr_agents_oas_adapter.types import DaprAgentConfig  # noqa: E402
from examples._shared.optional_dotenv import try_load_dotenv  # noqa: E402
from examples._shared.paths import find_repo_root  # noqa: E402

try_load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_agent_yaml(name: str) -> DaprAgentConfig:
    repo_root = find_repo_root(anchor_file=__file__)
    spec_path = (
        repo_root / "examples" / "to_oas" / "04-agent-based-workflows" / "exported" / f"{name}.yaml"
    )
    loader = DaprAgentSpecLoader()
    loaded = loader.load_yaml_file(spec_path)
    if not isinstance(loaded, DaprAgentConfig):
        raise TypeError(f"Expected DaprAgentConfig, got: {type(loaded).__name__}")
    return loaded


def main() -> int:
    loader = DaprAgentSpecLoader()

    extractor = loader.create_agent(_load_agent_yaml("DestinationExtractor"))
    planner = loader.create_agent(_load_agent_yaml("PlannerAgent"))
    expander = loader.create_agent(_load_agent_yaml("ItineraryAgent"))

    runtime = wf.WorkflowRuntime()

    @runtime.workflow(name="chained_planner_workflow_from_oas")
    def chained_planner_workflow(ctx: DaprWorkflowContext, user_msg: str) -> object:
        dest = yield ctx.call_activity(extract_destination, input=user_msg)
        outline = yield ctx.call_activity(plan_outline, input=dest["content"])
        itinerary = yield ctx.call_activity(expand_itinerary, input=outline["content"])
        return itinerary["content"]

    @runtime.activity(name="extract_destination")
    @agent_activity(agent=extractor)
    def extract_destination(ctx) -> dict:
        raise NotImplementedError

    @runtime.activity(name="plan_outline")
    @agent_activity(agent=planner)
    def plan_outline(ctx) -> dict:
        raise NotImplementedError

    @runtime.activity(name="expand_itinerary")
    @agent_activity(agent=expander)
    def expand_itinerary(ctx) -> dict:
        raise NotImplementedError

    runtime.start()
    time.sleep(5)

    client = wf.DaprWorkflowClient()
    user_input = "Plan a trip to Paris."
    logger.info("Starting workflow: %s", user_input)
    instance_id = client.schedule_new_workflow(workflow=chained_planner_workflow, input=user_input)
    logger.info("Workflow started: %s", instance_id)

    state = client.wait_for_workflow_completion(instance_id)
    if not state:
        logger.error("No state returned (instance may not exist).")
    elif state.runtime_status.name == "COMPLETED":
        logger.info("Trip Itinerary:\n%s", state.serialized_output)
    else:
        logger.error("Workflow ended with status: %s", state.runtime_status)
        if state.failure_details:
            fd = state.failure_details
            logger.error("Failure type: %s", fd.error_type)
            logger.error("Failure message: %s", fd.message)
            logger.error("Stack trace:\n%s", fd.stack_trace)
        else:
            logger.error("Custom status: %s", state.serialized_custom_status)

    runtime.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
