from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from pathlib import Path

import dapr.ext.workflow as wf
from dapr.clients import DaprClient
from dapr_agents.workflow.utils.registration import register_message_routes
from workflow import blog_workflow, create_outline, write_post


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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def _wait_for_shutdown() -> None:
    """Block until Ctrl+C or SIGTERM."""
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
    runtime = wf.WorkflowRuntime()

    runtime.register_workflow(blog_workflow)
    runtime.register_activity(create_outline)
    runtime.register_activity(write_post)
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
