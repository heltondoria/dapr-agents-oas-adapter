from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from asyncio import AbstractEventLoop
from logging import Logger

# Ensure the repo root is on sys.path so `import examples...` works when executing files directly.
from pathlib import Path
from typing import Any

from dapr.clients import DaprClient


def _ensure_repo_root_on_sys_path() -> None:
    """Ensure the repo root (the folder containing `pyproject.toml`) is on sys.path."""
    anchor: Path = Path(__file__).resolve()
    for candidate in [anchor, *anchor.parents]:
        if (candidate / "pyproject.toml").exists():
            candidate_str: str = str(object=candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_ensure_repo_root_on_sys_path()

PUBSUB_NAME: str = os.getenv(key="PUBSUB_NAME", default="messagepubsub")
TOPIC_NAME: str = os.getenv(key="TOPIC_NAME", default="blog.requests")
BLOG_TOPIC: str = os.getenv(key="BLOG_TOPIC", default="AI Agents")
RAW_DATA: str | None = os.getenv(key="RAW_DATA")
CONTENT_TYPE: str = os.getenv(key="CONTENT_TYPE", default="application/json")
CE_TYPE: str | None = os.getenv(key="CLOUDEVENT_TYPE")

PUBLISH_ONCE: bool = os.getenv(key="PUBLISH_ONCE", default="true").lower() in {"1", "true", "yes"}
INTERVAL_SEC = float(os.getenv(key="INTERVAL_SEC", default="0"))
MAX_ATTEMPTS = int(os.getenv(key="MAX_ATTEMPTS", default="8"))
INITIAL_DELAY = float(os.getenv(key="INITIAL_DELAY", default="0.5"))
BACKOFF_FACTOR = float(os.getenv(key="BACKOFF_FACTOR", default="2.0"))
JITTER_FRAC = float(os.getenv(key="JITTER_FRAC", default="0.2"))
STARTUP_DELAY = float(os.getenv(key="STARTUP_DELAY", default="1.0"))

logger: Logger = logging.getLogger("publisher")


async def _backoff_sleep(delay: float, jitter: float, factor: float) -> float:
    actual: float | int = max(0.0, delay * (1 + jitter))
    if actual:
        await asyncio.sleep(delay=actual)
    return delay * factor


def _build_payload() -> dict[str, Any]:
    if RAW_DATA:
        try:
            data: Any = json.loads(s=RAW_DATA)
        except Exception as exc:
            raise ValueError(f"Invalid RAW_DATA JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("RAW_DATA must be a JSON object")
        return data
    return {"topic": BLOG_TOPIC}


def _encode_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(obj=payload, ensure_ascii=False).encode(encoding="utf-8")


async def publish_once(client: DaprClient, payload: dict[str, Any]) -> None:
    initial_delay: float = INITIAL_DELAY
    body: bytes = _encode_payload(payload)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info("publish attempt %d → %s/%s", attempt, PUBSUB_NAME, TOPIC_NAME)
            client.publish_event(
                pubsub_name=PUBSUB_NAME,
                topic_name=TOPIC_NAME,
                data=body,
                data_content_type=CONTENT_TYPE,
                publish_metadata=({"cloudevent.type": CE_TYPE} if CE_TYPE else {}),
            )
            logger.info("published successfully")
            return
        except Exception as exc:
            logger.warning("publish failed: %s", exc)
            if attempt == MAX_ATTEMPTS:
                raise
            logger.info("retrying in ~%.2fs …", initial_delay)
            await _backoff_sleep(delay=initial_delay, jitter=JITTER_FRAC, factor=BACKOFF_FACTOR)


async def main() -> int:
    logging.basicConfig(level=logging.INFO)
    stop_event = asyncio.Event()

    loop: AbstractEventLoop = asyncio.get_running_loop()

    def _stop(*_: object) -> None:
        stop_event.set()

    try:
        loop.add_signal_handler(sig=signal.SIGINT, callback=_stop)
        loop.add_signal_handler(sig=signal.SIGTERM, callback=_stop)
    except NotImplementedError:
        signal.signal(signalnum=signal.SIGINT, handler=lambda *_: _stop())
        signal.signal(signalnum=signal.SIGTERM, handler=lambda *_: _stop())

    if STARTUP_DELAY > 0:
        await asyncio.sleep(delay=STARTUP_DELAY)

    payload: dict[str, Any] = _build_payload()
    logger.info("payload: %s", payload)

    try:
        with DaprClient() as client:
            if PUBLISH_ONCE:
                await publish_once(client=client, payload=payload)
                await asyncio.sleep(delay=0.2)
                return 0

            if INTERVAL_SEC <= 0:
                logger.error(msg="INTERVAL_SEC must be > 0 when PUBLISH_ONCE=false")
                return 2

            logger.info("starting periodic publisher every %.2fs", INTERVAL_SEC)
            while not stop_event.is_set():
                try:
                    await publish_once(client=client, payload=payload)
                except Exception as exc:
                    logger.error("giving up after %d attempts: %s", MAX_ATTEMPTS, exc)

                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(fut=stop_event.wait(), timeout=INTERVAL_SEC)

            logger.info("shutdown requested; exiting")
            return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception("fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
