from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys

# Ensure the repo root is on sys.path so `import examples...` works when executing files directly.
from pathlib import Path
from typing import Any

from dapr.clients import DaprClient


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

# ---------------------------
# Config via environment vars
# ---------------------------
PUBSUB_NAME = os.getenv("PUBSUB_NAME", "messagepubsub")
TOPIC_NAME = os.getenv("TOPIC_NAME", "blog.requests")
BLOG_TOPIC = os.getenv("BLOG_TOPIC", "AI Agents")  # used when RAW_DATA is not provided
RAW_DATA = os.getenv("RAW_DATA")  # if set, must be a JSON object (string)
CONTENT_TYPE = os.getenv("CONTENT_TYPE", "application/json")
CE_TYPE = os.getenv("CLOUDEVENT_TYPE")  # optional CloudEvent 'type' metadata

# Publish behavior
PUBLISH_ONCE = os.getenv("PUBLISH_ONCE", "true").lower() in {"1", "true", "yes"}
INTERVAL_SEC = float(os.getenv("INTERVAL_SEC", "0"))  # used when PUBLISH_ONCE=false
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "8"))
INITIAL_DELAY = float(os.getenv("INITIAL_DELAY", "0.5"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "2.0"))
JITTER_FRAC = float(os.getenv("JITTER_FRAC", "0.2"))

# Optional warmup (give sidecar/broker a moment)
STARTUP_DELAY = float(os.getenv("STARTUP_DELAY", "1.0"))

logger = logging.getLogger("publisher")


async def _backoff_sleep(delay: float, jitter: float, factor: float) -> float:
    """Sleep for ~delay seconds with ±jitter% randomness, then return the next delay."""
    # Note: we avoid randomness here (Bandit/S311); the example does not require it.
    actual = max(0.0, delay * (1 + jitter))
    if actual:
        await asyncio.sleep(actual)
    return delay * factor


def _build_payload() -> dict[str, Any]:
    """Build JSON payload for the topic."""
    if RAW_DATA:
        try:
            data = json.loads(RAW_DATA)
        except Exception as exc:
            raise ValueError(f"Invalid RAW_DATA JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("RAW_DATA must be a JSON object")
        return data

    return {"topic": BLOG_TOPIC}


def _encode_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


async def publish_once(client: DaprClient, payload: dict[str, Any]) -> None:
    """Publish once with retries and exponential backoff."""
    delay = INITIAL_DELAY
    body = _encode_payload(payload)

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
            logger.info("retrying in ~%.2fs …", delay)
            delay = await _backoff_sleep(delay, JITTER_FRAC, BACKOFF_FACTOR)


async def main() -> int:
    logging.basicConfig(level=logging.INFO)
    stop_event = asyncio.Event()

    # Signal-aware shutdown
    loop = asyncio.get_running_loop()

    def _stop(*_: object) -> None:
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, _stop)
        loop.add_signal_handler(signal.SIGTERM, _stop)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda *_: _stop())
        signal.signal(signal.SIGTERM, lambda *_: _stop())

    if STARTUP_DELAY > 0:
        await asyncio.sleep(STARTUP_DELAY)

    payload = _build_payload()
    logger.info("payload: %s", payload)

    try:
        with DaprClient() as client:
            if PUBLISH_ONCE:
                await publish_once(client, payload)
                await asyncio.sleep(0.2)
                return 0

            if INTERVAL_SEC <= 0:
                logger.error("INTERVAL_SEC must be > 0 when PUBLISH_ONCE=false")
                return 2

            logger.info("starting periodic publisher every %.2fs", INTERVAL_SEC)
            while not stop_event.is_set():
                try:
                    await publish_once(client, payload)
                except Exception as exc:
                    logger.error("giving up after %d attempts: %s", MAX_ATTEMPTS, exc)

                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=INTERVAL_SEC)

            logger.info("shutdown requested; exiting")
            return 0

    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception("fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
