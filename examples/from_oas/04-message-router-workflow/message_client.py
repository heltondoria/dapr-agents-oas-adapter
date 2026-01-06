from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import sys
from typing import Any

from dapr.clients import DaprClient

PUBSUB_NAME = os.getenv("PUBSUB_NAME", "messagepubsub")
TOPIC_NAME = os.getenv("TOPIC_NAME", "blog.requests")
BLOG_TOPIC = os.getenv("BLOG_TOPIC", "AI Agents")
RAW_DATA = os.getenv("RAW_DATA")
CONTENT_TYPE = os.getenv("CONTENT_TYPE", "application/json")
CE_TYPE = os.getenv("CLOUDEVENT_TYPE")

PUBLISH_ONCE = os.getenv("PUBLISH_ONCE", "true").lower() in {"1", "true", "yes"}
INTERVAL_SEC = float(os.getenv("INTERVAL_SEC", "0"))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "8"))
INITIAL_DELAY = float(os.getenv("INITIAL_DELAY", "0.5"))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", "2.0"))
JITTER_FRAC = float(os.getenv("JITTER_FRAC", "0.2"))
STARTUP_DELAY = float(os.getenv("STARTUP_DELAY", "1.0"))

logger = logging.getLogger("publisher")


async def _backoff_sleep(delay: float, jitter: float, factor: float) -> float:
    actual = max(0.0, delay * (1 + jitter))
    if actual:
        await asyncio.sleep(actual)
    return delay * factor


def _build_payload() -> dict[str, Any]:
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
