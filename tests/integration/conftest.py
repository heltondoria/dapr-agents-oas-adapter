"""Shared fixtures for integration tests.

These fixtures require a running Dapr sidecar. Tests are automatically
skipped when the sidecar is not available.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import pytest


def _dapr_sidecar_available(retries: int = 3, interval: float = 2.0) -> bool:
    """Check whether a Dapr sidecar is reachable.

    Retries up to *retries* times with *interval* seconds between attempts
    to account for sidecar startup latency in CI environments.
    """
    import time

    try:
        import httpx
    except ImportError:
        return False

    port = os.environ.get("DAPR_HTTP_PORT", "3500")
    for _ in range(retries):
        try:
            resp = httpx.get(f"http://localhost:{port}/v1.0/healthz", timeout=2.0)
            if resp.is_success:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(interval)
    return False


@functools.cache
def _check_sidecar() -> bool:
    """Lazy singleton check — only probes the sidecar once per session."""
    return _dapr_sidecar_available()


@pytest.fixture(autouse=True, scope="session")
def _require_dapr_sidecar() -> None:
    """Skip all integration tests when the Dapr sidecar is unreachable."""
    if not _check_sidecar():
        pytest.skip("Dapr sidecar not available (set DAPR_HTTP_PORT or start daprd)")


@pytest.fixture(scope="session")
def dapr_http_port() -> str:
    """Return the Dapr HTTP port (defaults to 3500)."""
    return os.environ.get("DAPR_HTTP_PORT", "3500")


@pytest.fixture(scope="session")
def dapr_grpc_port() -> str:
    """Return the Dapr gRPC port (defaults to 50001)."""
    return os.environ.get("DAPR_GRPC_PORT", "50001")


@pytest.fixture(scope="session")
def dapr_test_config() -> dict[str, Any]:
    """Provide common Dapr configuration used across integration tests."""
    return {
        "message_bus_name": os.environ.get("DAPR_PUBSUB_NAME", "messagepubsub"),
        "state_store_name": os.environ.get("DAPR_STATE_STORE", "statestore"),
        "agents_registry_store_name": os.environ.get("DAPR_REGISTRY_STORE", "agentsregistry"),
    }
