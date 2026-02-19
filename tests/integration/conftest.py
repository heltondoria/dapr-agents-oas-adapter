"""Shared fixtures for integration tests.

These fixtures require a running Dapr sidecar. Tests are automatically
skipped when the sidecar is not available.
"""

from __future__ import annotations

import os
from typing import Any

import pytest


def _dapr_sidecar_available() -> bool:
    """Check whether a Dapr sidecar is reachable."""
    try:
        import httpx  # noqa: PLC0415

        port = os.environ.get("DAPR_HTTP_PORT", "3500")
        resp = httpx.get(f"http://localhost:{port}/v1.0/healthz", timeout=2.0)
        return resp.status_code == 204
    except Exception:  # noqa: BLE001
        return False


requires_dapr = pytest.mark.skipif(
    not _dapr_sidecar_available(),
    reason="Dapr sidecar not available (set DAPR_HTTP_PORT or start daprd)",
)


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
        "agents_registry_store_name": os.environ.get(
            "DAPR_REGISTRY_STORE", "agentsregistry"
        ),
    }
