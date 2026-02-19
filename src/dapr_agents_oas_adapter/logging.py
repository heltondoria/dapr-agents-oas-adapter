"""Logging for dapr-agents-oas-adapter.

Provides an injectable stdlib logging pattern. By default, a standard
``logging.Logger`` named ``"dapr_agents_oas_adapter"`` is used. Consumers can
replace it with any stdlib-compatible logger via ``set_logger()``.
"""

from __future__ import annotations

import logging

_default_logger: logging.Logger = logging.getLogger("dapr_agents_oas_adapter")


def get_logger() -> logging.Logger:
    """Get the current logger instance.

    Returns:
        The active ``logging.Logger``.
    """
    return _default_logger


def set_logger(logger: logging.Logger) -> None:
    """Inject a custom logger to replace the default.

    Call this **before** creating ``DaprAgentSpecLoader``,
    ``DaprAgentSpecExporter``, or ``AsyncDaprAgentSpecLoader`` instances,
    because those classes capture the logger at construction time via
    ``get_logger()``. Objects created before ``set_logger()`` is called
    will continue using the previous logger.

    Args:
        logger: A ``logging.Logger`` instance to use for all library logging.
    """
    global _default_logger
    _default_logger = logger
