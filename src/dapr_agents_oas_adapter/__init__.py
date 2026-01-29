"""Dapr Agents Open Agent Spec Adapter.

This library provides bidirectional conversion between Open Agent Spec (OAS)
configurations and Dapr Agents components, enabling:
- Import OAS specifications to create Dapr Agents and Workflows
- Export Dapr Agents and Workflows to OAS format
"""

from importlib.metadata import version

from dapr_agents_oas_adapter.async_loader import AsyncDaprAgentSpecLoader, run_sync
from dapr_agents_oas_adapter.cache import (
    CacheBackend,
    CachedLoader,
    CacheStats,
    InMemoryCache,
)
from dapr_agents_oas_adapter.exporter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.loader import DaprAgentSpecLoader, StrictLoader
from dapr_agents_oas_adapter.logging import (
    LoggingMixin,
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    log_context,
    log_operation,
    unbind_context,
)
from dapr_agents_oas_adapter.utils import IDGenerator
from dapr_agents_oas_adapter.validation import (
    OASSchemaValidationError,
    OASSchemaValidator,
    ValidationResult,
    WorkflowValidationError,
    WorkflowValidator,
    validate_oas_dict,
    validate_workflow,
)

__version__ = version("dapr-agents-oas-adapter")
__all__ = [
    "AsyncDaprAgentSpecLoader",
    "CacheBackend",
    "CacheStats",
    "CachedLoader",
    "DaprAgentSpecExporter",
    "DaprAgentSpecLoader",
    "IDGenerator",
    "InMemoryCache",
    "LoggingMixin",
    "OASSchemaValidationError",
    "OASSchemaValidator",
    "StrictLoader",
    "ValidationResult",
    "WorkflowValidationError",
    "WorkflowValidator",
    "bind_context",
    "clear_context",
    "configure_logging",
    "get_logger",
    "log_context",
    "log_operation",
    "run_sync",
    "unbind_context",
    "validate_oas_dict",
    "validate_workflow",
]
