# API Reference

Complete API documentation for `dapr-agents-oas-adapter`.

## Core Classes

### Loaders

| Class | Description |
|-------|-------------|
| [`DaprAgentSpecLoader`](loader.md) | Main loader for OAS to Dapr conversion |
| [`AsyncDaprAgentSpecLoader`](async_loader.md) | Async version of the loader |
| [`CachedLoader`](caching.md) | Loader wrapper with caching |
| [`StrictLoader`](validation.md#strictloader) | Loader with schema validation |

### Exporter

| Class | Description |
|-------|-------------|
| [`DaprAgentSpecExporter`](exporter.md) | Export Dapr configs to OAS format |

### Validation

| Class/Function | Description |
|----------------|-------------|
| [`OASSchemaValidator`](validation.md#oasschemavalidator) | Validate OAS input dicts |
| [`WorkflowValidator`](validation.md#workflowvalidator) | Validate workflow structure |
| [`validate_oas_dict`](validation.md#validate_oas_dict) | Convenience validation function |
| [`validate_workflow`](validation.md#validate_workflow) | Convenience workflow validation |

### Caching

| Class | Description |
|-------|-------------|
| [`InMemoryCache`](caching.md#inmemorycache) | In-memory cache backend |
| [`CacheBackend`](caching.md#cachebackend) | Abstract cache interface |
| [`CacheStats`](caching.md#cachestats) | Cache statistics |

### Logging

| Function/Class | Description |
|----------------|-------------|
| [`get_logger`](logging.md#get_logger) | Get a structured logger |
| [`configure_logging`](logging.md#configure_logging) | Configure log level |
| [`log_operation`](logging.md#log_operation) | Decorator for operation logging |
| [`LoggingMixin`](logging.md#loggingmixin) | Mixin for class logging |

### Types

| Class | Description |
|-------|-------------|
| [`DaprAgentConfig`](types.md#dapragentconfig) | Agent configuration model |
| [`WorkflowDefinition`](types.md#workflowdefinition) | Workflow definition model |
| [`WorkflowTaskDefinition`](types.md#workflowtaskdefinition) | Task definition model |
| [`WorkflowEdgeDefinition`](types.md#workflowedgedefinition) | Edge definition model |

### Utilities

| Class/Function | Description |
|----------------|-------------|
| [`IDGenerator`](utils.md#idgenerator) | Deterministic ID generation |
| [`run_sync`](async_loader.md#run_sync) | Run async code synchronously |

## Exceptions

| Exception | Description |
|-----------|-------------|
| `ConversionError` | Raised on conversion failures |
| `OASSchemaValidationError` | Invalid OAS schema |
| `WorkflowValidationError` | Invalid workflow structure |

## Public API

All public exports are available from the main module:

```python
from dapr_agents_oas_adapter import (
    # Loaders
    DaprAgentSpecLoader,
    AsyncDaprAgentSpecLoader,
    CachedLoader,
    StrictLoader,

    # Exporter
    DaprAgentSpecExporter,

    # Validation
    OASSchemaValidator,
    OASSchemaValidationError,
    WorkflowValidator,
    WorkflowValidationError,
    ValidationResult,
    validate_oas_dict,
    validate_workflow,

    # Caching
    CacheBackend,
    CacheStats,
    InMemoryCache,

    # Logging
    LoggingMixin,
    bind_context,
    clear_context,
    configure_logging,
    get_logger,
    log_context,
    log_operation,
    unbind_context,

    # Utilities
    IDGenerator,
    run_sync,
)
```
