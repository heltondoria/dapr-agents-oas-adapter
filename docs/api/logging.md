# Logging

Structured logging utilities using structlog.

## get_logger

Get a structured logger instance.

::: dapr_agents_oas_adapter.logging.get_logger
    options:
      show_root_heading: true
      show_source: true

## configure_logging

Configure global logging settings.

::: dapr_agents_oas_adapter.logging.configure_logging
    options:
      show_root_heading: true
      show_source: true

## log_operation

Decorator for operation logging.

::: dapr_agents_oas_adapter.logging.log_operation
    options:
      show_root_heading: true
      show_source: true

## log_context

Context manager for scoped logging context.

::: dapr_agents_oas_adapter.logging.log_context
    options:
      show_root_heading: true
      show_source: true

## bind_context

Bind context variables to all subsequent logs.

::: dapr_agents_oas_adapter.logging.bind_context
    options:
      show_root_heading: true
      show_source: true

## unbind_context

Remove specific context variables.

::: dapr_agents_oas_adapter.logging.unbind_context
    options:
      show_root_heading: true
      show_source: true

## clear_context

Clear all bound context variables.

::: dapr_agents_oas_adapter.logging.clear_context
    options:
      show_root_heading: true
      show_source: true

## LoggingMixin

Mixin class for adding logging to classes.

::: dapr_agents_oas_adapter.logging.LoggingMixin
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Basic Logging

```python
from dapr_agents_oas_adapter import get_logger, configure_logging

# Configure
configure_logging(level="INFO")

# Get logger
logger = get_logger("my_module")

# Log with structured data
logger.info("Processing", item_id=123, status="started")
logger.error("Failed", error="timeout", retry_count=3)
```

### Context Binding

```python
from dapr_agents_oas_adapter import bind_context, clear_context

bind_context(request_id="req-123")
logger.info("Processing")  # Includes request_id
clear_context()
```

### Scoped Context

```python
from dapr_agents_oas_adapter import log_context

with log_context(user_id="user-456"):
    logger.info("User action")  # Includes user_id
# Context cleared
```

### Operation Decorator

```python
from dapr_agents_oas_adapter import log_operation

@log_operation("process_data")
def process_data(data: dict) -> dict:
    # Automatically logs start/end with duration
    return transform(data)
```

### Class Mixin

```python
from dapr_agents_oas_adapter import LoggingMixin

class MyService(LoggingMixin):
    def process(self):
        self.log.info("Starting process")
        # ...
        self.log.info("Process complete")
```

## Related

- [Logging Guide](../guide/logging.md) - Usage guide
