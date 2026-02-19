# Logging

Injectable stdlib logging for dapr-agents-oas-adapter.

## get_logger

Get the current logger instance.

::: dapr_agents_oas_adapter.logging.get_logger
    options:
      show_root_heading: true
      show_source: true

## set_logger

Inject a custom logger to replace the default.

::: dapr_agents_oas_adapter.logging.set_logger
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### Default Logger

```python
from dapr_agents_oas_adapter import get_logger

logger = get_logger()
logger.info("Processing started")
logger.error("Processing failed")
```

### Custom Logger

```python
import logging
from dapr_agents_oas_adapter import set_logger, get_logger

# Create and configure a custom logger
custom = logging.getLogger("my_app.adapter")
custom.setLevel(logging.DEBUG)

# Inject before creating loader/exporter instances
set_logger(custom)

# All library components now use the custom logger
logger = get_logger()
logger.debug("Using custom logger")
```

### Integrating with structlog

```python
import structlog
from dapr_agents_oas_adapter import set_logger

# structlog wraps stdlib loggers, so this works directly
set_logger(structlog.get_logger("dapr_agents_oas_adapter"))
```

## Related

- [Logging Guide](../guide/logging.md) - Usage guide
