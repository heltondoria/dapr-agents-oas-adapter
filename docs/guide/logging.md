# Logging

The library uses an injectable stdlib `logging.Logger`. By default a logger named `"dapr_agents_oas_adapter"` is used. You can replace it with any stdlib-compatible logger via `set_logger()`.

## Default Behaviour

Without any configuration the library logs through the standard `"dapr_agents_oas_adapter"` logger. Configure it the same way you configure any stdlib logger:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

## Getting the Logger

```python
from dapr_agents_oas_adapter import get_logger

logger = get_logger()
logger.info("Processing started")
```

## Injecting a Custom Logger

Call `set_logger()` **before** creating `DaprAgentSpecLoader`, `DaprAgentSpecExporter`, or `AsyncDaprAgentSpecLoader` instances, because those classes capture the logger at construction time.

```python
import logging
from dapr_agents_oas_adapter import set_logger

custom = logging.getLogger("my_app.adapter")
custom.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
custom.addHandler(handler)

set_logger(custom)
```

## Integration with structlog

If your application uses `structlog`, you can pass a structlog-wrapped logger:

```python
import structlog
from dapr_agents_oas_adapter import set_logger

set_logger(structlog.get_logger("dapr_agents_oas_adapter"))
```

## Log Levels

| Level | Use Case |
|-------|----------|
| DEBUG | Detailed conversion steps |
| INFO | Normal operations |
| WARNING | Non-critical issues |
| ERROR | Conversion failures |
