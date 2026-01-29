# Logging

The library uses `structlog` for structured logging with context propagation.

## Configuration

```python
from dapr_agents_oas_adapter import configure_logging

# Configure logging level
configure_logging(level="INFO")

# Or with debug output
configure_logging(level="DEBUG")
```

## Log Output

Conversions are automatically logged:

```
2024-01-15T10:30:00Z [info     ] load_component_started         component_name=my_agent component_type=Agent
2024-01-15T10:30:00Z [info     ] load_component_completed       component_name=my_agent component_type=Agent duration_ms=5.2
```

## Custom Loggers

Get a logger for your code:

```python
from dapr_agents_oas_adapter import get_logger

logger = get_logger("my_module")

logger.info("Processing started", item_count=10)
logger.error("Processing failed", error="Connection timeout")
```

## Context Binding

Add context to all subsequent log calls:

```python
from dapr_agents_oas_adapter import bind_context, unbind_context, clear_context

# Bind context
bind_context(request_id="req-123", user_id="user-456")

# All logs now include request_id and user_id
logger.info("Processing")  # Includes bound context

# Remove specific context
unbind_context("user_id")

# Clear all context
clear_context()
```

## Context Manager

Use `log_context` for scoped context:

```python
from dapr_agents_oas_adapter import log_context

with log_context(request_id="req-123"):
    # All logs in this block include request_id
    logger.info("Start")
    process_request()
    logger.info("End")
# Context automatically cleared
```

## Operation Logging

The `log_operation` decorator logs function execution:

```python
from dapr_agents_oas_adapter import log_operation

@log_operation("process_workflow")
def process_workflow(workflow_id: str) -> dict:
    # Function execution is automatically logged
    return {"status": "complete"}
```

Output:

```
[info     ] operation_started              operation=process_workflow
[info     ] operation_completed            operation=process_workflow duration_ms=42.5
```

## LoggingMixin

Add logging to your classes:

```python
from dapr_agents_oas_adapter import LoggingMixin

class MyProcessor(LoggingMixin):
    def process(self, data):
        self.log.info("Processing data", size=len(data))
        # Process...
        self.log.info("Processing complete")
```

## Integration with Dapr

Logs integrate with Dapr's observability:

```python
from dapr_agents_oas_adapter import configure_logging

# Configure with JSON output for Dapr
configure_logging(
    level="INFO",
    json_format=True  # Output as JSON for log aggregation
)
```

## Log Levels

| Level | Use Case |
|-------|----------|
| DEBUG | Detailed conversion steps |
| INFO | Normal operations |
| WARNING | Non-critical issues |
| ERROR | Conversion failures |

## Structured Data

All logs support structured data:

```python
logger.info(
    "Workflow loaded",
    workflow_name=workflow.name,
    task_count=len(workflow.tasks),
    has_branches=bool(branch_edges)
)
```
