# Utilities

Utility classes and functions.

## IDGenerator

Deterministic ID generation with optional seeding.

::: dapr_agents_oas_adapter.utils.IDGenerator
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - generate
        - reset
        - is_seeded
        - seed
        - get_instance
        - reset_instance

## Usage Examples

### Basic ID Generation

```python
from dapr_agents_oas_adapter import IDGenerator

gen = IDGenerator()

# Generate unique IDs
id1 = gen.generate("agent")  # "agent_a1b2c3d4"
id2 = gen.generate("agent")  # "agent_e5f6g7h8"
id3 = gen.generate("task")   # "task_i9j0k1l2"
```

### Deterministic IDs with Seed

```python
# With seed for reproducible IDs
gen = IDGenerator(seed=42)

id1 = gen.generate("node")
id2 = gen.generate("node")

# Reset to regenerate same sequence
gen.reset(seed=42)
assert gen.generate("node") == id1
assert gen.generate("node") == id2
```

### Singleton Pattern

```python
from dapr_agents_oas_adapter import IDGenerator

# Get global generator (singleton)
gen = IDGenerator.get_instance()
id1 = gen.generate("flow")

# Reset for testing
IDGenerator.reset_instance(seed=123)
```

### Testing with Seeds

```python
import pytest
from dapr_agents_oas_adapter import IDGenerator

@pytest.fixture(autouse=True)
def reset_ids():
    """Reset ID generator before each test."""
    IDGenerator.reset_instance(seed=42)
    yield
```

## Module Functions

### generate_id

Convenience function using global generator.

::: dapr_agents_oas_adapter.utils.generate_id
    options:
      show_root_heading: true
      show_source: true

## Template Utilities

### extract_template_variables

Extract placeholder variables from templates.

::: dapr_agents_oas_adapter.utils.extract_template_variables
    options:
      show_root_heading: true
      show_source: true

### render_template

Render a template with values.

::: dapr_agents_oas_adapter.utils.render_template
    options:
      show_root_heading: true
      show_source: true

## Related

- [Types](types.md) - Data models
