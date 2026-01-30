# Validation

The library provides two levels of validation:

1. **OAS Schema Validation** - Validates input before conversion
2. **Workflow Validation** - Validates workflow structure after conversion

## OAS Schema Validation

### StrictLoader

Use `StrictLoader` to validate OAS specs before conversion:

```python
from dapr_agents_oas_adapter import StrictLoader
from dapr_agents_oas_adapter.validation import OASSchemaValidationError

loader = StrictLoader()

try:
    config = loader.load_dict({
        "component_type": "Agent",
        "name": "valid_agent",
        "system_prompt": "Hello"
    })
except OASSchemaValidationError as e:
    print(f"Validation failed: {e}")
    for issue in e.issues:
        print(f"  - {issue}")
```

### Direct Validation

Validate without loading:

```python
from dapr_agents_oas_adapter import OASSchemaValidator, validate_oas_dict

validator = OASSchemaValidator()

# Returns ValidationResult
result = validator.validate({
    "component_type": "Agent",
    "name": "test"
})

if result.is_valid:
    print("Valid!")
else:
    for error in result.errors:
        print(f"Error: {error}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
```

### Convenience Function

```python
from dapr_agents_oas_adapter import validate_oas_dict

result = validate_oas_dict(spec_dict)
print(f"Valid: {result.is_valid}")
```

## Workflow Validation

Validate workflow structure after conversion:

```python
from dapr_agents_oas_adapter import (
    DaprAgentSpecLoader,
    WorkflowValidator,
    validate_workflow
)
from dapr_agents_oas_adapter.types import WorkflowDefinition

loader = DaprAgentSpecLoader()
workflow = loader.load_yaml(workflow_yaml)

if isinstance(workflow, WorkflowDefinition):
    # Using validator class
    validator = WorkflowValidator()
    result = validator.validate(workflow)

    # Or using convenience function
    result = validate_workflow(workflow)

    if not result.is_valid:
        for error in result.errors:
            print(f"Error: {error}")
```

### Validation Checks

The workflow validator checks:

| Check | Description |
|-------|-------------|
| Structure | Valid task types, required fields |
| References | All edge references exist |
| Start/End | Start node defined, end nodes reachable |
| Cycles | No infinite loops in control flow |
| Orphans | No disconnected tasks |

### Raise on Error

```python
from dapr_agents_oas_adapter.validation import WorkflowValidationError

# Raises WorkflowValidationError if invalid
result = validate_workflow(workflow, raise_on_error=True)
```

## ValidationResult

Both validators return a `ValidationResult`:

```python
from dapr_agents_oas_adapter import ValidationResult

result: ValidationResult

# Check validity
if result.is_valid:
    print("No errors")

# Access errors (blocking issues)
for error in result.errors:
    print(f"Error: {error}")

# Access warnings (non-blocking)
for warning in result.warnings:
    print(f"Warning: {warning}")

# Combined
for issue in result.issues:
    print(f"Issue: {issue}")
```

## Custom Validation

Add custom validation logic:

```python
from dapr_agents_oas_adapter.types import WorkflowDefinition

def validate_my_workflow(workflow: WorkflowDefinition) -> list[str]:
    errors = []

    # Custom business rules
    if not workflow.description:
        errors.append("Workflow must have a description")

    if len(workflow.tasks) < 3:
        errors.append("Workflow must have at least 3 tasks")

    return errors
```

## Validation Flow

```
OAS Dict
    │
    ▼
┌─────────────────┐
│ OAS Validation  │ ← StrictLoader / OASSchemaValidator
│ (Pre-conversion)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Conversion    │ ← DaprAgentSpecLoader
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Workflow Validate│ ← WorkflowValidator
│(Post-conversion)│
└────────┬────────┘
         │
         ▼
    Ready to Use
```
