# Validation

Classes and functions for validating OAS specifications and workflow structures.

## OASSchemaValidator

Validates OAS input dictionaries before conversion.

::: dapr_agents_oas_adapter.validation.OASSchemaValidator
    options:
      show_root_heading: true
      show_source: true

## StrictLoader

Loader wrapper that validates OAS specs before conversion.

::: dapr_agents_oas_adapter.loader.StrictLoader
    options:
      show_root_heading: true
      show_source: true

## WorkflowValidator

Validates workflow structure after conversion.

::: dapr_agents_oas_adapter.validation.WorkflowValidator
    options:
      show_root_heading: true
      show_source: true

## ValidationResult

Result object returned by validators.

::: dapr_agents_oas_adapter.validation.ValidationResult
    options:
      show_root_heading: true
      show_source: true

## validate_oas_dict

Convenience function for OAS validation.

::: dapr_agents_oas_adapter.validation.validate_oas_dict
    options:
      show_root_heading: true
      show_source: true

## validate_workflow

Convenience function for workflow validation.

::: dapr_agents_oas_adapter.validation.validate_workflow
    options:
      show_root_heading: true
      show_source: true

## Exceptions

### OASSchemaValidationError

::: dapr_agents_oas_adapter.validation.OASSchemaValidationError
    options:
      show_root_heading: true
      show_source: true

### WorkflowValidationError

::: dapr_agents_oas_adapter.validation.WorkflowValidationError
    options:
      show_root_heading: true
      show_source: true

## Usage Examples

### OAS Validation

```python
from dapr_agents_oas_adapter import (
    StrictLoader,
    OASSchemaValidator,
    validate_oas_dict
)
from dapr_agents_oas_adapter.validation import OASSchemaValidationError

# Using StrictLoader
loader = StrictLoader()
try:
    config = loader.load_dict(spec)
except OASSchemaValidationError as e:
    print(f"Errors: {e.issues}")

# Using validator directly
validator = OASSchemaValidator()
result = validator.validate(spec)
if not result.is_valid:
    for error in result.errors:
        print(error)

# Using convenience function
result = validate_oas_dict(spec)
```

### Workflow Validation

```python
from dapr_agents_oas_adapter import (
    WorkflowValidator,
    validate_workflow
)
from dapr_agents_oas_adapter.validation import WorkflowValidationError

# Using validator
validator = WorkflowValidator()
result = validator.validate(workflow)

# Using convenience function
result = validate_workflow(workflow)

# Raise on error
try:
    validate_workflow(workflow, raise_on_error=True)
except WorkflowValidationError as e:
    print(f"Validation failed: {e}")
```
