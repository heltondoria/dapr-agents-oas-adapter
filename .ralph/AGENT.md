# Agent Build Instructions

## Project Overview

**dapr-agents-oas-adapter** is a Python library enabling bidirectional conversion between [Open Agent Spec (OAS)](https://oracle.github.io/agent-spec/) and [Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/).

## Project Setup
```bash
# Install all dependencies including dev groups
uv sync --all-groups
```

## Running Tests
```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage (100% required)
uv run pytest --cov=src/dapr_agents_oas_adapter --cov-report=term-missing --cov-fail-under=100

# Run specific test file
uv run pytest tests/test_converters.py

# Run specific test by name
uv run pytest -k "test_branching_selects"
```

## Quality Gates (ALL MUST PASS)

### Linting & Formatting
```bash
# Lint code with ruff
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format code (PEP8)
uv run ruff format .

# Check formatting without changes
uv run ruff format --check .
```

### Type Checking
```bash
# Type check with mypy (strict mode)
uv run mypy src/ --strict

# Type check with pyright (cross-validation)
uv run pyright src/

# Optional: fast local feedback with ty
uv run ty check
```

### Code Quality Analysis
```bash
# Dead code detection
uv run vulture src/ --min-confidence 90

# Duplicate code detection (pylint similarity checker only)
uv run pylint --disable=all --enable=R0801 src/

# Complexity metrics (report)
uv run radon cc src/ -a

# Complexity gate (must be Grade A)
uv run xenon src/ --max-absolute A --max-modules A --max-average A

# Spell check
uv run codespell .
```

### Full Quality Check (CI Pipeline)
```bash
# Run all quality gates in sequence
uv run ruff check . && \
uv run ruff format --check . && \
uv run mypy src/ --strict && \
uv run pyright src/ && \
uv run vulture src/ --min-confidence 90 && \
uv run pylint --disable=all --enable=R0801 src/ && \
uv run xenon src/ --max-absolute A --max-modules A --max-average A && \
uv run pytest --cov=src/dapr_agents_oas_adapter --cov-fail-under=100
```

## Key Learnings
- FlowConverter.create_dapr_workflow() is complex (~350 lines) - decomposition planned (RF-004)
- Version inconsistency exists between __init__.py and pyproject.toml - fix planned (RF-001)
- Mixed data models (Pydantic + dataclasses) - unification planned (RF-002)
- Template system uses `{{ variable }}` syntax for prompts
- Subflows must be registered BEFORE parent workflows

## Feature Development Quality Standards

**CRITICAL**: All new features MUST meet the following mandatory requirements before being considered complete.

### Testing Requirements

- **Minimum Coverage**: 100% code coverage ratio required for all new code
- **Test Pass Rate**: 100% - all tests must pass, no exceptions
- **Test Types Required**:
  - Unit tests for all converters, loaders, and exporters
  - Integration tests for all conversion paths (OAS→Dapr, Dapr→OAS)
  - Edge case tests for error handling, boundary conditions
  - Roundtrip tests ensuring no data loss in conversions
- **Coverage Validation**: Run coverage reports before marking features complete:
  ```bash
  # Run with coverage report
  uv run pytest --cov=src/dapr_agents_oas_adapter --cov-report=term-missing --cov-fail-under=100

  # Generate HTML coverage report
  uv run pytest --cov=src/dapr_agents_oas_adapter --cov-report=html
  ```
- **Test Quality**: Tests must validate behavior, not just achieve coverage metrics
- **Test Documentation**: Complex test scenarios must include comments explaining the test strategy

### Git Workflow Requirements

Before moving to the next feature, ALL changes must be:

1. **Committed with Clear Messages**:
   ```bash
   git add .
   git commit -m "feat(converters): descriptive message following conventional commits"
   ```
   - Use conventional commit format: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, etc.
   - Include scope when applicable: `feat(flow):`, `fix(loader):`, `test(agent):`
   - Write descriptive messages that explain WHAT changed and WHY

2. **Pushed to Remote Repository**:
   ```bash
   git push origin <branch-name>
   ```
   - Never leave completed features uncommitted
   - Push regularly to maintain backup and enable collaboration
   - Ensure all quality gates pass before considering feature complete

3. **Branch Hygiene**:
   - Work on feature branches, never directly on `main`
   - Branch naming convention: `feature/<feature-name>`, `fix/<issue-name>`, `refactor/<rf-xxx>`
   - Create pull requests for all significant changes

4. **Ralph Integration**:
   - Follow RF-xxx items from .ralph/fix_plan.md in dependency order
   - Mark items complete in .ralph/fix_plan.md upon completion
   - Update .ralph/PROMPT.md if development patterns change
   - Test features work within Ralph's autonomous loop

### Documentation Requirements

**ALL implementation documentation MUST remain synchronized with the codebase**:

1. **Code Documentation**:
   - Language-appropriate documentation (JSDoc, docstrings, etc.)
   - Update inline comments when implementation changes
   - Remove outdated comments immediately

2. **Implementation Documentation**:
   - Update relevant sections in this AGENT.md file
   - Keep build and test commands current
   - Update configuration examples when defaults change
   - Document breaking changes prominently

3. **README Updates**:
   - Keep feature lists current
   - Update setup instructions when dependencies change
   - Maintain accurate command examples
   - Update version compatibility information

4. **AGENT.md Maintenance**:
   - Add new build patterns to relevant sections
   - Update "Key Learnings" with new insights
   - Keep command examples accurate and tested
   - Document new testing patterns or quality gates

### Feature Completion Checklist

Before marking ANY feature as complete, verify:

- [ ] All tests pass: `uv run pytest`
- [ ] Code coverage at 100%: `uv run pytest --cov=src/dapr_agents_oas_adapter --cov-fail-under=100`
- [ ] Ruff lint passes: `uv run ruff check .`
- [ ] Ruff format passes: `uv run ruff format --check .`
- [ ] mypy strict passes: `uv run mypy src/ --strict`
- [ ] pyright passes: `uv run pyright src/`
- [ ] No duplicate code: `uv run pylint --disable=all --enable=R0801 src/`
- [ ] No dead code: `uv run vulture src/ --min-confidence 90`
- [ ] Complexity Grade A: `uv run xenon src/ --max-absolute A --max-modules A --max-average A`
- [ ] All changes committed with conventional commit messages
- [ ] All commits pushed to remote repository
- [ ] .ralph/fix_plan.md task marked as complete
- [ ] Implementation documentation updated (if API changed)
- [ ] .ralph/AGENT.md updated (if new patterns introduced)
- [ ] Breaking changes documented in PRD.md changelog

### Rationale

These standards ensure:
- **Quality**: High test coverage and pass rates prevent regressions
- **Traceability**: Git commits and .ralph/fix_plan.md provide clear history of changes
- **Maintainability**: Current documentation reduces onboarding time and prevents knowledge loss
- **Collaboration**: Pushed changes enable team visibility and code review
- **Reliability**: Consistent quality gates maintain production stability
- **Automation**: Ralph integration ensures continuous development practices

**Enforcement**: AI agents should automatically apply these standards to all feature development tasks without requiring explicit instruction for each task.
