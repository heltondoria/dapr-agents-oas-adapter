# PRD: dapr-agents-oas-adapter - Bidirectional OAS/Dapr Agents Interoperability

> **Status:** Planned
> **Priority:** Standalone project (not part of DaprKit dependency tree)
> **Depends on:** pyagentspec >= 25.4.1, dapr-agents >= 0.10.5

---

## 1. Executive Summary

### 1.1 Product Vision

**dapr-agents-oas-adapter** is a Python library that enables bidirectional interoperability between [Open Agent Spec (OAS)](https://oracle.github.io/agent-spec/) specifications and the [Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/) framework, allowing organizations to define AI agents in a standardized declarative format and execute them on distributed Dapr infrastructure.

### 1.2 Problem Statement

There is a significant gap between:

1. **Declarative specifications (OAS)**: Standardized, portable, GitOps-friendly format for defining agents
2. **Executable runtimes (Dapr Agents)**: Python classes with decorators, distributed state, durable workflows

Without this adapter, teams must:
- Manually translate OAS specifications to Dapr code
- Maintain synchronization between OAS documentation and implementation
- Lose portability when committing to a specific runtime

### 1.3 Target Users

| Persona | Description | Primary Need |
|---------|-------------|--------------|
| **Platform Engineer** | Manages AI infrastructure | Deploy YAML-defined agents to Dapr cluster |
| **AI Developer** | Develops agents and workflows | Quickly convert OAS specs to executable code |
| **Solutions Architect** | Defines agent architectures | Export implementations to standardized format |
| **DevOps Engineer** | Automates pipelines | Integrate OAS↔Dapr conversion in CI/CD |

### 1.4 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Conversion Accuracy | 100% | Roundtrip tests with no data loss |
| API Coverage | 95%+ | Supported OAS components / Total OAS |
| Performance | < 100ms | Average conversion time per component |
| Adoption | 1000+ downloads/month | PyPI statistics |
| Reliability | 99.9% | Conversion success rate |

---

## 2. Market Analysis & Strategic Assessment

### 2.1 Landscape of AI Agent Standards (2025)

The AI agent standards ecosystem is rapidly evolving with multiple protocols and specifications competing for adoption:

#### Communication Protocols

| Protocol | Developer | Primary Focus | Enterprise Adoption |
|----------|-----------|---------------|---------------------|
| **MCP** (Model Context Protocol) | Anthropic | Agent-to-Tool | High (Microsoft VS Code, AWS) |
| **A2A** (Agent-to-Agent) | Google | Agent-to-Agent | Growing (50+ partners) |
| **ACP** (Agent Communication Protocol) | IBM/Linux Foundation | RESTful Messaging | Emerging |
| **AGP** (Agent Gateway Protocol) | AGNTCY Collective | gRPC Gateway | Emerging |

#### Declarative Specifications

| Specification | Developer | Primary Focus | Maturity |
|---------------|-----------|---------------|----------|
| **Open Agent Spec (OAS)** | Oracle | Declarative agent definition | Production (Oracle products) |
| **Agent Protocol** | AI-Engineer Foundation | Agent interaction API | Stable |
| **AGENTS.md** | Linux Foundation (AAIF) | Agent documentation | Emerging |

### 2.2 Analysis: Open Agent Spec (OAS)

#### Strengths

1. **Strong Corporate Backing**: Oracle is adopting internally across multiple products:
   - Oracle Applied AI
   - Oracle Financial Services Global Industries
   - Oracle Autonomous Database Select AI

2. **Complete Ecosystem**: Includes SDK (PyAgentSpec), reference runtime (WayFlow), and adapters for popular frameworks (LangGraph, CrewAI, AutoGen)

3. **Academic Validation**: [Technical Report published on arXiv](https://arxiv.org/abs/2510.04173v3) with benchmarks across 4 different runtimes

4. **Integration with Complementary Standards**:
   - Native MCP Tools support
   - AG-UI (Agent-User Interaction Protocol) integration

5. **Portability Focus**: "Write once, run anywhere" - agents can be exported from one framework and executed in another

#### Weaknesses

1. **Limited External Adoption**: Outside Oracle ecosystem, adoption is still nascent

2. **Single Vendor Dependency**: Specification controlled by Oracle, although open-source

3. **Standards Competition**: MCP and A2A have significant momentum from major players (Anthropic, Google, Microsoft)

4. **Immature Adapter Ecosystem**: Adapters for LangGraph/CrewAI still under active development

### 2.3 Analysis: Dapr Agents

#### Strengths

1. **CNCF Graduated Foundation**: Dapr graduated in November 2024, indicating maturity

2. **Durable Workflows**: Durable execution engine guarantees task completion even with failures

3. **Enterprise-Ready Scale**: Users like Derivco and Tempestive execute hundreds of millions of transactions/day

4. **Co-maintained by NVIDIA and Diagrid**: Strong community support

5. **Only CNCF Agent Framework**: Significant competitive differentiator

#### Weaknesses

1. **Relatively New**: Announced in March 2025, still maturing

2. **Infrastructure Complexity**: Requires Dapr sidecar, Kubernetes, state stores

3. **Learning Curve**: Concepts of actors, workflows, sidecars are non-trivial

### 2.4 Competitive Analysis: Alternatives to This Project

| Alternative | Description | Advantages | Disadvantages |
|-------------|-------------|------------|---------------|
| **Use LangGraph Adapter** | Official OAS adapter for LangGraph | Maintained by Oracle, more mature | Loses Dapr capabilities (durability, scale) |
| **Use CrewAI Adapter** | Official OAS adapter for CrewAI | Maintained by Oracle, native multi-agent | No durable workflows, less infra control |
| **A2A + Dapr directly** | Use A2A protocol for integration | More adopted standard, Google/Microsoft support | Doesn't offer definition portability |
| **Manual Implementation** | Create Dapr agents without OAS | Full control | Non-portable, no standardization |
| **Wait for official adapter** | Wait for Oracle to create Dapr adapter | Official support | Undefined timeline, may never happen |

### 2.5 Strategic Assessment: Is OAS the Right Choice?

#### Recommendation: **YES, with active risk monitoring**

**Arguments in favor:**

1. **Complementarity with other standards**: OAS focuses on agent *definition*, while MCP/A2A focus on *communication*. They are complementary, not competing.

2. **Unique niche**: OAS + Dapr Agents = only combination offering "portable definition + distributed durable execution"

3. **Corporate investment**: Oracle is using internally in real enterprise products

4. **Existing ecosystem**: PyAgentSpec, WayFlow runtime, adapters for LangGraph/CrewAI/AutoGen already exist

5. **Clear market gap**: No OAS→Dapr adapter exists. This project fills a specific niche.

**Risks to monitor:**

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| A2A gains dominance | Medium | High | Monitor adoption, prepare A2A adapter as backup |
| Oracle abandons OAS | Low | Critical | Maintain compatibility with stable pyagentspec |
| Official OAS→Dapr adapter emerges | Low | Medium | Differentiate with advanced features |

**Conclusion**: This project is well-positioned in a strategic niche. The OAS + Dapr Agents combination offers a unique **"define once, run distributed"** proposition that doesn't exist in any other tool combination.

### 2.6 Recommended Strategic Pivots

To maximize value and reduce risks:

1. **Add native MCP Tools support**: OAS already supports MCPTool. Ensure the Dapr adapter works seamlessly with MCP servers.

2. **Consider bidirectional A2A adapter**: Future complement to the project - allow Dapr agents to participate in A2A ecosystems.

3. **Contribute upstream to pyagentspec**: Ensure features needed for Dapr are incorporated into the official SDK.

4. **Publish comparative benchmarks**: Demonstrate performance/durability advantages vs LangGraph/CrewAI adapters.

### 2.7 Sources

- [Oracle Blog: Introducing Open Agent Specification](https://blogs.oracle.com/ai-and-datascience/introducing-open-agent-specification)
- [Oracle Blog: AG-UI Integration](https://blogs.oracle.com/ai-and-datascience/announcing-ag-ui-integration-for-agent-spec)
- [ArXiv: Open Agent Spec Technical Report](https://arxiv.org/abs/2510.04173v3)
- [GitHub: Oracle Agent Spec](https://github.com/oracle/agent-spec)
- [CNCF Blog: Announcing Dapr AI Agents](https://www.cncf.io/blog/2025/03/12/announcing-dapr-ai-agents/)
- [Dapr Docs: Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/)
- [InfoQ: Dapr Agents Overview](https://www.infoq.com/news/2025/03/dapr-agents/)
- [Nordic APIs: Comparing AI Agent Standards](https://nordicapis.com/comparing-7-ai-agent-to-api-standards/)
- [Auth0: MCP vs A2A](https://auth0.com/blog/mcp-vs-a2a/)
- [Medium: Open Standards for AI Agents](https://jtanruan.medium.com/open-standards-for-ai-agents-a-technical-comparison-of-a2a-mcp-langchain-agent-protocol-and-482be1101ad9)
- [Agent Protocol](https://agentprotocol.ai/)
- [Linux Foundation: Agentic AI Foundation](https://www.linuxfoundation.org/press/linux-foundation-announces-the-formation-of-the-agentic-ai-foundation)

---

## 3. Business Rules

### 3.1 Conversion Rules

#### BR-001: Agent Type Determination
```
WHEN converting OAS Agent to Dapr Agent:
  IF metadata contains "dapr_agent_type" THEN use specified type
  ELSE IF metadata contains DurableAgent keys (agent_topic, broadcast_topic, etc.) THEN DurableAgent
  ELSE IF agent has tools AND system_prompt contains "reason" or "think" THEN ReActAgent
  ELSE AssistantAgent
```

**Acceptance Criteria:**
- [ ] Agent type is correctly inferred based on configuration
- [ ] Explicit `dapr_agent_type` in metadata overrides inference
- [ ] DurableAgent-specific fields trigger DurableAgent creation

#### BR-002: Workflow Edge Resolution
```
WHEN processing workflow edges:
  IF edge has from_branch value THEN treat as conditional branch
  IF from_branch is null, "default", or "next" THEN treat as default path
  IF task output contains "branch", "branch_name", or branch_output_key THEN use value for routing
  IF no matching branch found THEN follow default edges
```

**Acceptance Criteria:**
- [ ] Branching routes to correct node based on task output
- [ ] Default branches execute when no specific match
- [ ] Missing branch values don't cause runtime errors

#### BR-003: Tool Registry Resolution
```
WHEN resolving tool implementations:
  IF tool name exists in tool_registry THEN use registered implementation
  ELSE IF tool is MCPTool THEN create MCP client with transport config
  ELSE create placeholder that raises error on invocation
```

**Acceptance Criteria:**
- [ ] Registered tools are correctly bound to agent
- [ ] MCP tools connect via configured transport
- [ ] Missing implementations fail fast with clear error message

#### BR-004: LLM Provider Mapping
```
MAPPING:
  VllmConfig    → provider: "vllm"
  OpenAIConfig  → provider: "openai"
  OllamaConfig  → provider: "ollama"
  OciGenAiConfig → provider: "oci"

REVERSE MAPPING applies for export
```

**Acceptance Criteria:**
- [ ] All provider types convert correctly in both directions
- [ ] Unknown providers default to OpenAI-compatible client
- [ ] Configuration parameters (temperature, max_tokens) are preserved

#### BR-005: Subflow Registration Order
```
WHEN workflow contains subflows:
  1. Parse all referenced flows from $referenced_components
  2. Detect circular references and skip already-visited flows
  3. Create child workflow functions recursively
  4. Attach child_workflows list to parent workflow function
  5. Registration order: children BEFORE parent
```

**Acceptance Criteria:**
- [ ] Circular flow references don't cause infinite recursion
- [ ] Child workflows are accessible via `workflow.child_workflows`
- [ ] Parent workflow can call children via `call_child_workflow`

#### BR-006: Data Flow Mapping
```
WHEN passing data between workflow tasks:
  FOR each DataFlowEdge:
    source_output from source_node → destination_input of destination_node
  IF source result is string that looks like JSON THEN attempt parse
  IF source result is scalar THEN wrap as {"result": value}
```

**Acceptance Criteria:**
- [ ] Data mappings correctly route outputs to inputs
- [ ] JSON strings are transparently parsed when valid
- [ ] Type mismatches produce clear error messages

#### BR-007: Retry and Timeout Policies
```
WHEN task config contains retry_policy:
  Create RetryPolicy with:
    - max_number_of_attempts
    - first_retry_interval (from initial_backoff_seconds)
    - max_retry_interval (from max_backoff_seconds)
    - backoff_coefficient (from backoff_multiplier)
    - retry_timeout (optional)

WHEN task config contains timeout_seconds:
  Race task against timer
  IF timer wins THEN raise TimeoutError
```

**Acceptance Criteria:**
- [ ] Retry policy is passed to activity calls
- [ ] Timeouts interrupt long-running tasks
- [ ] Both can be combined on same task

#### BR-008: Compensation Execution
```
WHEN workflow task fails AND task has compensation_activity:
  FOR each executed task in REVERSE order:
    IF task has compensation_activity THEN:
      Call compensation with {task, error, result}
      Continue even if compensation fails (log error)
  THEN re-raise original exception
```

**Acceptance Criteria:**
- [ ] Compensations run in reverse execution order
- [ ] Compensation failures don't mask original error
- [ ] Compensation receives context about failed task

### 3.2 Validation Rules

#### VR-001: OAS Component Validation
```
REQUIRED fields for all OAS components:
  - id: string (non-empty)
  - name: string (non-empty)

ADDITIONAL required fields by type:
  - Agent: llm_config
  - Flow: start_node, nodes[]
  - LlmNode: prompt_template, llm_config
  - ToolNode: tool
```

#### VR-002: Workflow Structure Validation
```
VALID workflow structure:
  - At least one task
  - start_node references existing task
  - All edge from_node/to_node reference existing tasks
  - No orphan tasks (except start/end nodes)
  - End nodes have no outgoing edges
```

#### VR-003: ID Generation
```
Generated IDs MUST:
  - Be unique within conversion session
  - Follow pattern: {type}_{random_suffix}
  - Be deterministic when seed is provided (for testing)
```

### 3.3 Error Handling Rules

#### Exception Hierarchy

All exceptions inherit from a package-level base exception:

```python
class DaprAgentsOasAdapterError(Exception):
    """Base exception for dapr-agents-oas-adapter."""


class ConversionError(DaprAgentsOasAdapterError):
    """Raised when OAS/Dapr conversion fails."""

    def __init__(
        self,
        message: str,
        component: Any = None,
        suggestion: str | None = None,
    ) -> None:
        self.component = component
        self.suggestion = suggestion

        full_message = message
        if component:
            full_message += f"\nComponent: {component}"
        if suggestion:
            full_message += f"\nSuggestion: {suggestion}"
        super().__init__(full_message)


class ValidationError(DaprAgentsOasAdapterError):
    """Raised when component validation fails."""

    def __init__(
        self,
        message: str,
        field_path: str | None = None,
    ) -> None:
        self.field_path = field_path

        full_message = message
        if field_path:
            full_message += f"\nField: {field_path}"
        super().__init__(full_message)
```

#### ER-001: ConversionError
```
RAISE ConversionError when:
  - Unsupported component type encountered
  - Required field missing
  - Circular reference detected (after max depth)
  - Tool implementation not found (on execution)
```

#### ER-002: ValidationError
```
RAISE ValidationError when:
  - Component missing id or name
  - Field type mismatch
  - Invalid reference ($component_ref to non-existent component)
```

#### ER-003: Error Context and Chaining
```
ALL errors MUST:
  - Include descriptive message
  - Include component that caused error (when applicable)
  - Include suggested resolution (when possible)
  - Chain original exception with `from e`
```

**Exception chaining example** — all error handling must preserve the original cause:

```python
try:
    oas_component = deserialize(raw_data)
except KeyError as e:
    raise ConversionError(
        f"Unknown node type: {raw_data.get('type', 'N/A')}",
        component=raw_data,
        suggestion="Supported node types: StartNode, EndNode, LlmNode, ToolNode, AgentNode, FlowNode, MapNode",
    ) from e

try:
    validated = WorkflowValidator().validate(workflow_def)
except ValueError as e:
    raise ValidationError(
        f"Invalid workflow structure: {e}",
        field_path="workflow.nodes",
    ) from e
```

---

## 4. Functional Requirements

### 4.1 Core Features

#### F-001: OAS to Dapr Agent Conversion
**Priority:** P0 (Must Have)

**Description:** Convert OAS Agent specifications to executable Dapr Agent instances.

**User Story:** As a developer, I want to load an OAS Agent YAML file and get a running Dapr agent so that I can execute agent logic without manual translation.

**Technical Requirements:**
- Input: JSON/YAML string, file path, or dict
- Output: `AssistantAgent`, `ReActAgent`, or `DurableAgent` instance
- Support all OAS Agent fields (name, description, system_prompt, tools, llm_config, inputs, outputs, metadata)

**Acceptance Criteria:**
- [ ] `loader.load_yaml_file("agent.yaml")` returns `DaprAgentConfig`
- [ ] `loader.create_agent(config)` returns runnable agent
- [ ] Agent responds to messages using configured LLM
- [ ] Tools are callable from agent context

#### F-002: Dapr Agent to OAS Export
**Priority:** P0 (Must Have)

**Description:** Export existing Dapr Agent instances to OAS specification format.

**User Story:** As an architect, I want to export my Dapr agents to OAS format so that I can document them in a standard way and potentially migrate to other frameworks.

**Technical Requirements:**
- Input: Dapr Agent instance or `DaprAgentConfig`
- Output: JSON/YAML string or file
- Preserve all configuration including tools, LLM config, metadata

**Acceptance Criteria:**
- [ ] `exporter.from_dapr_agent(agent)` extracts configuration
- [ ] `exporter.to_yaml(config)` produces valid OAS YAML
- [ ] Exported spec can be re-imported without loss

#### F-003: OAS Flow to Dapr Workflow Conversion
**Priority:** P0 (Must Have)

**Description:** Convert OAS Flow specifications to executable Dapr workflow functions.

**User Story:** As a developer, I want to define workflows in OAS YAML and execute them as Dapr durable workflows so that I get reliability features (retries, persistence) automatically.

**Technical Requirements:**
- Input: OAS Flow with nodes and edges
- Output: Python generator function compatible with Dapr workflow runtime
- Support node types: StartNode, EndNode, LlmNode, ToolNode, AgentNode, FlowNode, MapNode

**Acceptance Criteria:**
- [ ] `loader.create_workflow(workflow_def)` returns callable
- [ ] Workflow can be registered with `WorkflowRuntime`
- [ ] Tasks execute in correct order based on edges
- [ ] Branching follows `from_branch` values
- [ ] Subflows execute via `call_child_workflow`

#### F-004: Dapr Workflow to OAS Export
**Priority:** P0 (Must Have)

**Description:** Export Dapr workflow functions to OAS Flow specifications.

**User Story:** As a developer, I want to export my Dapr workflows to OAS so that I can share them with teams using other frameworks.

**Technical Requirements:**
- Input: Workflow function decorated with `@workflow`, optional task functions
- Output: `WorkflowDefinition` that can be serialized to OAS

**Limitations (Current):**
- Only sequential flow inference
- Cannot infer branching from code
- Manual edge specification may be needed

**Acceptance Criteria:**
- [ ] `exporter.from_dapr_workflow(func, tasks)` returns `WorkflowDefinition`
- [ ] Definition includes start/end nodes
- [ ] Task types are inferred from function characteristics

#### F-005: Workflow Code Generation
**Priority:** P0 (Must Have)

**Description:** Generate Python boilerplate code for Dapr workflows from OAS specifications.

**User Story:** As a developer, I want to generate starter code from OAS specs so that I can customize implementations without starting from scratch.

**Technical Requirements:**
- Input: `WorkflowDefinition`
- Output: Python code string with `@workflow` and `@task` decorators

**Acceptance Criteria:**
- [ ] Generated code is syntactically valid Python
- [ ] Includes proper imports
- [ ] Task functions have correct signatures
- [ ] TODO comments indicate implementation points

### 4.2 Advanced Workflow Features

#### F-006: Conditional Branching
**Priority:** P0 (Must Have)

**Description:** Support conditional branching in workflows based on task outputs.

**Technical Requirements:**
- Branch selection via `from_branch` edge attribute
- Branch value extraction from task output (branch, branch_name, or configured key)
- Default branch fallback

**Acceptance Criteria:**
- [ ] Tasks with multiple outgoing edges route based on output
- [ ] `branch_output_key` in metadata customizes extraction
- [ ] Default edges execute when no match

#### F-007: Map/Fan-out Pattern
**Priority:** P0 (Must Have)

**Description:** Support parallel processing of collections via MapNode.

**Technical Requirements:**
- MapNode iterates over input list
- Each item processed by inner flow
- Results collected (parallel or sequential)

**Acceptance Criteria:**
- [ ] `map_input_key` specifies list field
- [ ] `parallel: true` uses `when_all` for concurrent execution
- [ ] Results maintain order correspondence with inputs

#### F-008: Retry Policies
**Priority:** P0 (Must Have)

**Description:** Configure automatic retry behavior for workflow tasks.

**Technical Requirements:**
- Exponential backoff support
- Configurable max attempts
- Optional retry timeout

**Acceptance Criteria:**
- [ ] `retry_policy` in task metadata creates `RetryPolicy`
- [ ] Transient failures trigger retries
- [ ] Permanent failures propagate after max attempts

#### F-009: Task Timeouts
**Priority:** P0 (Must Have)

**Description:** Configure maximum execution time for workflow tasks.

**Technical Requirements:**
- Timer-based timeout using Dapr `create_timer`
- Race between task and timeout
- Clean error on timeout

**Acceptance Criteria:**
- [ ] `timeout_seconds` in metadata sets limit
- [ ] `TimeoutError` raised when exceeded
- [ ] Task cancellation attempted on timeout

#### F-010: Compensation/Saga Pattern
**Priority:** P0 (Must Have)

**Description:** Support rollback operations when workflow fails.

**Technical Requirements:**
- Compensation activity per task
- Reverse-order execution on failure
- Error context passed to compensation

**Acceptance Criteria:**
- [ ] `compensation_activity` specifies rollback function
- [ ] Compensations run in reverse execution order
- [ ] Original error preserved after compensations

### 4.3 Tool Support

#### F-011: Server Tool Conversion
**Priority:** P0 (Must Have)

**Description:** Convert OAS ServerTool to Dapr tool functions.

**Technical Requirements:**
- Extract name, description, inputs, outputs
- Bind to implementation from registry
- Apply `@tool` decorator for Dapr

**Acceptance Criteria:**
- [ ] Tool schema preserved in conversion
- [ ] Implementation callable from agent
- [ ] Missing implementation raises clear error

#### F-012: MCP Tool Support
**Priority:** P0 (Must Have)

**Description:** Support Model Context Protocol tools with SSE/HTTP transport.

**Technical Requirements:**
- Parse `client_transport` configuration
- Create MCP client connection
- Preserve transport settings in export

**Acceptance Criteria:**
- [ ] MCPTool with SSE transport connects to server
- [ ] Tool invocations route through MCP client
- [ ] Export preserves transport configuration

### 4.4 LLM Configuration

#### F-013: Multi-Provider LLM Support
**Priority:** P0 (Must Have)

**Description:** Support multiple LLM providers in configuration.

**Supported Providers:**
| Provider | OAS Type | Dapr Client |
|----------|----------|-------------|
| OpenAI | OpenAIConfig | OpenAIChatClient |
| vLLM | VllmConfig | OpenAIChatClient (compatible) |
| Ollama | OllamaConfig | OpenAIChatClient (compatible) |
| OCI GenAI | OciGenAiConfig | Custom client |

**Acceptance Criteria:**
- [ ] All providers create functional LLM clients
- [ ] Generation parameters (temperature, max_tokens) apply
- [ ] API keys securely handled

---

## 5. Non-Functional Requirements

### 5.1 Performance

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| Conversion latency | < 100ms per component | Interactive usage |
| Memory footprint | < 50MB for 100 components | Embedded in services |
| Startup time | < 500ms | Fast initialization |
| Throughput | > 1000 conversions/sec | Batch processing |

### 5.2 Reliability

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| Conversion success rate | 99.9% | Production stability |
| Roundtrip fidelity | 100% | No data loss |
| Error recovery | Graceful degradation | Partial results vs total failure |
| Idempotency | Deterministic output | Reproducible builds |

### 5.3 Security

| Requirement | Implementation |
|-------------|----------------|
| API key handling | Never log, mask in errors |
| Input validation | Sanitize all external inputs |
| Code generation | Escape user content in templates |
| Dependency security | Automated vulnerability scanning |

### 5.4 Maintainability & Code Quality Gates

#### Test Coverage Requirements

| Requirement | Target | Notes |
|-------------|--------|-------|
| Unit test coverage | 100% | All functions and branches |
| Integration test coverage | 100% | All conversion paths |
| Edge case coverage | 100% | Error handling, boundary conditions |
| Documentation coverage | 100% | All public API with examples |
| Type coverage | 100% | Strict typing, no `Any` escape hatches |

#### Static Analysis Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| **Ruff** (linter) | Lint + format | Rule sets: S, PT, ANN, ARG, T20, RET, SLF, PIE, PLC, PLE, PLR, PLW, E, F, W, I, N, UP, B, A, C4, SIM, TCH, RUF, BLE |
| **Ruff** (formatter) | Code formatting | PEP8 compliant |
| **mypy** | Type checking | Strict mode, no implicit optional |
| **pyright** | Type checking | Strict mode (cross-validation with mypy) |
| **pylint** | Duplicate code detection | Only R0801 (duplicate-code), min-similarity-lines=4 |
| **vulture** | Dead code detection | 90% confidence minimum |
| **radon** | Complexity metrics | Report all functions |
| **xenon** | Complexity gate | Grade A for all modules |

> **Note on type checkers**: Both mypy and pyright are used because they have slightly different philosophies and can catch different edge cases. Astral's ty was considered but is still in beta (~15% conformance) and not recommended for CI gates requiring maximum accuracy. ty may be used optionally for fast local feedback during development.

> **Tooling deviations from DaprKit standard:** This project's Ruff configuration is a **superset** of the standard DaprKit rule set (defined in `daprkit-specs/standards/TOOLING.md`). Additional rules include: ANN (type annotations), PT (pytest style), T20 (no print), RET (return consistency), SLF (private member access), PIE (misc lints), N (pep8-naming), BLE (blind except). These are stricter and fully compatible. Additionally, **pylint** is used exclusively for duplicate code detection (R0801) — this is project-specific and not part of the standard DaprKit toolchain.

#### Ruff Rule Sets Explained

```toml
[tool.ruff.lint]
select = [
    "S",    # flake8-bandit (security)
    "PT",   # flake8-pytest-style
    "ANN",  # flake8-annotations (type hints)
    "ARG",  # flake8-unused-arguments
    "T20",  # flake8-print (no print statements)
    "RET",  # flake8-return (return consistency)
    "SLF",  # flake8-self (private member access)
    "PIE",  # flake8-pie (misc lints)
    "PLC",  # pylint convention
    "PLE",  # pylint error
    "PLR",  # pylint refactor
    "PLW",  # pylint warning
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "W",    # pycodestyle warnings
    "I",    # isort (import sorting)
    "N",    # pep8-naming
    "UP",   # pyupgrade (modern Python)
    "B",    # flake8-bugbear
    "A",    # flake8-builtins
    "C4",   # flake8-comprehensions
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "RUF",  # ruff-specific rules
    "BLE",  # flake8-blind-except
]
```

#### Pylint Duplicate Code Configuration

Pylint is used **exclusively** for duplicate code detection (similarity checker):

```toml
[tool.pylint.main]
# Only enable the similarity checker
enable = ["R0801"]  # duplicate-code
disable = ["all"]

[tool.pylint.similarities]
min-similarity-lines = 4
ignore-comments = true
ignore-docstrings = true
ignore-imports = true
ignore-signatures = true
```

Usage:
```bash
pylint --disable=all --enable=R0801 src/
```

#### Complexity Gates

| Metric | Tool | Threshold | Action |
|--------|------|-----------|--------|
| Cyclomatic complexity | radon/xenon | Grade A (≤5) | Block merge |
| Maintainability index | radon | Grade A (≥20) | Block merge |
| Halstead effort | radon | Report only | Review |
| Lines per function | ruff PLR0915 | ≤50 | Block merge |
| Arguments per function | ruff PLR0913 | ≤5 | Block merge |

### 5.5 Compatibility

| Requirement | Versions |
|-------------|----------|
| Python | >= 3.10 |
| pyagentspec | >= 25.4.1 |
| dapr-agents | >= 0.10.5 |
| dapr | >= 1.16.0 |

### 5.6 Dependencies

#### Runtime Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pyagentspec` | >= 25.4.1 | OAS SDK — parsing and serialization of Open Agent Spec |
| `dapr-agents` | >= 0.10.5 | Dapr Agents framework — agent/workflow runtime |
| `pydantic` | >= 2.0 | Data validation and settings management |
| `pyyaml` | >= 6.0 | YAML parsing for OAS specification files |

#### Optional Dependencies

| Group | Packages | Purpose |
|-------|----------|---------|
| `[logging]` | `structlog >= 24.0` | Structured logging for conversion operations |
| `[validation]` | `jsonschema >= 4.0` | Strict JSON Schema validation mode (RF-010) |
| `[testing]` | `hypothesis >= 6.0` | Property-based testing for roundtrip fidelity |

#### Development Dependencies

| Package | Purpose |
|---------|---------|
| `pytest >= 8.0` | Test runner |
| `pytest-asyncio >= 0.23` | Async test support |
| `pytest-cov >= 4.0` | Coverage enforcement (100%) |
| `mypy >= 1.8` | Type checking (strict mode) |
| `pyright >= 1.1` | Type checking (cross-validation) |
| `ruff >= 0.2` | Linting and formatting |
| `xenon >= 0.9` | Complexity gates (grade A) |
| `vulture >= 2.11` | Dead code detection |
| `radon >= 6.0` | Complexity metrics |
| `pylint >= 3.0` | Duplicate code detection (R0801 only) |

### 5.7 Async/Threading Strategy

**Critical:** The Dapr Python SDK has synchronous operations. Using `asyncio.to_thread()` naively can cause issues with libraries that have their own thread-local state (e.g., SQLAlchemy sessions).

**Known issues:**
- Dapr client operations in `to_thread()` may not preserve async context
- Cache decorators on functions using thread-sensitive libraries may have thread context problems

**Required approach:**
- Test with thread-sensitive libraries when integrating with downstream code
- Consider alternatives to `to_thread()` where appropriate:
  - Native async Dapr operations (when available)
  - Proper context propagation
  - Task-based isolation

**Design principle:** All I/O must be async. The sync API is a convenience wrapper around the async core.

| Operation | Strategy | Rationale |
|-----------|----------|-----------|
| File I/O (YAML/JSON loading) | `asyncio.to_thread(path.read_text)` | File reads are blocking I/O |
| pyagentspec deserialization | Sync (no wrapping) | CPU-bound, no I/O; runs in microseconds |
| Dapr agent creation | `asyncio.to_thread(dapr_sdk_call)` | Dapr Python SDK is synchronous |
| MCP tool connections | Async-native via MCP client | MCP protocol is natively async |
| Pydantic validation | Sync (no wrapping) | CPU-bound, no I/O |

#### Thread Safety

Converter classes (`AgentConverter`, `FlowConverter`, etc.) are **stateless** and safe to share across async contexts. The `IDGenerator` uses an internal counter and must be instantiated per-conversion session or protected with `asyncio.Lock` when shared.

```python
import asyncio
from pathlib import Path

class AsyncDaprAgentSpecLoader:
    """Async-first loader for OAS specifications.

    Args:
        tool_registry: Mapping of tool names to implementations.
    """

    async def load_yaml_file(self, path: Path) -> DaprAgentConfig | WorkflowDefinition:
        """Load and parse an OAS YAML file.

        Args:
            path: Path to the OAS YAML specification file.

        Returns:
            Parsed configuration (agent or workflow).

        Raises:
            ConversionError: If the file cannot be parsed or converted.
        """
        content = await asyncio.to_thread(path.read_text)
        return self.load_yaml(content)

    async def create_agent(self, config: DaprAgentConfig, **kwargs: Any) -> Any:
        """Create a Dapr agent from configuration.

        Wraps the synchronous dapr-agents SDK with asyncio.to_thread().
        """
        return await asyncio.to_thread(self._create_agent_sync, config, **kwargs)


# Sync wrapper for convenience
class DaprAgentSpecLoader:
    """Synchronous wrapper around AsyncDaprAgentSpecLoader."""

    def __init__(self, **kwargs: Any) -> None:
        self._async = AsyncDaprAgentSpecLoader(**kwargs)

    def load_yaml_file(self, path: Path) -> DaprAgentConfig | WorkflowDefinition:
        return asyncio.run(self._async.load_yaml_file(path))
```

### 5.8 Test Strategy

#### Test Naming Convention

Tests follow the `test_<function>_<scenario>_<expected_result>` pattern:

```python
# Agent conversion
def test_from_oas_agent_with_tools_returns_react_agent(): ...
def test_from_oas_agent_with_durable_keys_returns_durable_agent(): ...
def test_from_oas_agent_explicit_type_overrides_inference(): ...

# Workflow conversion
def test_from_oas_flow_linear_returns_sequential_workflow(): ...
def test_from_oas_flow_circular_reference_raises_conversion_error(): ...
def test_from_oas_flow_with_subflows_registers_children_first(): ...

# Tool resolution
def test_resolve_tool_registered_returns_implementation(): ...
def test_resolve_tool_mcp_creates_client_with_transport(): ...
def test_resolve_tool_missing_raises_conversion_error(): ...

# LLM provider mapping
def test_map_llm_openai_config_returns_openai_client(): ...
def test_map_llm_unknown_provider_defaults_to_openai_compatible(): ...

# Roundtrip fidelity
def test_roundtrip_agent_config_preserves_all_fields(): ...
def test_roundtrip_workflow_definition_preserves_all_fields(): ...
```

#### Key Test Scenarios

| Category | Scenarios | Coverage Target |
|----------|-----------|----------------|
| **Agent conversion** | AssistantAgent, ReActAgent, DurableAgent (3 types) | 100% |
| **Workflow conversion** | Linear, branching, map/fan-out, compensation | 100% |
| **Tool resolution** | ServerTool, MCPTool, missing tool | 100% |
| **LLM provider mapping** | OpenAI, vLLM, Ollama, OCI GenAI, unknown | 100% |
| **Roundtrip fidelity** | Agent config, workflow definition, edge cases | 100% |
| **Error handling** | Invalid YAML, missing fields, circular refs, type mismatches | 100% |
| **Validation** | Orphan nodes, missing start, duplicate edges, invalid refs | 100% |

#### Test Organization

```
tests/
├── conftest.py                    # Shared fixtures, seeded IDGenerator
├── fixtures/
│   ├── simple_agent.yaml          # Basic OAS agent spec
│   ├── react_agent_with_tools.yaml
│   ├── durable_agent.yaml
│   ├── simple_flow.yaml           # Linear workflow
│   ├── branching_flow.yaml        # Conditional branching
│   ├── map_flow.yaml              # Fan-out pattern
│   └── compensation_flow.yaml     # Saga pattern
├── test_agent_converter.py
├── test_flow_converter.py
├── test_tool_converter.py
├── test_llm_converter.py
├── test_loader.py
├── test_exporter.py
├── test_validator.py
├── test_models.py
├── test_roundtrip.py              # Roundtrip fidelity tests
├── property/
│   └── test_roundtrip_properties.py  # Hypothesis property-based tests
└── integration/
    ├── conftest.py                # Dapr test container setup
    └── test_dapr_workflow.py      # Real Dapr runtime tests
```

---

## 6. API Surface

### 6.1 OAS Agent YAML Input Example

```yaml
# simple_agent.yaml — Example OAS specification for a ReAct agent
apiVersion: agent-spec/v1
kind: Agent
metadata:
  name: research-assistant
  annotations:
    dapr_agent_type: ReActAgent
spec:
  description: "AI research assistant with web search capabilities"
  system_prompt: |
    You are a research assistant. Use available tools to find
    accurate information and provide well-sourced answers.
  llm_config:
    type: OpenAIConfig
    model: gpt-4o
    temperature: 0.7
    max_tokens: 4096
  tools:
    - name: web_search
      type: ServerTool
      description: "Search the web for information"
      inputs:
        - name: query
          type: string
          description: "Search query"
      outputs:
        - name: results
          type: string
          description: "Search results"
    - name: code_interpreter
      type: MCPTool
      description: "Execute code snippets"
      client_transport:
        type: sse
        url: "http://localhost:8080/mcp"
  inputs:
    - name: question
      type: string
      description: "User question"
  outputs:
    - name: answer
      type: string
      description: "Agent response"
```

### 6.2 Complete Usage Examples

#### Importing an OAS Agent

```python
from pathlib import Path

from dapr_agents_oas_adapter import DaprAgentSpecLoader

# Create loader with tool registry
loader = DaprAgentSpecLoader(
    tool_registry={"web_search": web_search_fn},
)

# Load OAS YAML and parse into config
config = loader.load_yaml_file(Path("simple_agent.yaml"))
print(config.name)        # "research-assistant"
print(config.agent_type)  # "ReActAgent"

# Create executable Dapr agent from config
agent = loader.create_agent(config)

# Agent is now ready to receive messages
response = agent.run("What is the capital of France?")
```

#### Exporting a Dapr Agent to OAS

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter

exporter = DaprAgentSpecExporter()

# Export from a running Dapr agent instance
oas_config = exporter.from_dapr_agent(agent)

# Serialize to YAML file
exporter.to_yaml_file(oas_config, Path("exported_agent.yaml"))

# Or get YAML string
yaml_str = exporter.to_yaml(oas_config)
```

#### Workflow Conversion

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

loader = DaprAgentSpecLoader()

# Load OAS Flow and convert to Dapr workflow
workflow_def = loader.load_yaml_file(Path("branching_flow.yaml"))
workflow_fn = loader.create_workflow(
    workflow_def,
    task_implementations={"classify": classify_fn, "process": process_fn},
)

# Register and execute with Dapr workflow runtime
from dapr.ext.workflow import WorkflowRuntime

runtime = WorkflowRuntime()
runtime.register_workflow(workflow_fn)
runtime.start()

result = runtime.run_workflow(workflow_fn.__name__, input={"data": "test"})
```

#### Async Usage

```python
import asyncio
from pathlib import Path

from dapr_agents_oas_adapter import AsyncDaprAgentSpecLoader

async def main() -> None:
    loader = AsyncDaprAgentSpecLoader(
        tool_registry={"web_search": web_search_fn},
    )
    config = await loader.load_yaml_file(Path("simple_agent.yaml"))
    agent = await loader.create_agent(config)

asyncio.run(main())
```

### 6.3 Public API Summary

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `DaprAgentSpecLoader` | Sync loader for OAS specs | `load_yaml_file()`, `load_yaml()`, `load_dict()`, `create_agent()`, `create_workflow()` |
| `AsyncDaprAgentSpecLoader` | Async loader for OAS specs | Same as above, all async |
| `DaprAgentSpecExporter` | Export Dapr agents to OAS | `from_dapr_agent()`, `from_dapr_workflow()`, `to_yaml()`, `to_yaml_file()`, `to_dict()` |
| `DaprAgentConfig` | Agent configuration model | Pydantic model with `frozen=True` |
| `WorkflowDefinition` | Workflow configuration model | Pydantic model with `frozen=True` |
| `ConversionError` | Conversion failure | Includes component context and suggestion |
| `ValidationError` | Validation failure | Includes field path information |

---

## 7. Technical Architecture

### 7.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    dapr-agents-oas-adapter                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │ DaprAgentSpec   │              │ DaprAgentSpec   │           │
│  │    Loader       │              │    Exporter     │           │
│  │  (Public API)   │              │  (Public API)   │           │
│  └────────┬────────┘              └────────┬────────┘           │
│           │                                │                     │
│           ▼                                ▼                     │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              Converter Layer                         │        │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │        │
│  │  │  Agent   │ │   Flow   │ │   Node   │ │  Tool  │  │        │
│  │  │Converter │ │Converter │ │Converter │ │Converter│  │        │
│  │  └──────────┘ └──────────┘ └──────────┘ └────────┘  │        │
│  │  ┌──────────┐                                        │        │
│  │  │   LLM    │                                        │        │
│  │  │Converter │                                        │        │
│  │  └──────────┘                                        │        │
│  └─────────────────────────────────────────────────────┘        │
│           │                                │                     │
│           ▼                                ▼                     │
│  ┌─────────────────┐              ┌─────────────────┐           │
│  │     Types       │              │     Utils       │           │
│  │ (Data Models)   │              │ (Helpers)       │           │
│  └─────────────────┘              └─────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
           │                                │
           ▼                                ▼
┌─────────────────┐              ┌─────────────────┐
│   pyagentspec   │              │   dapr-agents   │
│  (OAS Library)  │              │   (Runtime)     │
└─────────────────┘              └─────────────────┘
```

### 7.2 Data Models

> **Convention:** All data models use Pydantic v2 with `frozen=True` and `Field()` descriptions per DaprKit standards. Workflow types that currently use dataclasses will be migrated to Pydantic in RF-002.

#### LlmProviderConfig

```python
from pydantic import BaseModel, ConfigDict, Field


class LlmProviderConfig(BaseModel):
    """LLM provider configuration."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(description="Provider identifier: openai, vllm, ollama, oci")
    model_name: str = Field(description="Model name or deployment identifier")
    api_key: str | None = Field(default=None, description="API key (masked in logs)")
    base_url: str | None = Field(default=None, description="Custom API endpoint URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, ge=1, description="Maximum tokens in response")
```

#### Environment Variable Resolution

When `api_key` or `base_url` are `None` in the OAS specification, the adapter delegates credential resolution to the underlying LLM client libraries at agent creation time (not at spec loading time):

| Provider | Environment Variable | Default |
|----------|---------------------|---------|
| OpenAI | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Required |
| vLLM | `OPENAI_API_KEY`, `OPENAI_BASE_URL` | Uses OpenAI-compatible client |
| Ollama | `OLLAMA_HOST` | `http://localhost:11434` |
| OCI GenAI | `OCI_CONFIG_FILE` | `~/.oci/config` |

This ensures sensitive credentials never appear in OAS YAML files:

```yaml
# OAS spec — no API key, resolved from environment
llm_config:
  type: OpenAIConfig
  model: gpt-4o
  temperature: 0.7
  # api_key omitted → resolved from OPENAI_API_KEY at runtime
```

```python
from pathlib import Path

from dapr_agents_oas_adapter import DaprAgentSpecLoader

loader = DaprAgentSpecLoader(tool_registry={"web_search": web_search_fn})

# Spec loading does NOT require API key — it's a pure data transformation
config = loader.load_yaml_file(Path("agent.yaml"))
assert config.llm_config.api_key is None  # Not in spec

# Agent creation resolves credentials from environment via the LLM client
agent = loader.create_agent(config)
# Underlying OpenAIChatClient reads OPENAI_API_KEY from os.environ
```

#### ToolDefinition

```python
class ToolDefinition(BaseModel):
    """Tool definition extracted from OAS specification."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Tool function name")
    description: str = Field(default="", description="Human-readable tool description")
    tool_type: str = Field(description="Tool type: server, mcp")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for tool inputs")
    output_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for tool outputs")
    transport_config: dict[str, str] | None = Field(default=None, description="MCP transport configuration")
```

#### DaprAgentConfig

```python
class DaprAgentConfig(BaseModel):
    """Configuration for a Dapr agent derived from OAS specification."""

    model_config = ConfigDict(frozen=True)

    # Identity
    name: str = Field(description="Unique agent identifier")
    role: str | None = Field(default=None, description="Agent persona or role description")
    goal: str | None = Field(default=None, description="Agent objective or mission")

    # Behavior
    instructions: list[str] = Field(default_factory=list, description="Behavioral guidelines")
    system_prompt: str | None = Field(default=None, description="Raw prompt template")
    tools: list[str] = Field(default_factory=list, description="Tool names to bind")
    input_variables: list[str] = Field(default_factory=list, description="Template placeholders")

    # Infrastructure
    message_bus_name: str = Field(default="messagepubsub", description="Dapr pub/sub component name")
    state_store_name: str = Field(default="statestore", description="Dapr state store component name")
    agents_registry_store_name: str = Field(default="agentsregistry", description="Agents registry store name")
    service_port: int = Field(default=8000, ge=1, le=65535, description="Service listening port")

    # Type
    agent_type: str | None = Field(default=None, description="Agent type: AssistantAgent, ReActAgent, or DurableAgent")
    llm_config: LlmProviderConfig | None = Field(default=None, description="LLM provider configuration")
    tool_definitions: list[ToolDefinition] = Field(default_factory=list, description="Tool definitions from OAS spec")

    # DurableAgent-specific
    agent_topic: str | None = Field(default=None, description="Topic for agent-to-agent messaging")
    broadcast_topic: str | None = Field(default=None, description="Topic for broadcast messages")
    state_key_prefix: str | None = Field(default=None, description="Prefix for state store keys")
    memory_store_name: str | None = Field(default=None, description="State store for agent memory")
    memory_session_id: str | None = Field(default=None, description="Session ID for memory scoping")
    registry_team_name: str | None = Field(default=None, description="Team name for agent registry")
```

#### WorkflowDefinition

```python
class WorkflowDefinition(BaseModel):
    """Workflow definition converted from OAS Flow specification."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Workflow identifier")
    description: str | None = Field(default=None, description="Human-readable description")
    flow_id: str | None = Field(default=None, description="OAS component ID")
    tasks: list[WorkflowTaskDefinition] = Field(default_factory=list, description="Workflow task nodes")
    edges: list[WorkflowEdgeDefinition] = Field(default_factory=list, description="Connections between tasks")
    start_node: str | None = Field(default=None, description="Entry point task name")
    end_nodes: list[str] = Field(default_factory=list, description="Exit point task names")
    inputs: list[PropertySchema] = Field(default_factory=list, description="Input schema")
    outputs: list[PropertySchema] = Field(default_factory=list, description="Output schema")
    subflows: dict[str, WorkflowDefinition] = Field(default_factory=dict, description="Nested sub-workflow definitions")
```

#### WorkflowTaskDefinition

```python
class WorkflowTaskDefinition(BaseModel):
    """Individual task node within a workflow."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Task identifier")
    task_type: str = Field(description="Node type: start, end, llm, tool, agent, flow, map")
    config: dict[str, Any] = Field(default_factory=dict, description="Type-specific configuration")
    inputs: list[str] = Field(default_factory=list, description="Input field names")
    outputs: list[str] = Field(default_factory=list, description="Output field names")

    # Config keys by type:
    # llm: prompt_template, llm_config
    # tool: tool_name, tool
    # flow: flow_id, flow_name
    # map: inner_flow_id, parallel, map_input_key, map_item_key
    # all: retry_policy, timeout_seconds, compensation_activity
```

#### WorkflowEdgeDefinition

```python
class WorkflowEdgeDefinition(BaseModel):
    """Connection between two workflow tasks."""

    model_config = ConfigDict(frozen=True)

    from_node: str = Field(description="Source task name")
    to_node: str = Field(description="Target task name")
    from_branch: str | None = Field(default=None, description="Conditional branch value for routing")
    condition: str | None = Field(default=None, description="Future: expression-based condition")
    data_mapping: dict[str, str] = Field(default_factory=dict, description="Output key to input key mapping")
```

### 7.3 Conversion Flow

```
┌────────────────────────────────────────────────────────────────┐
│                        IMPORT FLOW                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  YAML/JSON ──► Deserializer ──► OAS Component ──► Converter    │
│      │              │                │                │         │
│      │              │                │                ▼         │
│      │              │                │         DaprAgentConfig  │
│      │              │                │         or               │
│      │              │                │         WorkflowDefinition│
│      │              │                │                │         │
│      │              │                │                ▼         │
│      │              │                │         create_agent()   │
│      │              │                │         or               │
│      │              │                │         create_workflow()│
│      │              │                │                │         │
│      │              │                │                ▼         │
│      │              │                │         Executable       │
│      │              │                │         Dapr Object      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│                        EXPORT FLOW                              │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dapr Instance ──► from_dapr_*() ──► Config ──► Converter      │
│       │                  │              │            │          │
│       │                  │              │            ▼          │
│       │                  │              │      OAS Component    │
│       │                  │              │            │          │
│       │                  │              │            ▼          │
│       │                  │              │      Serializer       │
│       │                  │              │            │          │
│       │                  │              │            ▼          │
│       │                  │              │      YAML/JSON        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 8. Refactoring Roadmap

> **Development Methodology**: This project will be developed using the **ralph-loop** technique with AI assistance. The roadmap is organized by **logical dependencies** between items, not by sprints or time estimates. Each refactoring item must be completed with 100% test coverage before proceeding.

### 8.1 Phase 1: Foundation Cleanup

#### RF-001: Fix Version Inconsistency
**Current State:** `__init__.py` has "0.4.1", `pyproject.toml` has "0.6.0"
**Target State:** Single source of truth for version

**Implementation:**
```python
# __init__.py
from importlib.metadata import version
__version__ = version("dapr-agents-oas-adapter")
```

**Acceptance Criteria:**
- [ ] Version retrieved from package metadata
- [ ] `dapr_agents_oas_adapter.__version__` matches pyproject.toml

#### RF-002: Unify Data Models
**Current State:** Mix of Pydantic (`DaprAgentConfig`) and dataclasses (workflow types)
**Target State:** Consistent use of Pydantic for all models

**Implementation:**
- Convert `WorkflowDefinition`, `WorkflowTaskDefinition`, `WorkflowEdgeDefinition` to Pydantic
- Add validators for business rules
- Enable JSON Schema generation

**Acceptance Criteria:**
- [ ] All data models inherit from `pydantic.BaseModel`
- [ ] Validators enforce required fields
- [ ] `model_json_schema()` works for all models

#### RF-003: Deterministic ID Generation
**Current State:** IDs generated with random components
**Target State:** Deterministic IDs with optional seed for testing

**Implementation:**
```python
class IDGenerator:
    def __init__(self, seed: int | None = None):
        self._counter = 0
        self._seed = seed

    def generate(self, prefix: str) -> str:
        self._counter += 1
        if self._seed is not None:
            return f"{prefix}_{self._seed}_{self._counter}"
        return f"{prefix}_{uuid4().hex[:8]}"
```

**Acceptance Criteria:**
- [ ] Same seed produces same IDs
- [ ] No seed produces unique IDs
- [ ] Tests use seeded generator

### 8.2 Phase 2: Code Quality

#### RF-004: Decompose FlowConverter.create_dapr_workflow()
**Current State:** ~350 lines with 15+ nested functions
**Target State:** < 50 lines with delegated responsibilities

**Implementation:**
- Extract `WorkflowExecutor` class for runtime logic
- Extract `BranchRouter` for edge selection
- Extract `TaskInputBuilder` for data mapping
- Extract `CompensationHandler` for saga pattern

**Target Structure:**
```python
class FlowConverter:
    def create_dapr_workflow(self, workflow_def, task_implementations):
        executor = WorkflowExecutor(
            workflow_def,
            task_implementations,
            branch_router=BranchRouter(),
            input_builder=TaskInputBuilder(),
            compensation_handler=CompensationHandler(),
        )
        return executor.build_workflow_function()
```

**Acceptance Criteria:**
- [ ] No function > 50 lines
- [ ] Cyclomatic complexity < 10
- [ ] Each class has single responsibility
- [ ] All existing tests pass

#### RF-005: Add Comprehensive Validation
**Current State:** Minimal validation (only id/name presence)
**Target State:** Full schema validation with clear errors

**Implementation:**
```python
class WorkflowValidator:
    def validate(self, workflow: WorkflowDefinition) -> list[ValidationError]:
        errors = []
        errors.extend(self._validate_structure(workflow))
        errors.extend(self._validate_references(workflow))
        errors.extend(self._validate_edges(workflow))
        return errors

    def _validate_structure(self, workflow):
        # At least one task
        # start_node exists
        # end_nodes exist

    def _validate_references(self, workflow):
        # All edge nodes exist
        # All subflow references valid

    def _validate_edges(self, workflow):
        # No orphan tasks
        # End nodes have no outgoing edges
        # No duplicate edges
```

**Acceptance Criteria:**
- [ ] Invalid workflows fail fast with list of errors
- [ ] Error messages include field path
- [ ] Validation is optional (for partial conversions)

#### RF-006: Improve Error Messages
**Current State:** Generic error messages
**Target State:** Actionable error messages with context

**Implementation** (uses exception hierarchy from section 3.3):
```python
# ConversionError is defined in section 3.3 with component + suggestion support.
# All raise sites must use standard Python exception chaining (`from e`):

try:
    dapr_config = agent_converter.from_oas(oas_agent)
except KeyError as e:
    raise ConversionError(
        f"Missing required field in agent spec: {e}",
        component=oas_agent,
        suggestion="Ensure the OAS agent has all required fields (name, llm_config)",
    ) from e  # <-- Always chain with `from e`

try:
    workflow_fn = flow_converter.create_dapr_workflow(workflow_def)
except RecursionError as e:
    raise ConversionError(
        "Circular reference detected in workflow subflows",
        component=workflow_def,
        suggestion="Check for cyclic $referenced_components in the OAS Flow",
    ) from e
```

**Acceptance Criteria:**
- [ ] All errors include component context when applicable
- [ ] Errors suggest resolution when possible
- [ ] All `raise` statements use `from e` for exception chaining
- [ ] No custom `caused_by` parameter — use standard Python `__cause__`

### 8.3 Phase 3: Enterprise Features

#### RF-007: Add Observability
**Current State:** No logging or metrics
**Target State:** Structured logging and optional metrics

**Implementation** (standard `logging` interface per DaprKit CONVENTIONS.md):
```python
import logging

logger = logging.getLogger(__name__)

class FlowConverter:
    """Converts OAS Flow specifications to Dapr workflow definitions."""

    def from_oas(self, component: Any) -> WorkflowDefinition:
        """Convert an OAS Flow component to a WorkflowDefinition.

        Args:
            component: OAS Flow component from pyagentspec.

        Returns:
            Converted workflow definition.

        Raises:
            ConversionError: If the flow cannot be converted.
        """
        logger.info("converting_flow", extra={"flow_id": component.id, "name": component.name})
        try:
            result = self._convert(component)
            logger.info("flow_converted", extra={"tasks": len(result.tasks)})
            return result
        except Exception as e:
            logger.error("conversion_failed", extra={"error": str(e), "flow_id": component.id})
            raise
```

> **Note:** If the project opts for `structlog` instead of standard `logging`, this is a deliberate deviation from DaprKit convention and should be installed via the `[logging]` optional dependency group. The standard `logging.getLogger(__name__)` with `extra={}` kwargs is the default convention.

**Acceptance Criteria:**
- [ ] All conversions logged at INFO level
- [ ] Errors logged at ERROR level with context
- [ ] Uses `logging.getLogger(__name__)` with `extra={}` kwargs (or documents structlog as explicit deviation)
- [ ] Optional metrics export (Prometheus/OpenTelemetry)

#### RF-008: Add Caching Layer
**Current State:** No caching, repeated conversions re-process
**Target State:** Optional conversion cache

**Implementation:**
```python
class CachedLoader(DaprAgentSpecLoader):
    def __init__(self, cache: ConversionCache | None = None, **kwargs):
        super().__init__(**kwargs)
        self._cache = cache or InMemoryCache()

    def load_yaml(self, content: str) -> DaprAgentConfig | WorkflowDefinition:
        cache_key = hashlib.sha256(content.encode()).hexdigest()
        if cached := self._cache.get(cache_key):
            return cached
        result = super().load_yaml(content)
        self._cache.set(cache_key, result)
        return result
```

**Acceptance Criteria:**
- [ ] Cache hit avoids re-parsing
- [ ] Cache is optional (off by default)
- [ ] Cache invalidation on tool_registry change

#### RF-009: Add Async Support
**Current State:** Synchronous only
**Target State:** Async-first with sync wrapper

> See section 5.7 (Async/Threading Strategy) for the design approach, per-operation strategy table, and implementation examples.

**Acceptance Criteria:**
- [ ] All I/O operations have async variants
- [ ] Sync API unchanged (backwards compatible)
- [ ] Async operations run concurrently when possible
- [ ] Thread safety documented for shared converter instances

#### RF-010: Add Schema Validation Mode
**Current State:** Trusts input structure
**Target State:** Optional strict JSON Schema validation

**Implementation:**
```python
class StrictLoader(DaprAgentSpecLoader):
    def __init__(self, schema_version: str = "25.4.1", **kwargs):
        super().__init__(**kwargs)
        self._schema = self._load_schema(schema_version)

    def load_yaml(self, content: str):
        data = yaml.safe_load(content)
        jsonschema.validate(data, self._schema)  # Raises on invalid
        return super().load_yaml(content)
```

**Acceptance Criteria:**
- [ ] Invalid OAS rejected before conversion
- [ ] Schema errors include JSON path
- [ ] Schema version is configurable

### 8.4 Phase 4: Testing & Documentation

#### RF-011: Add Property-Based Testing
**Current State:** Example-based tests only
**Target State:** Property-based tests for conversion isomorphism

**Implementation:**
```python
from hypothesis import given, strategies as st

@given(agent_config=st.builds(DaprAgentConfig, ...))
def test_agent_roundtrip_preserves_data(agent_config):
    exporter = DaprAgentSpecExporter()
    loader = DaprAgentSpecLoader()

    exported = exporter.to_dict(agent_config)
    imported = loader.load_dict(exported)

    assert imported.name == agent_config.name
    assert imported.role == agent_config.role
    # ... all fields preserved
```

**Acceptance Criteria:**
- [ ] Roundtrip tests for all data models
- [ ] Edge cases discovered and fixed
- [ ] CI runs property tests (limited iterations)

#### RF-012: Add Integration Tests
**Current State:** Unit tests with mocks
**Target State:** Integration tests with real Dapr runtime

**Implementation:**
```python
@pytest.fixture(scope="session")
def dapr_runtime():
    # Start Dapr sidecar in test mode
    with DaprTestContainer() as container:
        yield container.client

def test_workflow_executes_in_dapr(dapr_runtime):
    loader = DaprAgentSpecLoader()
    workflow_def = loader.load_yaml_file("fixtures/simple_flow.yaml")
    workflow = loader.create_workflow(workflow_def)

    runtime = WorkflowRuntime()
    runtime.register_workflow(workflow)
    runtime.start()

    result = runtime.run_workflow(workflow.__name__, {"input": "test"})
    assert result["status"] == "completed"
```

**Acceptance Criteria:**
- [ ] CI runs integration tests (separate job)
- [ ] Real Dapr runtime exercises all paths
- [ ] Test fixtures cover all node types

#### RF-013: Generate API Documentation
**Current State:** Docstrings only
**Target State:** Published API docs with examples

**Implementation:**
- Add mkdocs with mkdocstrings
- Generate from docstrings
- Include usage examples
- Publish to GitHub Pages

**Acceptance Criteria:**
- [ ] All public API documented
- [ ] Examples for common use cases
- [ ] Auto-updated on release

---

## 9. Development Strategy

### 9.1 Methodology: Ralph-Loop with AI Assistance

This project will be developed using the **ralph-loop** technique - iterative AI-assisted development cycles where each iteration:

1. **Defines** clear scope of work item
2. **Implements** with AI assistance
3. **Tests** with 100% coverage
4. **Validates** against quality gates
5. **Iterates** until approval on all gates

### 9.2 Dependency Graph

Refactoring items have the following dependencies:

```
RF-001 (Version Fix)
    │
    └──► RF-002 (Pydantic Models) ──► RF-005 (Validation)
              │
              └──► RF-003 (ID Generator)
              │
              └──► RF-004 (FlowConverter Decomposition)
                        │
                        └──► RF-006 (Error Messages)
                        │
                        └──► RF-007 (Observability)
                        │
                        └──► RF-008 (Caching)
                        │
                        └──► RF-009 (Async Support)
                        │
                        └──► RF-010 (Schema Validation)

RF-011 (Property Tests) ◄── Requires: RF-002, RF-003
RF-012 (Integration Tests) ◄── Requires: RF-004, RF-009
RF-013 (Documentation) ◄── Requires: All above
```

### 9.3 Quality Gate Checklist (Per Item)

Each refactoring item is only complete when:

- [ ] Implementation finalized
- [ ] Unit tests: 100% coverage
- [ ] Integration tests: 100% coverage
- [ ] Edge cases covered
- [ ] Ruff lint: 0 errors
- [ ] Ruff format: 0 changes
- [ ] mypy --strict: 0 errors
- [ ] pyright: 0 errors
- [ ] pylint duplicate-code: 0 duplications
- [ ] vulture: 0 dead code
- [ ] xenon: Grade A on all modules
- [ ] Documentation updated

### 9.4 Validation Strategy

The project will be validated on real projects before upstream contribution:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VALIDATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Implement   │───►│   Test in    │───►│  Refine &    │          │
│  │   Feature    │    │ Real Projects│    │   Stabilize  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌─────────────────────────────────────────────────────┐           │
│  │              ENTERPRISE USE CASES                    │           │
│  │  • Workflow branching scenarios                      │           │
│  │  • Map/fan-out parallel processing                   │           │
│  │  • Retry/timeout policies                            │           │
│  │  • Compensation/saga patterns                        │           │
│  └─────────────────────────────────────────────────────┘           │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────┐           │
│  │              UPSTREAM CONTRIBUTION                   │           │
│  │  • Offer specification to dapr-agents               │           │
│  │  • Submit PR with implementation                     │           │
│  │  • Or maintain as independent adapter               │           │
│  └─────────────────────────────────────────────────────┘           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.5 Upstream Contribution Path

The Dapr Agents team has shown openness to analyzing potential OAS support in an initial contact. The strategy is:

1. **Develop this library** as an independent implementation
2. **Validate in real enterprise projects** within the organization
3. **Document learnings** and discovered edge cases
4. **Offer to the dapr-agents project** as:
   - Feature request with detailed specification
   - Pull request based on learnings from this lib
   - Or maintain as independent adapter if there's divergence in vision

---

## 10. Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| pyagentspec API changes | High | High | Pin version, adapter pattern for API |
| Dapr SDK breaking changes | Medium | High | Version matrix testing, compatibility shims |
| Performance degradation from refactoring | Low | Medium | Benchmark suite, performance tests in CI |
| Async migration breaks sync users | Medium | High | Sync wrapper, deprecation period |
| Python 3.10 compatibility issues | Low | Medium | CI matrix testing across 3.10, 3.11, 3.12 |

### Business/Strategic Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Low adoption | Medium | High | Example projects, documentation, blog posts |
| Competitor adapter emerges | Low | Medium | Feature velocity, community engagement |
| OAS spec evolves incompatibly | Low | High | Active OAS community participation |
| A2A protocol gains dominance over OAS | Medium | Medium | Monitor adoption trends, prepare A2A extension |
| Dapr Agents adds native OAS support | Low | Positive | Contribute learnings upstream, transition gracefully |

### A2A Protocol Adoption Risk

**Context**: Google's A2A (Agent-to-Agent) protocol has strong momentum with 50+ partners.

**Analysis**:
- A2A and OAS are **complementary**, not competing
- OAS defines *agent structure*, A2A defines *inter-agent communication*
- Dapr Agents may add A2A support independently of OAS
- This adapter can coexist with future A2A adapters

**Recommended action**: Monitor A2A adoption in the Dapr ecosystem. If it gains traction, consider extension to allow OAS agents to participate in A2A communication.

---

## 11. Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| OAS | Open Agent Spec - Oracle's specification for describing AI agents |
| Dapr | Distributed Application Runtime - CNCF project for microservices |
| Flow | OAS term for a workflow with nodes and edges |
| Workflow | Dapr term for a durable, resumable process |
| Converter | Class that transforms between OAS and Dapr representations |
| MCP | Model Context Protocol - standard for tool server communication |

### B. Related Documents

- [Open Agent Spec Documentation](https://oracle.github.io/agent-spec/)
- [Dapr Agents Documentation](https://docs.dapr.io/developing-applications/dapr-agents/)
- [PyAgentSpec Repository](https://github.com/oracle/agent-spec)
- [Project README](./README.md)
- [CLAUDE.md](./CLAUDE.md)

### C. Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-28 | Initial PRD creation |
| 1.1 | 2025-01-28 | Added Market Analysis section; Updated Python requirement to 3.10+; All features marked P0; Removed sprint-based roadmap in favor of ralph-loop methodology; Enhanced quality gates (100% coverage, comprehensive static analysis); Added upstream contribution strategy; Translated to English |
| 1.2 | 2026-02-17 | DaprKit spec review compliance: standardized header format (Status/Priority/Depends blockquotes); added API Surface section with YAML fixture and full usage examples; added Async/Threading Strategy section referencing PRD-daprkit.md 5.6; defined exception hierarchy with base class and `from e` chaining; added `frozen=True` and `Field()` descriptions to all Pydantic models; replaced `dict[str, Any]` with typed `LlmProviderConfig` and `ToolDefinition` models; added consolidated Dependencies section with optional groups; added dedicated Test Strategy section with naming conventions and key scenarios; fixed logging examples to use standard `logging.getLogger(__name__)` interface; documented tooling deviations from DaprKit standard; added optional DaprKit integration note; renumbered sections for new API Surface (6) and Test Strategy (5.8) |
| 1.3 | 2026-02-17 | Added Environment Variable Resolution subsection documenting LLM credential resolution via provider-specific env vars; removed Related PRDs header reference and Optional DaprKit Integration subsection (standalone project with no daprkit dependency); inlined async/threading concerns previously referenced from external PRD |
| 1.4 | 2026-02-17 | Spec review fixes: added `field_path` attribute to `ValidationError` to match Public API Summary description; added `pylint >= 3.0` to Development Dependencies table |
