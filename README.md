# Dapr Agents OAS Adapter

Biblioteca adapter que permite interoperabilidade bidirecional entre [Open Agent Spec (OAS)](https://oracle.github.io/agent-spec/) e [Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/).

## Características

- **Importação OAS → Dapr Agents**: Carrega especificações OAS (JSON/YAML) e cria agentes e workflows Dapr Agents executáveis
- **Exportação Dapr Agents → OAS**: Exporta agentes e workflows Dapr para formato OAS
- **Conversores de Componentes**: Suporte para Agent, Flow, LlmConfig, Tool, Node e Edge
- **Geração de Código**: Gera código Python para workflows Dapr a partir de especificações OAS
- **Compatibilidade**: Segue os padrões dos adapters existentes (LangGraph, CrewAI)

## Instalação

```bash
pip install dapr-agents-oas-adapter
```

Ou com dependências de desenvolvimento:

```bash
pip install dapr-agents-oas-adapter[dev]
```

## Uso Rápido

### Carregar uma especificação OAS

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader

# Registrar implementações de tools
def search_tool(query: str) -> list[str]:
    """Busca na web."""
    return ["resultado1", "resultado2"]

loader = DaprAgentSpecLoader(
    tool_registry={"search": search_tool}
)

# Carregar de JSON
with open("agent_spec.json") as f:
    config = loader.load_json(f.read())

# Criar agente Dapr executável
agent = loader.create_agent(config)

# Iniciar o agente
await agent.start()
```

### Exportar para OAS

```python
from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import DaprAgentConfig

exporter = DaprAgentSpecExporter()

# Configurar agente
config = DaprAgentConfig(
    name="meu_assistente",
    role="Assistente",
    goal="Ajudar usuários",
    instructions=["Seja útil", "Seja conciso"],
    tools=["search", "calculator"],
)

# Exportar para JSON
json_spec = exporter.to_json(config)

# Exportar para arquivo YAML
exporter.to_yaml_file(config, "agent_spec.yaml")
```

### Trabalhar com Workflows

```python
from dapr_agents_oas_adapter import DaprAgentSpecLoader, DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowTaskDefinition,
    WorkflowEdgeDefinition,
)

# Criar definição de workflow
workflow = WorkflowDefinition(
    name="meu_workflow",
    description="Processa dados do usuário",
    tasks=[
        WorkflowTaskDefinition(name="start", task_type="start"),
        WorkflowTaskDefinition(
            name="analyze",
            task_type="llm",
            config={"prompt_template": "Analise: {{input}}"},
        ),
        WorkflowTaskDefinition(name="end", task_type="end"),
    ],
    edges=[
        WorkflowEdgeDefinition(from_node="start", to_node="analyze"),
        WorkflowEdgeDefinition(from_node="analyze", to_node="end"),
    ],
    start_node="start",
    end_nodes=["end"],
)

# Exportar para OAS
exporter = DaprAgentSpecExporter()
oas_spec = exporter.to_json(workflow)

# Gerar código Python do workflow
loader = DaprAgentSpecLoader()
code = loader.generate_workflow_code(workflow)
print(code)
```

## Mapeamento de Componentes

| OAS Component | Dapr Agents Equivalent |
|---------------|------------------------|
| `Agent` | `AssistantAgent` / `ReActAgent` |
| `Flow` | `@workflow` decorated function |
| `LlmNode` | `@task` com chamada LLM |
| `ToolNode` | `@task` com tool call |
| `StartNode` | Entry point do workflow |
| `EndNode` | Return do workflow |
| `ServerTool` | `@tool` decorated function |
| `MCPTool` | MCP integration via Dapr |
| `LlmConfig` | `DaprChatClient` config |
| `ControlFlowEdge` | Sequência de `yield ctx.call_activity()` |
| `DataFlowEdge` | Passagem de parâmetros entre tasks |

## API Reference

### DaprAgentSpecLoader

```python
class DaprAgentSpecLoader:
    def __init__(self, tool_registry: dict[str, Callable] | None = None)
    def load_json(self, json_content: str) -> DaprAgentConfig | WorkflowDefinition
    def load_yaml(self, yaml_content: str) -> DaprAgentConfig | WorkflowDefinition
    def load_json_file(self, file_path: str | Path) -> DaprAgentConfig | WorkflowDefinition
    def load_yaml_file(self, file_path: str | Path) -> DaprAgentConfig | WorkflowDefinition
    def load_component(self, component: Component) -> DaprAgentConfig | WorkflowDefinition
    def load_dict(self, spec_dict: dict) -> DaprAgentConfig | WorkflowDefinition
    def create_agent(self, config: DaprAgentConfig, additional_tools: dict | None = None) -> Any
    def create_workflow(self, workflow_def: WorkflowDefinition, task_implementations: dict | None = None) -> Callable
    def generate_workflow_code(self, workflow_def: WorkflowDefinition) -> str
    def register_tool(self, name: str, implementation: Callable) -> None
```

### DaprAgentSpecExporter

```python
class DaprAgentSpecExporter:
    def to_json(self, component: DaprAgentConfig | WorkflowDefinition, indent: int = 2) -> str
    def to_yaml(self, component: DaprAgentConfig | WorkflowDefinition) -> str
    def to_dict(self, component: DaprAgentConfig | WorkflowDefinition) -> dict
    def to_component(self, component: DaprAgentConfig | WorkflowDefinition) -> Component
    def to_json_file(self, component: DaprAgentConfig | WorkflowDefinition, file_path: str | Path) -> None
    def to_yaml_file(self, component: DaprAgentConfig | WorkflowDefinition, file_path: str | Path) -> None
    def from_dapr_agent(self, agent: Any) -> DaprAgentConfig
    def from_dapr_workflow(self, workflow_func: Callable, task_funcs: list[Callable] | None = None) -> WorkflowDefinition
    def export_agent_to_json(self, agent: Any) -> str
    def export_agent_to_yaml(self, agent: Any) -> str
```

## Desenvolvimento

### Setup

```bash
git clone https://github.com/seu-usuario/dapr-agents-oas-adapter.git
cd dapr-agents-oas-adapter
uv sync --all-extras
```

### Executar Testes

```bash
pytest
```

### Linting e Type Checking

```bash
ruff check src/
mypy src/
```

## Requisitos

- Python >= 3.12
- pyagentspec >= 25.4.1
- dapr-agents >= 0.10.4
- dapr >= 1.16.0
- pydantic >= 2.0.0

## Licença

MIT License

## Links

- [Open Agent Spec](https://oracle.github.io/agent-spec/)
- [Dapr Agents](https://docs.dapr.io/developing-applications/dapr-agents/)
- [PyAgentSpec](https://github.com/oracle/agent-spec)
- [Dapr](https://dapr.io/)

