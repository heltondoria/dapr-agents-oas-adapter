"""OAS specification fixtures for integration tests.

These fixtures represent realistic OAS configurations for testing
the full conversion pipeline.
"""

# Simple Agent specification - uses VllmConfig which is supported by pyagentspec
SIMPLE_AGENT_YAML = """
component_type: Agent
id: agent_simple
name: simple_assistant
description: A simple assistant agent for testing
system_prompt: You are a helpful assistant.
llm_config:
  component_type: VllmConfig
  id: llm_vllm
  name: vllm_config
  model_id: gpt-4
  url: http://localhost:8000
tools: []
inputs: []
outputs: []
agentspec_version: "25.4.1"
"""

SIMPLE_AGENT_JSON = """{
  "component_type": "Agent",
  "id": "agent_simple",
  "name": "simple_assistant",
  "description": "A simple assistant agent for testing",
  "system_prompt": "You are a helpful assistant.",
  "llm_config": {
    "component_type": "VllmConfig",
    "id": "llm_vllm",
    "name": "vllm_config",
    "model_id": "gpt-4",
    "url": "http://localhost:8000"
  },
  "tools": [],
  "inputs": [],
  "outputs": [],
  "agentspec_version": "25.4.1"
}"""

# Agent with tools specification
AGENT_WITH_TOOLS_YAML = """
component_type: Agent
id: agent_tools
name: research_assistant
description: Research assistant with search capabilities
system_prompt: You are a research assistant with tools.
llm_config:
  component_type: OllamaConfig
  id: llm_ollama
  name: llama_config
  model_id: llama3
  host: localhost
  port: 11434
tools:
  - component_type: ServerTool
    id: tool_search
    name: web_search
    description: Search the web for information
    parameters:
      type: object
      properties:
        query:
          type: string
          description: Search query
      required:
        - query
  - component_type: ServerTool
    id: tool_calculator
    name: calculator
    description: Perform mathematical calculations
    parameters:
      type: object
      properties:
        expression:
          type: string
          description: Mathematical expression
      required:
        - expression
inputs: []
outputs: []
agentspec_version: "25.4.1"
"""

# Simple Flow specification (start -> llm -> end)
SIMPLE_FLOW_YAML = """
component_type: Flow
id: flow_simple
name: simple_workflow
description: A simple workflow with one LLM task
nodes:
  - component_type: StartNode
    id: start
    name: start
    inputs: []
    outputs: []
  - component_type: LlmNode
    id: process
    name: process_input
    inputs:
      - name: input_text
        type: string
    outputs:
      - name: output_text
        type: string
    llm_config:
      component_type: OpenAIConfig
      id: llm_1
      name: gpt4
      model_id: gpt-4
    prompt_template: "Process this: {{ input_text }}"
  - component_type: EndNode
    id: end
    name: end
    inputs: []
    outputs: []
control_flow_connections:
  - id: edge_1
    name: start_to_process
    from_node:
      $component_ref: start
    to_node:
      $component_ref: process
  - id: edge_2
    name: process_to_end
    from_node:
      $component_ref: process
    to_node:
      $component_ref: end
data_flow_connections: []
start_node:
  $component_ref: start
agentspec_version: "25.4.1"
"""

# Flow with branching specification
BRANCHING_FLOW_YAML = """
component_type: Flow
id: flow_branching
name: branching_workflow
description: A workflow with conditional branching
nodes:
  - component_type: StartNode
    id: start
    name: start
    inputs: []
    outputs: []
  - component_type: LlmNode
    id: classifier
    name: classifier
    inputs:
      - name: text
        type: string
    outputs:
      - name: category
        type: string
    llm_config:
      component_type: OpenAIConfig
      id: llm_classifier
      name: classifier_llm
      model_id: gpt-4
    prompt_template: "Classify this text: {{ text }}"
  - component_type: ToolNode
    id: urgent_handler
    name: handle_urgent
    inputs:
      - name: data
        type: object
    outputs:
      - name: result
        type: object
    tool:
      $component_ref: urgent_tool
  - component_type: ToolNode
    id: normal_handler
    name: handle_normal
    inputs:
      - name: data
        type: object
    outputs:
      - name: result
        type: object
    tool:
      $component_ref: normal_tool
  - component_type: EndNode
    id: end
    name: end
    inputs: []
    outputs: []
control_flow_connections:
  - id: edge_start
    name: start_to_classify
    from_node:
      $component_ref: start
    to_node:
      $component_ref: classifier
  - id: edge_urgent
    name: classify_to_urgent
    from_node:
      $component_ref: classifier
    to_node:
      $component_ref: urgent_handler
    from_branch: urgent
  - id: edge_normal
    name: classify_to_normal
    from_node:
      $component_ref: classifier
    to_node:
      $component_ref: normal_handler
    from_branch: normal
  - id: edge_urgent_end
    name: urgent_to_end
    from_node:
      $component_ref: urgent_handler
    to_node:
      $component_ref: end
  - id: edge_normal_end
    name: normal_to_end
    from_node:
      $component_ref: normal_handler
    to_node:
      $component_ref: end
data_flow_connections: []
start_node:
  $component_ref: start
agentspec_version: "25.4.1"
"""

# Flow with retry and timeout configuration
RETRY_TIMEOUT_FLOW_YAML = """
component_type: Flow
id: flow_retry
name: retry_workflow
description: A workflow with retry and timeout policies
metadata:
  dapr:
    retry_policy:
      max_attempts: 3
      initial_interval_seconds: 1
      backoff_coefficient: 2.0
      max_interval_seconds: 30
    timeout_seconds: 60
nodes:
  - component_type: StartNode
    id: start
    name: start
    inputs: []
    outputs: []
  - component_type: ToolNode
    id: api_call
    name: external_api
    inputs:
      - name: request
        type: object
    outputs:
      - name: response
        type: object
    tool:
      $component_ref: api_tool
    metadata:
      dapr:
        retry_policy:
          max_attempts: 5
          initial_interval_seconds: 2
        timeout_seconds: 30
  - component_type: EndNode
    id: end
    name: end
    inputs: []
    outputs: []
control_flow_connections:
  - id: edge_1
    name: start_to_api
    from_node:
      $component_ref: start
    to_node:
      $component_ref: api_call
  - id: edge_2
    name: api_to_end
    from_node:
      $component_ref: api_call
    to_node:
      $component_ref: end
data_flow_connections: []
start_node:
  $component_ref: start
agentspec_version: "25.4.1"
"""

# Flow with map/fan-out pattern
MAP_FLOW_YAML = """
component_type: Flow
id: flow_map
name: map_workflow
description: A workflow with map/fan-out pattern
nodes:
  - component_type: StartNode
    id: start
    name: start
    inputs: []
    outputs: []
  - component_type: MapNode
    id: parallel_process
    name: process_items
    inputs:
      - name: items
        type: array
    outputs:
      - name: results
        type: array
    inner_flow:
      $component_ref: item_processor
    parallel: true
    metadata:
      dapr:
        map_input_key: items
  - component_type: EndNode
    id: end
    name: end
    inputs: []
    outputs: []
control_flow_connections:
  - id: edge_1
    name: start_to_map
    from_node:
      $component_ref: start
    to_node:
      $component_ref: parallel_process
  - id: edge_2
    name: map_to_end
    from_node:
      $component_ref: parallel_process
    to_node:
      $component_ref: end
data_flow_connections: []
start_node:
  $component_ref: start
agentspec_version: "25.4.1"
"""

# Flow with subflow reference
SUBFLOW_YAML = """
component_type: Flow
id: flow_parent
name: parent_workflow
description: A workflow that calls a child workflow
nodes:
  - component_type: StartNode
    id: start
    name: start
    inputs: []
    outputs: []
  - component_type: FlowNode
    id: child_task
    name: child_processor
    inputs:
      - name: data
        type: object
    outputs:
      - name: result
        type: object
    subflow:
      $component_ref: child_workflow
  - component_type: EndNode
    id: end
    name: end
    inputs: []
    outputs: []
control_flow_connections:
  - id: edge_1
    name: start_to_child
    from_node:
      $component_ref: start
    to_node:
      $component_ref: child_task
  - id: edge_2
    name: child_to_end
    from_node:
      $component_ref: child_task
    to_node:
      $component_ref: end
data_flow_connections: []
start_node:
  $component_ref: start
agentspec_version: "25.4.1"
"""

# Multi-step complex flow
COMPLEX_FLOW_YAML = """
component_type: Flow
id: flow_complex
name: complex_workflow
description: A complex multi-step workflow with various node types
nodes:
  - component_type: StartNode
    id: start
    name: start
    inputs: []
    outputs: []
  - component_type: LlmNode
    id: extract
    name: extract_info
    inputs:
      - name: raw_text
        type: string
    outputs:
      - name: extracted
        type: object
    llm_config:
      component_type: OpenAIConfig
      id: llm_extract
      name: extractor
      model_id: gpt-4
    prompt_template: "Extract key information from: {{ raw_text }}"
  - component_type: ToolNode
    id: validate
    name: validate_data
    inputs:
      - name: data
        type: object
    outputs:
      - name: validated
        type: object
    tool:
      $component_ref: validation_tool
  - component_type: LlmNode
    id: summarize
    name: summarize_results
    inputs:
      - name: validated_data
        type: object
    outputs:
      - name: summary
        type: string
    llm_config:
      component_type: OpenAIConfig
      id: llm_summarize
      name: summarizer
      model_id: gpt-4
    prompt_template: "Summarize: {{ validated_data }}"
  - component_type: EndNode
    id: end
    name: end
    inputs: []
    outputs: []
control_flow_connections:
  - id: edge_1
    from_node:
      $component_ref: start
    to_node:
      $component_ref: extract
  - id: edge_2
    from_node:
      $component_ref: extract
    to_node:
      $component_ref: validate
  - id: edge_3
    from_node:
      $component_ref: validate
    to_node:
      $component_ref: summarize
  - id: edge_4
    from_node:
      $component_ref: summarize
    to_node:
      $component_ref: end
data_flow_connections: []
start_node:
  $component_ref: start
agentspec_version: "25.4.1"
"""

# All LLM provider types
ALL_LLM_PROVIDERS_YAML = """
component_type: Agent
id: agent_multi_llm
name: multi_llm_agent
description: Agent demonstrating all LLM provider configurations
system_prompt: You are a test agent.
llm_config:
  component_type: VllmConfig
  id: llm_vllm
  name: vllm_provider
  model_id: meta-llama/Llama-2-7b
  url: http://localhost:8000
tools: []
inputs: []
outputs: []
agentspec_version: "25.4.1"
"""

# Agent with OCI LLM
OCI_LLM_AGENT_YAML = """
component_type: Agent
id: agent_oci
name: oci_agent
description: Agent with OCI Generative AI
system_prompt: You are an OCI-powered assistant.
llm_config:
  component_type: OCIConfig
  id: llm_oci
  name: oci_provider
  model_id: cohere.command-r-plus
  compartment_id: ocid1.compartment.oc1..test
  auth_profile: DEFAULT
tools: []
inputs: []
outputs: []
agentspec_version: "25.4.1"
"""

# Ollama LLM Agent
OLLAMA_AGENT_YAML = """
component_type: Agent
id: agent_ollama
name: ollama_agent
description: Agent with Ollama local LLM
system_prompt: You are a local Ollama assistant.
llm_config:
  component_type: OllamaConfig
  id: llm_ollama
  name: ollama_provider
  model_id: llama3
  host: localhost
  port: 11434
tools: []
inputs: []
outputs: []
agentspec_version: "25.4.1"
"""
