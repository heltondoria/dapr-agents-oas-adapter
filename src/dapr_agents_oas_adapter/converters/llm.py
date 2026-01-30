"""LLM configuration converter for OAS <-> Dapr Agents."""

from typing import Any, ClassVar

from pyagentspec.llms import LlmConfig, OllamaConfig, OpenAiConfig, VllmConfig

from dapr_agents_oas_adapter.converters.base import (
    ComponentConverter,
    ConversionError,
)
from dapr_agents_oas_adapter.types import (
    DAPR_PROVIDER_TO_OAS_LLM,
    OAS_LLM_TO_DAPR_PROVIDER,
    LlmClientConfig,
    ToolRegistry,
)
from dapr_agents_oas_adapter.utils import generate_id


class LlmConfigConverter(ComponentConverter[LlmConfig, LlmClientConfig]):
    """Converter for OAS LlmConfig <-> Dapr LLM client configuration.

    Supports conversion between various OAS LLM configurations
    (VllmConfig, OpenAiConfig, OllamaConfig) and Dapr's LlmClientConfig.
    """

    # Mapping of OAS LLM config types to their classes
    OAS_LLM_TYPES: ClassVar[dict[str, type[LlmConfig]]] = {
        "VllmConfig": VllmConfig,
        "OpenAIConfig": OpenAiConfig,
        "OllamaConfig": OllamaConfig,
    }

    def __init__(self, tool_registry: ToolRegistry | None = None) -> None:
        """Initialize the converter."""
        super().__init__(tool_registry)

    def from_oas(self, component: LlmConfig) -> LlmClientConfig:
        """Convert an OAS LlmConfig to Dapr LlmClientConfig.

        Args:
            component: The OAS LlmConfig to convert

        Returns:
            Dapr LlmClientConfig with equivalent settings

        Raises:
            ConversionError: If the LLM config type is not supported
        """
        self.validate_oas_component(component)

        component_type = type(component).__name__
        provider = OAS_LLM_TO_DAPR_PROVIDER.get(component_type)

        if provider is None:
            supported = ", ".join(OAS_LLM_TO_DAPR_PROVIDER.keys())
            raise ConversionError(
                f"Unsupported LLM config type: {component_type}",
                component,
                suggestion=f"Supported types are: {supported}",
            )

        # Extract common fields
        model_id = getattr(component, "model_id", "")
        url = getattr(component, "url", None)

        # Extract generation parameters
        extra_params: dict[str, Any] = {}
        default_gen_params = getattr(component, "default_generation_parameters", None)
        if default_gen_params:
            if isinstance(default_gen_params, dict):
                extra_params = default_gen_params.copy()
            else:
                # Handle pydantic model or dataclass
                if hasattr(default_gen_params, "__iter__"):
                    extra_params = dict(default_gen_params)
                else:
                    extra_params = {}

        # Extract temperature and max_tokens from extra params if present
        temperature = extra_params.pop("temperature", 0.7)
        max_tokens = extra_params.pop("max_tokens", None)

        # Handle OpenAI-specific fields
        api_key = None
        if component_type == "OpenAIConfig":
            api_key = getattr(component, "api_key", None)

        return LlmClientConfig(
            provider=provider,
            model_id=model_id,
            url=url,
            api_key=api_key,
            temperature=float(temperature) if temperature else 0.7,
            max_tokens=int(max_tokens) if max_tokens else None,
            extra_params=extra_params,
        )

    def to_oas(self, component: LlmClientConfig) -> VllmConfig | OpenAiConfig | OllamaConfig:
        """Convert a Dapr LlmClientConfig to OAS LlmConfig.

        Args:
            component: The Dapr LlmClientConfig to convert

        Returns:
            OAS LlmConfig with equivalent settings

        Raises:
            ConversionError: If the provider is not supported
        """
        oas_type = DAPR_PROVIDER_TO_OAS_LLM.get(component.provider)

        if oas_type is None:
            supported = ", ".join(DAPR_PROVIDER_TO_OAS_LLM.keys())
            raise ConversionError(
                f"Unsupported Dapr LLM provider: {component.provider}",
                component,
                suggestion=f"Supported providers are: {supported}",
            )

        # Build generation parameters
        gen_params: dict[str, Any] = {}
        if component.temperature != 0.7:
            gen_params["temperature"] = component.temperature
        if component.max_tokens:
            gen_params["max_tokens"] = component.max_tokens
        gen_params.update(component.extra_params)

        # Build the OAS LlmConfig
        config_id = generate_id("llm")
        name = f"{component.provider}_{component.model_id}"

        if oas_type == "VllmConfig":
            return VllmConfig(
                id=config_id,
                name=name,
                model_id=component.model_id,
                url=component.url or "",
            )
        elif oas_type == "OpenAIConfig":
            return OpenAiConfig(
                id=config_id,
                name=name,
                model_id=component.model_id,
            )
        elif oas_type == "OllamaConfig":
            return OllamaConfig(
                id=config_id,
                name=name,
                model_id=component.model_id,
                url=component.url or "http://localhost:11434",
            )
        else:
            raise ConversionError(
                f"Unhandled OAS type: {oas_type}",
                component,
                suggestion="This is an internal error, please report it",
            )

    def can_convert(self, component: Any) -> bool:
        """Check if this converter can handle the given component.

        Args:
            component: The component to check

        Returns:
            True if this converter can handle the component
        """
        if isinstance(component, LlmConfig):
            return True
        if isinstance(component, LlmClientConfig):
            return True
        # Check for dict representation
        if isinstance(component, dict):
            comp_type = component.get("component_type", "")
            return comp_type in self.OAS_LLM_TYPES
        return False

    def from_dict(self, config_dict: dict[str, Any]) -> LlmClientConfig:
        """Convert a dictionary representation to LlmClientConfig.

        Args:
            config_dict: Dictionary with LLM configuration

        Returns:
            LlmClientConfig with the converted settings
        """
        component_type = config_dict.get("component_type", "VllmConfig")
        provider = OAS_LLM_TO_DAPR_PROVIDER.get(component_type, "openai")

        return LlmClientConfig(
            provider=provider,
            model_id=config_dict.get("model_id", ""),
            url=config_dict.get("url"),
            api_key=config_dict.get("api_key"),
            temperature=config_dict.get("temperature", 0.7),
            max_tokens=config_dict.get("max_tokens"),
            extra_params=config_dict.get("default_generation_parameters", {}),
        )

    def to_dict(self, config: LlmClientConfig) -> dict[str, Any]:
        """Convert LlmClientConfig to a dictionary representation.

        Args:
            config: The LlmClientConfig to convert

        Returns:
            Dictionary representation of the config
        """
        oas_type = DAPR_PROVIDER_TO_OAS_LLM.get(config.provider, "VllmConfig")

        result: dict[str, Any] = {
            "component_type": oas_type,
            "id": generate_id("llm"),
            "name": f"{config.provider}_{config.model_id}",
            "model_id": config.model_id,
        }

        if config.url:
            result["url"] = config.url
        if config.api_key:
            result["api_key"] = config.api_key

        gen_params: dict[str, Any] = {}
        if config.temperature != 0.7:
            gen_params["temperature"] = config.temperature
        if config.max_tokens:
            gen_params["max_tokens"] = config.max_tokens
        gen_params.update(config.extra_params)

        if gen_params:
            result["default_generation_parameters"] = gen_params

        return result
