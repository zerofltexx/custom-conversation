"""Configuration for supported LiteLLM providers."""
from dataclasses import dataclass


@dataclass
class LiteLLMProvider:
    """Configuration for supported LiteLLM providers."""

    key: str
    provider_name: str
    supports_custom_base_url: bool = False

openai = LiteLLMProvider(
    key="openai",
    provider_name="OpenAI",
    supports_custom_base_url=True,
)

gemini = LiteLLMProvider(
    key="gemini",
    provider_name="Gemini - Google AI Studio",
    # See: https://github.com/BerriAI/litellm/issues/7830
    #supports_custom_base_url=True,
)

ollama = LiteLLMProvider(
    key="ollama",
    provider_name="Ollama",
    supports_custom_base_url=True,
)

ollama_chat = LiteLLMProvider(
    key="ollama_chat",
    provider_name="Ollama Chat",
    supports_custom_base_url=True,
)

SUPPORTED_PROVIDERS = [
    openai,
    gemini,
    # ollama, disabled pending litellm fixes for https://github.com/BerriAI/litellm/issues/6135 and https://github.com/BerriAI/litellm/issues/9602
    # ollama_chat
]

def get_provider(provider_key: str) -> LiteLLMProvider | None:
    """Get the provider by key."""
    for provider in SUPPORTED_PROVIDERS:
        if provider.key == provider_key:
            return provider
    return None
