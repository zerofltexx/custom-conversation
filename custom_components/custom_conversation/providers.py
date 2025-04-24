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

SUPPORTED_PROVIDERS = [openai, gemini]

def get_provider(provider_key: str) -> LiteLLMProvider | None:
    """Get the provider by key."""
    for provider in SUPPORTED_PROVIDERS:
        if provider.key == provider_key:
            return provider
    return None
