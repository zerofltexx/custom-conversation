"""Fixtures for Custom Conversation E2E tests."""
import dataclasses
import os
from pathlib import Path

from dotenv import load_dotenv
import pytest
from pytest_socket import enable_socket, socket_allow_hosts

from custom_components.custom_conversation.const import (
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_LLM_PARAMETERS_SECTION,
    RECOMMENDED_BASE_URL,
    RECOMMENDED_CHAT_MODEL,
)
from homeassistant.const import CONF_API_KEY

env_path = Path(__file__).parent / ".env"
if env_path.is_file():
    load_dotenv(dotenv_path=env_path, verbose=True)

ALLOWED_HOSTS = ["api.openai.com", "generativelanguage.googleapis.com"]

@pytest.hookimpl(trylast=True)
def pytest_runtest_setup():
    """Enable socket access and allow specific hosts for E2E tests."""
    enable_socket()
    socket_allow_hosts([*ALLOWED_HOSTS, "localhost"], allow_unix_socket=True)


# --- LLM Provider Configuration ---

@dataclasses.dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""

    id: str
    model_env_var: str
    api_key_env_var: str # API key is required for OpenAI and Gemini
    base_url_env_var: str | None = None # Base URL is optional for standard OpenAI
    default_model: str = RECOMMENDED_CHAT_MODEL
    default_base_url: str | None = None # Default base URL varies per provider

    # --- Methods to get values with fallbacks ---
    def get_api_key(self) -> str:
        """Get API key, skipping test if not found."""
        key = os.getenv(self.api_key_env_var)
        if not key:
             pytest.skip(f"{self.api_key_env_var} environment variable not set. Skipping {self.id} E2E tests.")
        return key

    def get_model(self) -> str:
        """Get model name from env or use default."""
        return os.getenv(self.model_env_var, self.default_model)

    def get_base_url(self) -> str | None:
        """Get base URL from env or use default."""
        # Return env var value if set, otherwise the default for this provider
        # If base_url_env_var is None, it means base URL isn't typically configured via env for this provider (like standard OpenAI)
        # but we still check the default_base_url which might be set (like for Gemini)
        if self.base_url_env_var:
            return os.getenv(self.base_url_env_var, self.default_base_url)
        return self.default_base_url # Return the provider's default if no env var name specified

    def get_mock_config_entry_data(self) -> dict:
        """Generate the data dict for MockConfigEntry (API Key, Base URL)."""
        data = {
            # Use the component's const key name for API key
            CONF_API_KEY: self.get_api_key()
        }
        base_url = self.get_base_url()
        if base_url:
             # Use the component's const key name for Base URL
            data[CONF_BASE_URL] = base_url
        return data

    def get_mock_config_entry_options(self) -> dict:
        """Generate the options dict for MockConfigEntry (Model, etc.)."""
        return {
            CONF_LLM_PARAMETERS_SECTION:{
                # Use the component's const key name for Model
                CONF_CHAT_MODEL: self.get_model(),
            }
            # Add other common options here if needed (e.g., temperature, max_tokens)
            # Ensure these match how the component expects them in the options dict
        }


# Define the providers to test
SUPPORTED_PROVIDERS = [
    LLMProviderConfig(
        id="openai",
        api_key_env_var="OPENAI_API_KEY",
        model_env_var="OPENAI_MODEL",
        base_url_env_var="OPENAI_BASE_URL", # Optional override
        default_base_url=RECOMMENDED_BASE_URL, # Standard OpenAI endpoint
        default_model=RECOMMENDED_CHAT_MODEL,
    ),
    LLMProviderConfig(
        id="google_gemini_openai_compat",
        api_key_env_var="GEMINI_API_KEY", # Expecting a Gemini key
        model_env_var="GEMINI_MODEL", # Expecting a Gemini model name (e.g., models/gemini-1.5-flash-latest)
        base_url_env_var="GEMINI_OPENAI_BASE_URL", # Expecting the OpenAI-compatible endpoint URL
        default_base_url=None, # No standard default, must be provided via env
        default_model="gemini-1.5-flash-latest", # Example default Gemini model
    ),
]

# Parametrized fixture to provide each LLM config
@pytest.fixture(params=SUPPORTED_PROVIDERS, ids=[p.id for p in SUPPORTED_PROVIDERS])
def llm_config(request) -> LLMProviderConfig:
    """Fixture to provide LLM provider configuration, skipping if keys/URLs are missing."""
    config: LLMProviderConfig = request.param
    # Trigger the skip logic within the fixture by accessing mandatory fields
    config.get_api_key()
    # For Gemini compat, base URL is essential
    if config.id == "google_gemini_openai_compat" and not config.get_base_url():
         pytest.skip(f"{config.base_url_env_var} environment variable not set. Skipping {config.id} E2E tests.")

    return config

# Example fixture to check if OpenAI API key is loaded (can be expanded later)
@pytest.fixture(scope="session")
def openai_api_key() -> str | None:
    """Fixture to provide the OpenAI API key from environment variables."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY environment variable not set. Skipping OpenAI E2E tests.")
    return key

# Auto-enable custom integrations fixture
# This ensures that the custom_components directory is properly loaded
# when running tests from the e2e_tests directory.
@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations for E2E testing."""
    yield
