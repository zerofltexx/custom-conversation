"""Tests for the Custom Conversation prompt manager."""
from unittest.mock import Mock, patch

import pytest

from homeassistant.helpers import entity_registry as er

from custom_components.custom_conversation.prompt_manager import (
    PromptContext, PromptManager
)
from custom_components.custom_conversation.const import (
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_PROMPT_BASE,
    CONF_API_PROMPT_BASE,
    DEFAULT_BASE_PROMPT,
    DEFAULT_INSTRUCTIONS_PROMPT,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
)


@pytest.fixture
def prompt_manager(hass):
    """Create a PromptManager instance."""
    return PromptManager(hass)


async def test_get_base_prompt_default(prompt_manager, hass):
    """Test getting base prompt with defaults."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        user_name="Test User",
    )
    
    prompt = await prompt_manager.async_get_base_prompt(context)
    
    assert "Current time is" in prompt
    assert DEFAULT_INSTRUCTIONS_PROMPT.strip() in prompt


async def test_get_base_prompt_custom(prompt_manager, hass, config_entry):
    """Test getting base prompt with custom configuration."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        user_name="Test User",
    )
    
    prompt = await prompt_manager.async_get_base_prompt(context, config_entry)
    
    assert "Custom base prompt for Test Home" in prompt
    assert "Custom instructions for Test User" in prompt


async def test_get_api_prompt_no_entities(prompt_manager, hass):
    """Test API prompt when no entities are exposed."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        exposed_entities=None,
    )
    
    prompt = await prompt_manager.get_api_prompt(context)
    
    assert prompt == DEFAULT_PROMPT_NO_ENABLED_ENTITIES


async def test_get_api_prompt_with_location(prompt_manager, hass, config_entry):
    """Test API prompt with location information."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        location="Living Room",
        exposed_entities={"light.test": {"name": "Test Light"}},
    )
    
    prompt = await prompt_manager.get_api_prompt(context, config_entry)
    
    assert "Custom API base prompt" in prompt
    assert "Custom location prompt for Living Room" in prompt
    assert "Test Light" in prompt


async def test_get_api_prompt_no_timers(prompt_manager, hass):
    """Test API prompt when timers are not supported."""
    context = PromptContext(
        hass=hass,
        ha_name="Test Home",
        exposed_entities={"light.test": {"name": "Test Light"}},
        supports_timers=False,
    )
    
    prompt = await prompt_manager.get_api_prompt(context)
    
    assert "This device is not able to start timers" in prompt



def test_get_prompt_config_no_config_entry(prompt_manager):
    """Test getting prompt config with no config entry."""
    result = prompt_manager._get_prompt_config(None, "test_key", "default_value")
    
    assert result == "default_value"


def test_get_prompt_config_with_config_entry(prompt_manager, config_entry):
    """Test getting prompt config with config entry."""
    result = prompt_manager._get_prompt_config(
        config_entry,
        CONF_API_PROMPT_BASE,
        DEFAULT_API_PROMPT_BASE
    )
    
    assert result == "Custom API base prompt"