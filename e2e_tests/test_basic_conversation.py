"""E2E test for basic conversation flow across different LLM providers."""

import pytest
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    DOMAIN,
    CONVERSATION_STARTED_EVENT, # Import event constants
    CONVERSATION_ENDED_EVENT,
)
from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.setup import async_setup_component

# Import LLMProviderConfig type hint for the fixture
from .conftest import LLMProviderConfig

# Fixtures provided by e2e_tests/conftest.py:
# - hass: HomeAssistant instance
# - auto_enable_custom_integrations: Enables custom components
# - llm_config: Parametrized fixture providing LLMProviderConfig for each supported provider


async def test_basic_conversation(
    hass: HomeAssistant, llm_config: LLMProviderConfig
):
    """Test a basic conversation flow using the configured LLM provider."""

    # --- Configuration ---
    # Use the LLMProviderConfig fixture to get data and options
    config_data = llm_config.get_mock_config_entry_data()
    config_options = llm_config.get_mock_config_entry_options()

    # Add specific test options (agent config, disable HA API)
    test_specific_options = {
        CONF_LLM_HASS_API: None, # Ensure we don't use HA's internal API handling
        CONF_AGENTS_SECTION: {
            CONF_ENABLE_HASS_AGENT: False, # Disable HA agent
            CONF_ENABLE_LLM_AGENT: True,   # Enable LLM agent
        },
        # Add other necessary default options if the component requires them
    }
    # Merge provider options with test-specific options
    final_options = {**config_options, **test_specific_options}

    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title=f"{llm_config.id} E2E Test",
        data=config_data,
        options=final_options,
        entry_id=f"{llm_config.id}_e2e_test_entry", # Unique ID per provider
    )
    config_entry.add_to_hass(hass)

    # --- Setup Component ---
    # Ensure core components are loaded
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    # Setup our custom component with the specific config entry
    assert await async_setup_component(hass, DOMAIN, {})
    await hass.async_block_till_done() # Wait for setup to complete

    # --- Test Conversation ---
    test_prompt = "What is the capital of France?"
    conversation_id = f"test-{llm_config.id}-convo-1"

    # --- Setup Event Listeners ---
    started_events = []
    ended_events = []

    hass.bus.async_listen(CONVERSATION_STARTED_EVENT, lambda e: started_events.append(e))
    hass.bus.async_listen(CONVERSATION_ENDED_EVENT, lambda e: ended_events.append(e))

    # --- Run Conversation ---
    result = await conversation.async_converse(
        hass,
        text=test_prompt,
        conversation_id=conversation_id,
        context=Context(),
        agent_id=config_entry.entry_id,
    )

    # Ensure events are processed
    await hass.async_block_till_done()
    # Removed extraneous parenthesis from previous line

    # --- Assertions ---
    assert result is not None, f"[{llm_config.id}] Conversation result was None"
    assert result.conversation_id == conversation_id, f"[{llm_config.id}] Conversation ID mismatch"
    assert result.response is not None, f"[{llm_config.id}] Conversation response was None"

    assert result.response.error_code is None, f"[{llm_config.id}] Conversation failed: {result.response.error_code}"

    response_text = result.response.speech.get("plain", {}).get("speech", "")
    assert response_text is not None, f"[{llm_config.id}] Response text was None"
    assert isinstance(response_text, str), f"[{llm_config.id}] Response text is not a string"
    assert len(response_text) > 0, f"[{llm_config.id}] Response text was empty"
    assert "Paris" in response_text, f"[{llm_config.id}] Expected keyword 'Paris' not found in response: {response_text}"

    print(f"[{llm_config.id}] LLM Response: {response_text}") # Optional: print response

    # --- Assert Events ---
    assert len(started_events) == 1, f"[{llm_config.id}] Expected 1 conversation_started event, got {len(started_events)}"
    assert started_events[0].data["agent_id"] == config_entry.entry_id
    assert started_events[0].data["conversation_id"] == conversation_id
    assert started_events[0].data["text"] == test_prompt
    # No device was specified for this test
    assert started_events[0].data.get("device_id") is None
    assert started_events[0].data.get("device_name") == "Unknown"
    assert started_events[0].data.get("device_area") == "Unknown"
    # No user was specified for this test
    assert started_events[0].data.get("user_id") is None

    assert len(ended_events) == 1, f"[{llm_config.id}] Expected 1 conversation_ended event, got {len(ended_events)}"
    assert ended_events[0].data["agent_id"] == config_entry.entry_id
    assert ended_events[0].data["result"]["conversation_id"] == conversation_id
    assert ended_events[0].data["handling_agent"] == 'LLM'
    assert ended_events[0].data["request"] == test_prompt
    # Check if response data is included in the ended event (adjust key if necessary)
    assert ended_events[0].data.get("result") is not None
    assert "Paris" in ended_events[0].data["result"]["response"]["speech"]["plain"]["speech"]
    assert ended_events[0].data["result"]["response"]["response_type"] == "action_done"
    # No device was specified for this test
    assert ended_events[0].data.get("device_id") is None
    assert ended_events[0].data.get("device_name") == "Unknown"
    assert ended_events[0].data.get("device_area") == "Unknown"
    # No user was specified for this test
    assert ended_events[0].data.get("user_id") is None
