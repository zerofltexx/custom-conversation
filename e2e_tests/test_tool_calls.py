"""E2E tests for conversation flows involving tool calls."""

import pytest
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONVERSATION_ENDED_EVENT,
    CONVERSATION_STARTED_EVENT,
    DOMAIN,
    LLM_API_ID,
)
from homeassistant.components import conversation, switch
from homeassistant.const import ATTR_ENTITY_ID, CONF_LLM_HASS_API, SERVICE_TURN_ON
from homeassistant.core import Context, HomeAssistant
from homeassistant.setup import async_setup_component

# Import LLMProviderConfig type hint for the fixture
from .conftest import LLMProviderConfig

# Fixtures provided by e2e_tests/conftest.py:
# - hass: HomeAssistant instance
# - auto_enable_custom_integrations: Enables custom components
# - llm_config: Parametrized fixture providing LLMProviderConfig


async def test_tool_call_switch(
    hass: HomeAssistant, llm_config: LLMProviderConfig
):
    """Test a conversation flow that should result in a switch.turn_on tool call."""

    # --- Configuration ---
    config_data = llm_config.get_mock_config_entry_data()
    config_options = llm_config.get_mock_config_entry_options()

    # Configure to use the custom LLM API for tool handling
    test_specific_options = {
        CONF_LLM_HASS_API: LLM_API_ID, # Use the integration's API for tool calls
        CONF_AGENTS_SECTION: {
            CONF_ENABLE_HASS_AGENT: False, # Disable HA agent to force LLM path
            CONF_ENABLE_LLM_AGENT: True,   # Enable LLM agent
        },
    }
    final_options = {**config_options, **test_specific_options}

    config_entry = MockConfigEntry(
        domain=DOMAIN,
        title=f"{llm_config.id} E2E Tool Test",
        data=config_data,
        options=final_options,
        entry_id=f"{llm_config.id}_e2e_tool_test_entry",
    )
    config_entry.add_to_hass(hass)

    # --- Setup Mock Entity ---
    # Create a mock switch entity for the LLM to interact with
    hass.states.async_set("switch.test_switch", "off", {"friendly_name": "Test Switch"})

    # --- Setup Component ---
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})
    # Ensure the switch domain is loaded so the service call can be mocked/handled
    assert await async_setup_component(hass, switch.DOMAIN, {})
    assert await async_setup_component(hass, DOMAIN, {})
    await hass.async_block_till_done()

    # --- Setup Service Mock ---
    # Mock the switch.turn_on service call to verify it's called correctly
    turn_on_calls = async_mock_service(hass, switch.DOMAIN, SERVICE_TURN_ON)

    # --- Test Conversation ---
    test_prompt = "Turn on the test switch"
    conversation_id = f"test-{llm_config.id}-tool-convo-1"

 # --- Event Listeners ---
    started_events = []
    ended_events = []

    hass.bus.async_listen(CONVERSATION_STARTED_EVENT, lambda e: started_events.append(e))
    hass.bus.async_listen(CONVERSATION_ENDED_EVENT, lambda e: ended_events.append(e))

    result = await conversation.async_converse(
        hass,
        text=test_prompt,
        conversation_id=conversation_id,
        context=Context(),
        agent_id=config_entry.entry_id,
    )
    await hass.async_block_till_done() # Ensure service calls are processed


    # --- Assertions ---
    assert result is not None, f"[{llm_config.id}] Tool call result was None"
    assert result.response is not None, f"[{llm_config.id}] Tool call response was None"
    assert result.response.error_code is None, f"[{llm_config.id}] Tool call failed: {result.response.error_code}"

    # Assert service call
    assert len(turn_on_calls) == 1, f"[{llm_config.id}] Expected 1 switch.turn_on call, got {len(turn_on_calls)}"
    assert turn_on_calls[0].data[ATTR_ENTITY_ID] == ["switch.test_switch"], f"[{llm_config.id}] Service called with wrong entity_id"

    # Basic check on response text (might vary significantly based on LLM)
    response_text = result.response.speech.get("plain", {}).get("speech", "")
    assert response_text is not None, f"[{llm_config.id}] Tool call response text was None"
    # Example: Check if it mentions turning something on or the switch name
    assert "test switch" in response_text.lower() or "turned on" in response_text.lower(), \
        f"[{llm_config.id}] Response text doesn't seem to confirm action: {response_text}"

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
    # Assert the conversation ended event
    assert len(ended_events) == 1, f"[{llm_config.id}] Expected 1 conversation_ended event, got {len(ended_events)}"
    assert ended_events[0].data["agent_id"] == config_entry.entry_id
    assert ended_events[0].data["result"]["conversation_id"] == conversation_id
    assert ended_events[0].data["handling_agent"] == 'LLM'
    assert ended_events[0].data["request"] == test_prompt
    # Assert success targets
    assert len(ended_events[0].data["result"]["response"]["data"]["success"]) == 1
    assert ended_events[0].data["result"]["response"]["data"]["success"][0]["id"] == "switch.test_switch"
    assert ended_events[0].data["result"]["response"]["data"]["success"][0]["type"] == "entity"
    assert ended_events[0].data["result"]["response"]["data"]["success"][0]["name"] == "Test Switch"

