"""E2E tests for conversation flows involving tool calls."""

from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
    async_mock_service,
)

from custom_components.custom_conversation.const import (
    CONVERSATION_ENDED_EVENT,
    CONVERSATION_STARTED_EVENT,
)
from homeassistant.components import conversation, switch
from homeassistant.const import ATTR_ENTITY_ID, SERVICE_TURN_ON
from homeassistant.core import Context, HomeAssistant


async def test_tool_call_switch( # Add mock_test_switch fixture dependency
    hass: HomeAssistant, setup_config_entry: MockConfigEntry, mock_test_switch: None
):
    """Test a conversation flow that should result in a switch.turn_on tool call."""
    config_entry = setup_config_entry
    provider_id = config_entry.entry_id.replace("_e2e_test_entry", "")

    turn_on_calls = async_mock_service(hass, switch.DOMAIN, SERVICE_TURN_ON)

    test_prompt = "Turn on the test switch"
    conversation_id = f"test-{provider_id}-tool-convo-1"
    if len(conversation_id) == 26:
        conversation_id = f"test-{provider_id}-tool-convo-01"

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
    await hass.async_block_till_done()

    assert result is not None, f"[{provider_id}] Tool call result was None"
    assert result.response is not None, f"[{provider_id}] Tool call response was None"
    assert result.response.error_code is None, f"[{provider_id}] Tool call failed: {result.response.error_code}"

    assert len(turn_on_calls) == 1, f"[{provider_id}] Expected 1 switch.turn_on call, got {len(turn_on_calls)}"
    assert turn_on_calls[0].data[ATTR_ENTITY_ID] == ["switch.test_switch"], f"[{provider_id}] Service called with wrong entity_id"

    response_text = result.response.speech.get("plain", {}).get("speech", "")
    assert response_text is not None, f"[{provider_id}] Tool call response text was None"
    assert "test switch" in response_text.lower() or "turned on" in response_text.lower(), \
        f"[{provider_id}] Response text doesn't seem to confirm action: {response_text}"

    assert len(started_events) == 1, f"[{provider_id}] Expected 1 conversation_started event, got {len(started_events)}"
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
    assert len(ended_events) == 1, f"[{provider_id}] Expected 1 conversation_ended event, got {len(ended_events)}"
    assert ended_events[0].data["agent_id"] == config_entry.entry_id
    assert ended_events[0].data["result"]["conversation_id"] == conversation_id
    assert ended_events[0].data["handling_agent"] == 'LLM'
    assert ended_events[0].data["request"] == test_prompt
    # Assert success targets
    assert len(ended_events[0].data["result"]["response"]["data"]["success"]) == 1
    assert ended_events[0].data["result"]["response"]["data"]["success"][0]["id"] == "switch.test_switch"
    assert ended_events[0].data["result"]["response"]["data"]["success"][0]["type"] == "entity"
    assert ended_events[0].data["result"]["response"]["data"]["success"][0]["name"] == "Test Switch"

