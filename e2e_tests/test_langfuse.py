"""E2E tests for Langfuse integration (tracing and prompt management)."""

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


async def test_langfuse_tool_call(
    hass: HomeAssistant,
    langfuse_enabled_entry: MockConfigEntry,
    mock_test_switch: None,
):
    """Test a tool call conversation flow with Langfuse enabled.

    Verifies that the flow completes and the tool call is made.
    Implicitly tests that Langfuse integration doesn't break the core logic.
    """
    provider_id = langfuse_enabled_entry.entry_id.replace("_e2e_test_entry", "")

    # --- Setup Service Mock ---
    # Mock the switch.turn_on service call to verify it's called correctly
    turn_on_calls = async_mock_service(hass, switch.DOMAIN, SERVICE_TURN_ON)

    # --- Event Listeners (Optional but recommended) ---
    started_events = []
    ended_events = []
    hass.bus.async_listen(CONVERSATION_STARTED_EVENT, lambda e: started_events.append(e))
    hass.bus.async_listen(CONVERSATION_ENDED_EVENT, lambda e: ended_events.append(e))

    # --- Test Conversation ---
    test_prompt = "Turn on the test switch"
    conversation_id = f"test-{provider_id}-langfuse-tool-convo-1"

    result = await conversation.async_converse(
        hass,
        text=test_prompt,
        conversation_id=conversation_id,
        context=Context(),
        agent_id=langfuse_enabled_entry.entry_id,
    )
    await hass.async_block_till_done()

    # --- Assertions ---
    assert result is not None, f"[{provider_id}] Langfuse tool call result was None"
    assert result.response is not None, f"[{provider_id}] Langfuse tool call response was None"
    assert result.response.error_code is None, f"[{provider_id}] Langfuse tool call failed: {result.response.error_code}"

    # Assert service call
    assert len(turn_on_calls) == 1, f"[{provider_id}] Expected 1 switch.turn_on call with Langfuse, got {len(turn_on_calls)}"
    assert turn_on_calls[0].data[ATTR_ENTITY_ID] == ["switch.test_switch"], f"[{provider_id}] Langfuse service called with wrong entity_id"

    # Basic check on response text
    response_text = result.response.speech.get("plain", {}).get("speech", "")
    assert response_text is not None, f"[{provider_id}] Langfuse tool call response text was None"
