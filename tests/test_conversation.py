import pytest
from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant, Context
from homeassistant.helpers import intent
from homeassistant.setup import async_setup_component
from unittest.mock import Mock, patch
from custom_components.custom_conversation.conversation import CustomConversationEntity
from custom_components.custom_conversation.const import CONF_AGENTS_SECTION, CONF_ENABLE_HASS_AGENT, CONF_ENABLE_LLM_AGENT, LLM_API_ID
from custom_components.custom_conversation import CustomConversationConfigEntry


async def test_custom_conversation_entity_initialization(hass: HomeAssistant, config_entry: CustomConversationConfigEntry):
    """Test the initialization of CustomConversationEntity."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()
    state = hass.states.get("conversation.test")
    
    assert state
    assert state.attributes["supported_features"] == 0

    hass.config_entries.async_update_entry(
        config_entry,
        options={
            **config_entry.options,
            CONF_LLM_HASS_API: "assist",
        },
    )
    await hass.config_entries.async_reload(config_entry.entry_id)

    state = hass.states.get("conversation.test")
    assert state
    assert (
        state.attributes["supported_features"]
        == conversation.ConversationEntityFeature.CONTROL
    )

async def test_custom_conversation_tries_hass_agent_first(hass: HomeAssistant, config_entry: CustomConversationConfigEntry):
    """Test that CustomConversationEntity tries the Home Assistant agent first when both are enabled."""
    assert await async_setup_component(hass, "custom_conversation", {})
    await hass.async_block_till_done()
    mock_response = intent.IntentResponse(language="en", intent=Mock())
    mock_response.error_code = None
    mock_result = conversation.ConversationResult(mock_response, "test-conversation-id")
    with patch(
        "custom_components.custom_conversation.conversation.CustomConversationEntity._async_process_hass", return_value=mock_result
    ) as mock_process_hass:

        hass.config_entries.async_update_entry(
            config_entry,
            options={
                **config_entry.options,
                CONF_LLM_HASS_API: LLM_API_ID,
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: True,
                },
            },
        )
        await hass.config_entries.async_reload(config_entry.entry_id)

        result = await conversation.async_converse(hass, "hello", "test-conversation-id", Context(), agent_id=config_entry.entry_id)
    assert result.conversation_id == "test-conversation-id"
    assert mock_process_hass.called
    