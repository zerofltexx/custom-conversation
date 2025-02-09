import pytest
from homeassistant.components import conversation
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from unittest.mock import Mock
from custom_components.custom_conversation.conversation import CustomConversationEntity
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