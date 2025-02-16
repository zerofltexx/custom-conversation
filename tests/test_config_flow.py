"""Test the Custom Conversation config flow."""
from unittest.mock import patch

from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_IGNORED_INTENTS_SECTION,
    CONF_IGNORED_INTENTS,
    CONF_CHAT_MODEL,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONF_IGNORED_INTENTS,
    CONF_IGNORED_INTENTS_SECTION,
    CONF_LLM_PARAMETERS_SECTION,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LLM_API_ID,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm
from homeassistant import config_entries




async def test_show_config_form(hass: HomeAssistant):
    """Test that the setup form is served."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    assert result["type"] == FlowResultType.FORM
    assert result["errors"] is None

    with patch(
        "custom_components.custom_conversation.async_setup", return_value=True
    ) as mock_setup , patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ) as mock_setup_entry:
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], {"api_key": "test-api-key"}
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.CREATE_ENTRY
    assert result["title"] == "CustomConversation"
    assert result["data"]["api_key"] == "test-api-key"

    # Recommended Options should be set by default at this point
    assert result["options"][CONF_PROMPT] == llm.DEFAULT_INSTRUCTIONS_PROMPT
    assert result["options"][CONF_LLM_HASS_API] == LLM_API_ID
    assert result["options"][CONF_AGENTS_SECTION][CONF_ENABLE_HASS_AGENT]
    assert result["options"][CONF_AGENTS_SECTION][CONF_ENABLE_LLM_AGENT]
    assert result["options"][CONF_LLM_PARAMETERS_SECTION][CONF_CHAT_MODEL] == RECOMMENDED_CHAT_MODEL
    assert result["options"][CONF_LLM_PARAMETERS_SECTION][CONF_MAX_TOKENS] == RECOMMENDED_MAX_TOKENS
    assert result["options"][CONF_LLM_PARAMETERS_SECTION][CONF_TOP_P] == RECOMMENDED_TOP_P
    assert result["options"][CONF_LLM_PARAMETERS_SECTION][CONF_TEMPERATURE] == RECOMMENDED_TEMPERATURE
    

async def test_options_flow(hass: HomeAssistant, config_entry):
    """Test config flow options with sections in recommended mode."""
    
    config_entry.add_to_hass(hass)

    with patch(
        "custom_components.custom_conversation.async_setup", return_value=True
    ), patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ):
        result = await hass.config_entries.options.async_init(config_entry.entry_id)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_PROMPT: "test-prompt",
                CONF_LLM_HASS_API: "none",
                CONF_IGNORED_INTENTS_SECTION: {
                    CONF_IGNORED_INTENTS: ["HassGetState"],
                },
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: False,
                },
                CONF_LLM_PARAMETERS_SECTION: {
                    CONF_CHAT_MODEL: "test-chat-model",
                    CONF_MAX_TOKENS: 50,
                    CONF_TOP_P: 0.5,
                    CONF_TEMPERATURE: 0.5,
                }

            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_PROMPT] == "test-prompt"
        assert CONF_LLM_HASS_API not in result["data"]
        assert "agents" in result["data"]
        assert result["data"][CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] == ["HassGetState"]
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_HASS_AGENT] is True
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_LLM_AGENT] is False
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_CHAT_MODEL] == "test-chat-model"
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_MAX_TOKENS] == 50
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_TOP_P] == 0.5
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_TEMPERATURE] == 0.5

async def test_options_flow_empty_fields_reset(hass: HomeAssistant, config_entry):
    """Test config flow options with empty fields reset to recommended."""
    
    config_entry.add_to_hass(hass)

    with patch(
        "custom_components.custom_conversation.async_setup", return_value=True
    ), patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ):
        result = await hass.config_entries.options.async_init(config_entry.entry_id)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"

        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_LLM_HASS_API: "none",
                CONF_IGNORED_INTENTS_SECTION: {
                    CONF_IGNORED_INTENTS: [],
                },
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: False,
                },
                CONF_LLM_PARAMETERS_SECTION: {
                    CONF_MAX_TOKENS: 50,
                    CONF_TOP_P: 0.5,
                    CONF_TEMPERATURE: 0.5,
                }

            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_PROMPT] == llm.DEFAULT_INSTRUCTIONS_PROMPT.strip()
        assert CONF_LLM_HASS_API not in result["data"]
        assert "agents" in result["data"]
        assert result["data"][CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] == llm.AssistAPI.IGNORE_INTENTS
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_HASS_AGENT] is True
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_LLM_AGENT] is False
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_CHAT_MODEL] == RECOMMENDED_CHAT_MODEL
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_MAX_TOKENS] == 50
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_TOP_P] == 0.5
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_TEMPERATURE] == 0.5

async def test_options_flow_ignored_intents(hass: HomeAssistant, config_entry):
    """Test config flow options with ignored intents."""
    
    config_entry.add_to_hass(hass)
    with patch(
        "custom_components.custom_conversation.async_setup", return_value=True
    ), patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ), patch(
        "homeassistant.helpers.intent.async_get",
        return_value=[
            type("Intent", (), {"intent_type": "HassTurnOn"}),
            type("Intent", (), {"intent_type": "HassTurnOff"}),
        ],
    ):
        result = await hass.config_entries.options.async_init(config_entry.entry_id)
        assert result["type"] == FlowResultType.FORM
        assert result["step_id"] == "init"
        
        test_intents = ["HassTurnOn", "HassTurnOff"]
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_IGNORED_INTENTS_SECTION: {
                    CONF_IGNORED_INTENTS: test_intents,
                },
                CONF_AGENTS_SECTION: {
                    CONF_ENABLE_HASS_AGENT: True,
                    CONF_ENABLE_LLM_AGENT: True,
                },
                CONF_LLM_PARAMETERS_SECTION: {
                    CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
                    CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
                    CONF_TOP_P: RECOMMENDED_TOP_P,
                    CONF_TEMPERATURE: RECOMMENDED_TEMPERATURE,
                }
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] == test_intents