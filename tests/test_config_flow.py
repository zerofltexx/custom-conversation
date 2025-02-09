"""Test the Custom Conversation config flow."""
from unittest.mock import patch

from custom_components.custom_conversation.const import DOMAIN, CONF_RECOMMENDED, CONF_PROMPT, CONF_ENABLE_HASS_AGENT, CONF_ENABLE_LLM_AGENT, CONF_MAX_TOKENS, CONF_TEMPERATURE

from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
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

async def test_options_flow_recommended_no_llm_api(hass: HomeAssistant, config_entry):
    """Test config flow options with recommended settings and no LLM API."""
    
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
                CONF_RECOMMENDED: True,
                CONF_PROMPT: "test-prompt",
                CONF_LLM_HASS_API: "none",
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_RECOMMENDED]
        assert result["data"][CONF_PROMPT] == "test-prompt"
        assert CONF_LLM_HASS_API not in result["data"]

async def test_options_flow_recommended_with_llm_api(hass: HomeAssistant, config_entry):
    """Test config flow options with recommended settings and an LLM API selected."""
    
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
                CONF_RECOMMENDED: True,
                CONF_PROMPT: "test-prompt",
                CONF_LLM_HASS_API: "assist",
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_RECOMMENDED]
        assert result["data"][CONF_PROMPT] == "test-prompt"
        assert result["data"][CONF_LLM_HASS_API] == "assist"

async def test_options_flow_not_recommended_no_llm_api(hass: HomeAssistant, config_entry):
    """Test config flow with custom settings and no LLM API."""
    
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
                CONF_RECOMMENDED: False,
                CONF_PROMPT: "test-prompt",
                CONF_LLM_HASS_API: "none",
            },
        )
        # Since we're switching recommended to False, we should get the advanced options form
        assert result["type"] == FlowResultType.FORM

        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            user_input={
                CONF_RECOMMENDED: False,
                CONF_PROMPT: "test-prompt",
                CONF_LLM_HASS_API: "none",
                CONF_TEMPERATURE: 0.5,
                CONF_MAX_TOKENS: 100,
            },
        )

        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert not result["data"][CONF_RECOMMENDED]
        assert result["data"][CONF_PROMPT] == "test-prompt"
        assert CONF_LLM_HASS_API not in result["data"]
        assert result["data"][CONF_TEMPERATURE] == 0.5
        assert result["data"][CONF_MAX_TOKENS] == 100