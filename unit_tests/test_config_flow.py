"""Test the Custom Conversation config flow."""
from unittest.mock import patch

from custom_components.custom_conversation.const import (
    CONF_AGENTS_SECTION,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_PROMPT_BASE,
    CONF_PROMPT_NO_ENABLED_ENTITIES,
    CONF_API_PROMPT_BASE,
    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
    CONF_PROMPT_TIMERS_UNSUPPORTED,
    CONF_PROMPT_EXPOSED_ENTITIES,
    CONF_IGNORED_INTENTS_SECTION,
    CONF_IGNORED_INTENTS,
    CONF_CHAT_MODEL,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONF_IGNORED_INTENTS,
    CONF_IGNORED_INTENTS_SECTION,
    CONF_ENABLE_LANGFUSE,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_BASE_PROMPT_ID,
    CONF_LANGFUSE_API_PROMPT_ID,
    CONF_LANGFUSE_TRACING_ENABLED,
    CONF_LLM_PARAMETERS_SECTION,
    CONF_MAX_TOKENS,
    CONF_INSTRUCTIONS_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_BASE_PROMPT,
    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
    DEFAULT_INSTRUCTIONS_PROMPT,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
    DOMAIN,
    LLM_API_ID,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    CONF_BASE_URL,
)

from homeassistant.const import CONF_LLM_HASS_API, CONF_API_KEY
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm
from homeassistant import config_entries
from pytest_homeassistant_custom_component.common import MockConfigEntry



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
    assert result["options"][CONF_INSTRUCTIONS_PROMPT] == llm.DEFAULT_INSTRUCTIONS_PROMPT
    assert result["options"][CONF_LLM_HASS_API] == LLM_API_ID
    assert result["options"][CONF_AGENTS_SECTION][CONF_ENABLE_HASS_AGENT]
    assert result["options"][CONF_AGENTS_SECTION][CONF_ENABLE_LLM_AGENT]
    assert result["options"][CONF_LLM_PARAMETERS_SECTION][CONF_CHAT_MODEL] == RECOMMENDED_CHAT_MODEL
    assert result["options"][CONF_LLM_PARAMETERS_SECTION][CONF_MAX_TOKENS] == RECOMMENDED_MAX_TOKENS
    assert result["options"][CONF_LLM_PARAMETERS_SECTION][CONF_TOP_P] == RECOMMENDED_TOP_P
    assert result["options"][CONF_LLM_PARAMETERS_SECTION][CONF_TEMPERATURE] == RECOMMENDED_TEMPERATURE
    

async def test_options_flow(hass: HomeAssistant, config_entry):
    """Test config flow options with sections."""
    
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
                CONF_INSTRUCTIONS_PROMPT: "test-prompt",
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
                },
                CONF_CUSTOM_PROMPTS_SECTION: {
                    CONF_PROMPT_BASE: "test-prompt-base",
                    CONF_PROMPT_NO_ENABLED_ENTITIES: "test-prompt-no-entities",
                    CONF_API_PROMPT_BASE: "test-api-prompt-base",
                    CONF_PROMPT_DEVICE_KNOWN_LOCATION: "test-prompt-known-location",
                    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION: "test-prompt-unknown-location",
                    CONF_PROMPT_TIMERS_UNSUPPORTED: "test-prompt-timers-unsupported",
                    CONF_PROMPT_EXPOSED_ENTITIES: "test-prompt-exposed-entities",
                },
                CONF_LANGFUSE_SECTION: {
                    CONF_ENABLE_LANGFUSE: True,
                    CONF_LANGFUSE_SECRET_KEY: "sk-test-secret-key",
                    CONF_LANGFUSE_PUBLIC_KEY: "pk-test-public-key",
                    CONF_LANGFUSE_HOST: "http://langfuse.test",
                    CONF_LANGFUSE_BASE_PROMPT_ID: "test-base-prompt-id",
                    CONF_LANGFUSE_API_PROMPT_ID: "test-api-prompt-id",
                    CONF_LANGFUSE_TRACING_ENABLED: False
                }
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_INSTRUCTIONS_PROMPT] == "test-prompt"
        assert CONF_LLM_HASS_API not in result["data"]
        assert "agents" in result["data"]
        assert result["data"][CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] == ["HassGetState"]
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_HASS_AGENT] is True
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_LLM_AGENT] is False
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_CHAT_MODEL] == "test-chat-model"
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_MAX_TOKENS] == 50
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_TOP_P] == 0.5
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_TEMPERATURE] == 0.5
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_BASE] == "test-prompt-base"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_NO_ENABLED_ENTITIES] == "test-prompt-no-entities"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_API_PROMPT_BASE] == "test-api-prompt-base"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_DEVICE_KNOWN_LOCATION] == "test-prompt-known-location"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_DEVICE_UNKNOWN_LOCATION] == "test-prompt-unknown-location"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_TIMERS_UNSUPPORTED] == "test-prompt-timers-unsupported"
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_EXPOSED_ENTITIES] == "test-prompt-exposed-entities"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_ENABLE_LANGFUSE] is True
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_SECRET_KEY] == "sk-test-secret-key"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_PUBLIC_KEY] == "pk-test-public-key"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_HOST] == "http://langfuse.test"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_BASE_PROMPT_ID] == "test-base-prompt-id"
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_API_PROMPT_ID] == "test-api-prompt-id"

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
                },
                CONF_CUSTOM_PROMPTS_SECTION: {
                    CONF_PROMPT_BASE: "",
                    CONF_PROMPT_NO_ENABLED_ENTITIES: "",
                    CONF_API_PROMPT_BASE: "",
                    CONF_PROMPT_DEVICE_KNOWN_LOCATION: "",
                    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION: "",
                    CONF_PROMPT_TIMERS_UNSUPPORTED: "",
                    CONF_PROMPT_EXPOSED_ENTITIES: "",
                },
                CONF_LANGFUSE_SECTION: {
                    CONF_ENABLE_LANGFUSE: False,
                    CONF_LANGFUSE_SECRET_KEY: "",
                    CONF_LANGFUSE_PUBLIC_KEY: "",
                    CONF_LANGFUSE_HOST: "",
                    CONF_LANGFUSE_BASE_PROMPT_ID: "",
                    CONF_LANGFUSE_API_PROMPT_ID: "",
                    CONF_LANGFUSE_TRACING_ENABLED: False
                }

            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_INSTRUCTIONS_PROMPT] == DEFAULT_INSTRUCTIONS_PROMPT.strip()
        assert CONF_LLM_HASS_API not in result["data"]
        assert "agents" in result["data"]
        assert result["data"][CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] == llm.AssistAPI.IGNORE_INTENTS
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_HASS_AGENT] is True
        assert result["data"][CONF_AGENTS_SECTION][CONF_ENABLE_LLM_AGENT] is False
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_CHAT_MODEL] == RECOMMENDED_CHAT_MODEL
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_MAX_TOKENS] == 50
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_TOP_P] == 0.5
        assert result["data"][CONF_LLM_PARAMETERS_SECTION][CONF_TEMPERATURE] == 0.5
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_BASE].strip() == DEFAULT_BASE_PROMPT.strip()
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_NO_ENABLED_ENTITIES] == DEFAULT_PROMPT_NO_ENABLED_ENTITIES
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_API_PROMPT_BASE] == DEFAULT_API_PROMPT_BASE
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_DEVICE_KNOWN_LOCATION] == DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_DEVICE_UNKNOWN_LOCATION] == DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_TIMERS_UNSUPPORTED] == DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED
        assert result["data"][CONF_CUSTOM_PROMPTS_SECTION][CONF_PROMPT_EXPOSED_ENTITIES] == DEFAULT_API_PROMPT_EXPOSED_ENTITIES
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_ENABLE_LANGFUSE] is False
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_SECRET_KEY] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_PUBLIC_KEY] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_HOST] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_BASE_PROMPT_ID] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_API_PROMPT_ID] == ""
        assert result["data"][CONF_LANGFUSE_SECTION][CONF_LANGFUSE_TRACING_ENABLED] is False


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
                },
                CONF_CUSTOM_PROMPTS_SECTION: {},
                CONF_LANGFUSE_SECTION: {}
            },
        )
        assert result["type"] == FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_IGNORED_INTENTS_SECTION][CONF_IGNORED_INTENTS] == test_intents


async def test_step_reconfigure_prefills_data(hass: HomeAssistant, config_entry):
    """Test reconfigure step prefills with existing config data."""

    result = await config_entry.start_reconfigure_flow(hass)

    assert result["type"] == FlowResultType.FORM
    assert result["step_id"] == "reconfigure"
    # Check that the defaults are set from existing config
    assert result["data_schema"].schema
    api_key = next(key.default() for key in result["data_schema"].schema if key == CONF_API_KEY)
    assert api_key == "test-api-key"
    url = next(key.default() for key in result["data_schema"].schema if key == CONF_BASE_URL)
    assert url == "https://api.openai.com/v1"

    with patch(
        "custom_components.custom_conversation.async_setup", return_value=True
    ), patch(
        "custom_components.custom_conversation.async_setup_entry",
        return_value=True,
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_API_KEY: "new-api-key",
                CONF_BASE_URL: "https://new-url.com",
            },
        )
        await hass.async_block_till_done()

    assert result["type"] == FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"