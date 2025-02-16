"""Config flow for Custom Conversation integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import section
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    CONF_AGENTS_SECTION,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONF_LLM_PARAMETERS_SECTION,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    LLM_API_ID,
    RECOMMENDED_BASE_URL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
        vol.Required(CONF_BASE_URL, default=RECOMMENDED_BASE_URL): str,
    }
)

RECOMMENDED_OPTIONS = {
    CONF_LLM_HASS_API: LLM_API_ID,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,   
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
}

def custom_conversation_config_option_schema(
        hass: HomeAssistant,
        options: dict[str, Any] | MappingProxyType[str, Any],
) -> vol.Schema:
    """ Return a Schema for Custom Conversation Options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    # Basic options that are always shown
    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                ),
            },
            default=llm.DEFAULT_INSTRUCTIONS_PROMPT,
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        # Agent section
        vol.Required(CONF_AGENTS_SECTION): section(
            vol.Schema(
                {
                    vol.Required(
                        CONF_ENABLE_HASS_AGENT,
                        default=options.get("agents", {}).get(CONF_ENABLE_HASS_AGENT, True),
                    ): bool,
                    vol.Required(
                        CONF_ENABLE_LLM_AGENT,
                        default=options.get("agents", {}).get(CONF_ENABLE_LLM_AGENT, True),
                    ): bool,
                }
            )
        ),
        vol.Required(CONF_LLM_PARAMETERS_SECTION): section(
            vol.Schema(
                {
                    vol.Optional(
                        CONF_CHAT_MODEL,
                        description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                        default=RECOMMENDED_CHAT_MODEL,
                    ): str,
                    vol.Optional(
                        CONF_MAX_TOKENS,
                        description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                        default=RECOMMENDED_MAX_TOKENS,
                    ): int,
                    vol.Optional(
                        CONF_TOP_P,
                        description={"suggested_value": options.get(CONF_TOP_P)},
                        default=RECOMMENDED_TOP_P,
                    ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
                    vol.Optional(
                        CONF_TEMPERATURE,
                        description={"suggested_value": options.get(CONF_TEMPERATURE)},
                        default=RECOMMENDED_TEMPERATURE,
                    ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
                }
            )
        ),
    }

    return schema

async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    client = openai.AsyncOpenAI(
        api_key=data[CONF_API_KEY], base_url=data[CONF_BASE_URL]
    )
    await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)


class CustomConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Custom Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title="CustomConversation",
                data=user_input,
                options=RECOMMENDED_OPTIONS,
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return CustomConversationOptionsFlow()


class CustomConversationOptionsFlow(OptionsFlow):
    """Custom Conversation config flow options handler."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            if user_input[CONF_LLM_HASS_API] == "none":
                user_input.pop(CONF_LLM_HASS_API)
            return self.async_create_entry(title="", data=user_input)

        schema = custom_conversation_config_option_schema(self.hass, options)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )