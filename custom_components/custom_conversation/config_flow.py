"""Config flow for Custom Conversation integration."""

from __future__ import annotations

from typing import Any

from litellm.exceptions import APIConnectionError, AuthenticationError, BadRequestError
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry, ConfigFlow, OptionsFlow
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult, section
from homeassistant.helpers import intent, llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
)

from .const import (
    CONF_AGENTS_SECTION,
    CONF_API_PROMPT_BASE,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LANGFUSE,
    CONF_ENABLE_LLM_AGENT,
    CONF_IGNORED_INTENTS,
    CONF_IGNORED_INTENTS_SECTION,
    CONF_INSTRUCTIONS_PROMPT,
    CONF_LANGFUSE_API_PROMPT_ID,
    CONF_LANGFUSE_API_PROMPT_LABEL,
    CONF_LANGFUSE_BASE_PROMPT_ID,
    CONF_LANGFUSE_BASE_PROMPT_LABEL,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_SCORE_ENABLED,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_TAGS,
    CONF_LANGFUSE_TRACING_ENABLED,
    CONF_MAX_TOKENS,
    CONF_PRIMARY_API_KEY,
    CONF_PRIMARY_BASE_URL,
    CONF_PRIMARY_CHAT_MODEL,
    CONF_PRIMARY_PROVIDER,
    CONF_PROMPT_BASE,
    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
    CONF_PROMPT_EXPOSED_ENTITIES,
    CONF_PROMPT_NO_ENABLED_ENTITIES,
    CONF_PROMPT_TIMERS_UNSUPPORTED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONFIG_VERSION,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_BASE_PROMPT,
    DEFAULT_INSTRUCTIONS_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DOMAIN,
    HASS_DEPRECATED_INTENTS,
    LLM_API_ID,
    LOGGER,
    # RECOMMENDED_BASE_URL, # Removed
    # RECOMMENDED_CHAT_MODEL, # Removed
    # RECOMMENDED_MAX_TOKENS, # Removed
    # RECOMMENDED_TEMPERATURE, # Removed
    # RECOMMENDED_TOP_P, # Removed
)
from .litellm_utils import get_valid_models
from .providers import SUPPORTED_PROVIDERS, get_provider

_LOGGER = LOGGER  # Use logger from const.py

# Default values for the OPTIONAL parameters stored in config_entry.options
DEFAULT_OPTIONS = {
    CONF_LLM_HASS_API: "none",  # Default to no Hass API control
    CONF_INSTRUCTIONS_PROMPT: DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_AGENTS_SECTION: {
        CONF_ENABLE_HASS_AGENT: True,
        CONF_ENABLE_LLM_AGENT: True,
    },
    # LLM Params moved to options flow schema defaults
    # CONF_IGNORED_INTENTS: llm.AssistAPI.IGNORE_INTENTS, # Default handled in schema
    CONF_CUSTOM_PROMPTS_SECTION: {
        CONF_PROMPT_BASE: DEFAULT_BASE_PROMPT,
        # CONF_INSTRUCTIONS_PROMPT: DEFAULT_INSTRUCTIONS_PROMPT, # Handled directly
        CONF_PROMPT_NO_ENABLED_ENTITIES: DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
        CONF_API_PROMPT_BASE: DEFAULT_API_PROMPT_BASE,
        CONF_PROMPT_DEVICE_KNOWN_LOCATION: DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
        CONF_PROMPT_DEVICE_UNKNOWN_LOCATION: DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
        CONF_PROMPT_TIMERS_UNSUPPORTED: DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
        CONF_PROMPT_EXPOSED_ENTITIES: DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
    },
    CONF_LANGFUSE_SECTION: {
        CONF_ENABLE_LANGFUSE: False,
        CONF_LANGFUSE_HOST: "",
        CONF_LANGFUSE_PUBLIC_KEY: "",
        CONF_LANGFUSE_SECRET_KEY: "",
        CONF_LANGFUSE_BASE_PROMPT_ID: "",
        CONF_LANGFUSE_BASE_PROMPT_LABEL: "production",
        CONF_LANGFUSE_API_PROMPT_ID: "",
        CONF_LANGFUSE_API_PROMPT_LABEL: "production",
        CONF_LANGFUSE_TRACING_ENABLED: False,
        CONF_LANGFUSE_TAGS: [],
        CONF_LANGFUSE_SCORE_ENABLED: False,
    },
    # Add defaults for LLM parameters directly here if needed,
    # but better handled in the options schema itself.
    # CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS, # Handled in schema
    # CONF_TOP_P: DEFAULT_TOP_P, # Handled in schema
    # CONF_TEMPERATURE: DEFAULT_TEMPERATURE, # Handled in schema
}


class CustomConversationConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Custom Conversation."""

    VERSION = CONFIG_VERSION
    _flow_data: dict[str, Any] = {}  # Store data between steps

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the provider selection step."""
        if user_input is not None:
            primary_provider = get_provider(user_input.get(CONF_PRIMARY_PROVIDER))
            self._flow_data.update({CONF_PRIMARY_PROVIDER: primary_provider})
            return await self.async_step_credentials()

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_PRIMARY_PROVIDER, default=SUPPORTED_PROVIDERS[0].key
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(
                                label=provider.provider_name, value=provider.key
                            )
                            for provider in SUPPORTED_PROVIDERS
                        ]
                    )
                )
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema)

    async def async_step_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the credentials step."""
        errors: dict[str, str] = {}
        provider = self._flow_data[CONF_PRIMARY_PROVIDER]

        if user_input is not None:
            self._flow_data.update(user_input)
            try:
                # The get_valid_models call appends /v1 to the base URL, so we need to remove it for this call
                model_check_url = user_input.get(CONF_PRIMARY_BASE_URL, None)
                if model_check_url and model_check_url.endswith("/v1"):
                    model_check_url = model_check_url[:-3]

                # Get the valid models, which will also validate credentials
                valid_models = await self.hass.async_add_executor_job(
                    lambda: get_valid_models(
                        check_provider_endpoint=True,
                        custom_llm_provider=provider.key,
                        litellm_params=GenericLiteLLMParams(
                            api_key=user_input[CONF_PRIMARY_API_KEY],
                            api_base=model_check_url,
                        ),
                    )
                )
                return await self.async_step_model(valid_models=valid_models)
            except AuthenticationError:
                errors["base"] = "invalid_auth"
            except APIConnectionError:
                errors["base"] = "cannot_connect"
            except Exception as err:  # Catch-all for unexpected validation errors
                _LOGGER.error(f"Unexpected error: {err}")
                errors["base"] = "unknown"

        schema_dict = {
            vol.Required(CONF_PRIMARY_API_KEY): TextSelector(
                TextSelectorConfig(type="password")
            ),
        }
        # Currently all of our providers have a default base URL that can also be changed
        default_base_url = ProviderConfigManager.get_provider_model_info(
            model="", provider=provider.key
        ).get_api_base()
        schema_dict[vol.Optional(CONF_PRIMARY_BASE_URL, default=default_base_url)] = str

        schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="credentials",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "provider": provider
            },  # For potential translation strings
        )

    async def async_step_model(
        self,
        user_input: dict[str, Any] | None = None,
        valid_models: list[str] | None = None,
    ) -> FlowResult:
        """Handle the model selection step."""
        errors: dict[str, str] = {}
        provider = self._flow_data[CONF_PRIMARY_PROVIDER]

        if user_input is not None:
            self._flow_data.update(user_input)

            # Final step: create the config entry
            return self.async_create_entry(
                title="Custom Conversation",  # Or use a user-provided title later
                data=self._flow_data,
                options=DEFAULT_OPTIONS,  # Start with default options
            )

        schema_dict = {}
        if valid_models:
            schema_dict[vol.Required(CONF_PRIMARY_CHAT_MODEL)] = SelectSelector(
                SelectSelectorConfig(options=valid_models, custom_value=True, sort=True)
            )
        else:
            # Allow manual entry if models couldn't be fetched
            schema_dict[vol.Required(CONF_PRIMARY_CHAT_MODEL)] = str

        schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="model",
            data_schema=schema,
            errors=errors,
            description_placeholders={"provider": provider},
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle reconfiguration of the integration."""
        # Reconfiguration should re-run the main flow but start with existing data
        # We store existing data in _flow_data to prepopulate steps
        if not hasattr(self, "_flow_data") or not self._flow_data:
            self._flow_data = {**self._get_reconfigure_entry().data}

        # Start reconfigure flow at the provider step
        return await self.async_step_reconfigure_provider()

    async def async_step_reconfigure_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle provider selection during reconfiguration."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._flow_data.update(user_input)
            # If provider changed, might need to clear old incompatible data? For now, just proceed.
            return await self.async_step_reconfigure_credentials()
        current_provider = get_provider(
            self._flow_data.get(CONF_PRIMARY_PROVIDER)["key"]
        )
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_PRIMARY_PROVIDER, default=current_provider.key
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(
                                label=provider.provider_name, value=provider.key
                            )
                            for provider in SUPPORTED_PROVIDERS
                        ]
                    )
                )
            }
        )
        return self.async_show_form(
            step_id="reconfigure_provider", data_schema=schema, errors=errors
        )

    async def async_step_reconfigure_credentials(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle credentials during reconfiguration."""
        errors: dict[str, str] = {}
        provider = get_provider(self._flow_data[CONF_PRIMARY_PROVIDER])

        if user_input is not None:
            self._flow_data.update(user_input)
            try:
                # The get_valid_models call appends /v1 to the base URL, so we need to remove it for this call
                model_check_url = user_input.get(CONF_PRIMARY_BASE_URL, None)
                if model_check_url and model_check_url.endswith("/v1"):
                    model_check_url = model_check_url[:-3]

                # Get the valid models, which will also validate credentials
                valid_models = await self.hass.async_add_executor_job(
                    lambda: get_valid_models(
                        check_provider_endpoint=True,
                        custom_llm_provider=provider.key,
                        litellm_params=GenericLiteLLMParams(
                            api_key=user_input[CONF_PRIMARY_API_KEY],
                            api_base=model_check_url,
                        ),
                    )
                )
                return await self.async_step_reconfigure_model(
                    valid_models=valid_models
                )
            except AuthenticationError:
                errors["base"] = "invalid_auth"
            except APIConnectionError:
                errors["base"] = "cannot_connect"
            except Exception:
                errors["base"] = "unknown"

        # Build schema dynamically, defaulting to existing values
        schema_dict = {
            vol.Required(
                CONF_PRIMARY_API_KEY,
                default=self._flow_data.get(CONF_PRIMARY_API_KEY, ""),
            ): TextSelector(TextSelectorConfig(type="password")),
        }
        if provider == "openai":
            schema_dict[
                vol.Optional(
                    CONF_PRIMARY_BASE_URL,
                    default=self._flow_data.get(CONF_PRIMARY_BASE_URL, ""),
                )
            ] = str

        schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="reconfigure_credentials",
            data_schema=schema,
            errors=errors,
            description_placeholders={"provider": provider},
        )

    async def async_step_reconfigure_model(
        self,
        user_input: dict[str, Any] | None = None,
        valid_models: list[str] | None = None,
    ) -> FlowResult:
        """Handle model selection during reconfiguration."""
        errors: dict[str, str] = {}
        provider = self._flow_data[CONF_PRIMARY_PROVIDER]

        if user_input is not None:
            self._flow_data.update(user_input)
            # Update the config entry and reload
            return self.async_update_reload_and_abort(
                self._get_reconfigure_entry(), data=self._flow_data
            )

        schema_dict = {}
        current_model = self._flow_data.get(CONF_PRIMARY_CHAT_MODEL)
        if valid_models:
            schema_dict[
                vol.Required(CONF_PRIMARY_CHAT_MODEL, default=current_model)
            ] = SelectSelector(
                SelectSelectorConfig(options=valid_models, custom_value=True, sort=True)
            )
        else:
            # Allow manual entry if models couldn't be fetched
            schema_dict[
                vol.Required(CONF_PRIMARY_CHAT_MODEL, default=current_model)
            ] = str

        schema = vol.Schema(schema_dict)

        return self.async_show_form(
            step_id="reconfigure_model",
            data_schema=schema,
            errors=errors,
            description_placeholders={"provider": provider},
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return CustomConversationOptionsFlow(config_entry)


# --- Options Flow Handler ---


class CustomConversationOptionsFlow(OptionsFlow):
    """Custom Conversation config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        # Assuming self.hass is available via the OptionsFlow base class
        # If not, hass might need to be retrieved differently if required by helpers

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            # Process user input before saving
            processed_input = {**user_input}  # Start with a copy

            # Handle potential "none" value for Hass API control
            if processed_input.get(CONF_LLM_HASS_API) == "none":
                processed_input.pop(CONF_LLM_HASS_API, None)  # Remove if 'none'

            # Handle empty ignored intents - use default
            ignored_intents_section = processed_input.get(
                CONF_IGNORED_INTENTS_SECTION, {}
            )
            if not ignored_intents_section.get(CONF_IGNORED_INTENTS):
                ignored_intents_section[CONF_IGNORED_INTENTS] = (
                    llm.AssistAPI.IGNORE_INTENTS
                )
                processed_input[CONF_IGNORED_INTENTS_SECTION] = ignored_intents_section

            # Remove empty prompt defaults logic - user should explicitly set or clear

            return self.async_create_entry(title="", data=processed_input)

        # Build schema for options
        options = self.config_entry.options
        # Pass hass explicitly if needed by helpers
        # OptionsFlow inherits from FlowHandler which has self.hass
        hass = self.hass
        hass_apis = self._get_hass_apis(hass)
        intents = await self._get_intents(hass)
        default_ignored = llm.AssistAPI.IGNORE_INTENTS

        # Define the schema for options, using existing options as defaults
        schema = vol.Schema(
            {
                # LLM Parameters (now direct options)
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
                vol.Optional(
                    CONF_TOP_P,
                    default=options.get(CONF_TOP_P, DEFAULT_TOP_P),
                ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                ): vol.All(
                    vol.Coerce(int), vol.Range(min=1)
                ),  # Ensure positive integer
                # Hass API Control
                vol.Optional(
                    CONF_LLM_HASS_API,
                    description={
                        "suggested_value": options.get(CONF_LLM_HASS_API)
                    },  # Keep suggested value if present
                    default="none",  # Default to none if not set
                ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
                # Agent Section
                vol.Required(CONF_AGENTS_SECTION): section(
                    vol.Schema(
                        {
                            vol.Required(
                                CONF_ENABLE_HASS_AGENT,
                                default=options.get(CONF_AGENTS_SECTION, {}).get(
                                    CONF_ENABLE_HASS_AGENT, True
                                ),
                            ): bool,
                            vol.Required(
                                CONF_ENABLE_LLM_AGENT,
                                default=options.get(CONF_AGENTS_SECTION, {}).get(
                                    CONF_ENABLE_LLM_AGENT, True
                                ),
                            ): bool,
                        }
                    )
                ),
                # Ignored Intents Section
                vol.Required(CONF_IGNORED_INTENTS_SECTION): section(
                    vol.Schema(
                        {
                            vol.Required(
                                CONF_IGNORED_INTENTS,
                                default=options.get(
                                    CONF_IGNORED_INTENTS_SECTION, {}
                                ).get(CONF_IGNORED_INTENTS, default_ignored),
                            ): SelectSelector(
                                SelectSelectorConfig(
                                    options=intents, multiple=True, sort=True
                                )
                            ),
                        }
                    )
                ),
                # Custom Prompts Section
                vol.Required(CONF_CUSTOM_PROMPTS_SECTION): section(
                    vol.Schema(
                        {
                            # Use .get for defaults to avoid KeyError if section doesn't exist yet
                            vol.Optional(
                                CONF_INSTRUCTIONS_PROMPT,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_INSTRUCTIONS_PROMPT,
                                    DEFAULT_INSTRUCTIONS_PROMPT,
                                ),
                            ): TemplateSelector(),
                            vol.Optional(
                                CONF_PROMPT_BASE,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(CONF_PROMPT_BASE, DEFAULT_BASE_PROMPT),
                            ): TemplateSelector(),
                            vol.Optional(
                                CONF_PROMPT_NO_ENABLED_ENTITIES,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_PROMPT_NO_ENABLED_ENTITIES,
                                    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
                                ),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_API_PROMPT_BASE,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(CONF_API_PROMPT_BASE, DEFAULT_API_PROMPT_BASE),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_PROMPT_DEVICE_KNOWN_LOCATION,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
                                    DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
                                ),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
                                    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
                                ),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_PROMPT_TIMERS_UNSUPPORTED,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_PROMPT_TIMERS_UNSUPPORTED,
                                    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
                                ),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                            vol.Optional(
                                CONF_PROMPT_EXPOSED_ENTITIES,
                                default=options.get(
                                    CONF_CUSTOM_PROMPTS_SECTION, {}
                                ).get(
                                    CONF_PROMPT_EXPOSED_ENTITIES,
                                    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
                                ),
                            ): TextSelector(TextSelectorConfig(multiline=True)),
                        }
                    )
                ),
                # Langfuse Section
                vol.Required(CONF_LANGFUSE_SECTION): section(
                    vol.Schema(
                        {
                            vol.Required(
                                CONF_ENABLE_LANGFUSE,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_ENABLE_LANGFUSE, False
                                ),
                            ): bool,
                            vol.Optional(
                                CONF_LANGFUSE_HOST,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_HOST, ""
                                ),
                            ): str,
                            vol.Optional(
                                CONF_LANGFUSE_PUBLIC_KEY,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_PUBLIC_KEY, ""
                                ),
                            ): str,
                            vol.Optional(
                                CONF_LANGFUSE_SECRET_KEY,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_SECRET_KEY, ""
                                ),
                            ): TextSelector(TextSelectorConfig(type="password")),
                            vol.Optional(
                                CONF_LANGFUSE_BASE_PROMPT_ID,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_BASE_PROMPT_ID, ""
                                ),
                            ): str,
                            vol.Optional(
                                CONF_LANGFUSE_BASE_PROMPT_LABEL,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_BASE_PROMPT_LABEL, "production"
                                ),
                            ): str,
                            vol.Optional(
                                CONF_LANGFUSE_API_PROMPT_ID,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_API_PROMPT_ID, ""
                                ),
                            ): str,
                            vol.Optional(
                                CONF_LANGFUSE_API_PROMPT_LABEL,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_API_PROMPT_LABEL, "production"
                                ),
                            ): str,
                            vol.Optional(
                                CONF_LANGFUSE_TRACING_ENABLED,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_TRACING_ENABLED, False
                                ),
                            ): bool,
                            vol.Optional(
                                CONF_LANGFUSE_TAGS,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_TAGS, []
                                ),
                            ): SelectSelector(
                                SelectSelectorConfig(
                                    options=[],
                                    multiple=True,
                                    custom_value=True,
                                    sort=True,
                                )
                            ),
                            vol.Optional(
                                CONF_LANGFUSE_SCORE_ENABLED,
                                default=options.get(CONF_LANGFUSE_SECTION, {}).get(
                                    CONF_LANGFUSE_SCORE_ENABLED, False
                                ),
                            ): bool,
                        }
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
        )

    # Helper methods moved inside the class or made static if they don't need self
    # Pass hass explicitly as OptionsFlow might not have self.hass reliably
    def _get_hass_apis(self, hass: HomeAssistant) -> list[SelectOptionDict]:
        """Get available Home Assistant LLM APIs."""
        hass_apis: list[SelectOptionDict] = [
            SelectOptionDict(label="No control", value="none")
        ]
        try:
            hass_apis.extend(
                SelectOptionDict(label=api.name, value=api.id)
                for api in llm.async_get_apis(hass)
            )
        except Exception as e:
            _LOGGER.exception("Error fetching HASS APIs: %s", e)
        return hass_apis

    async def _get_intents(self, hass: HomeAssistant) -> list[SelectOptionDict]:
        """Get available intents."""
        hass_recommended_ignored = llm.AssistAPI.IGNORE_INTENTS
        intents_list: list[SelectOptionDict] = [
            {
                "value": intent.intent_type,
                "label": f"{intent.intent_type} (Hass Recommended)"
                if intent.intent_type in hass_recommended_ignored
                else intent.intent_type,
            }
            for intent in intent.async_get(hass)
        ]
        return intents_list
