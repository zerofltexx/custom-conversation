"""Prompt management for  Custom Conversation component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import TemplateError
from homeassistant.helpers import entity_registry as er, template
from homeassistant.util import yaml

from .const import (
    CONF_API_PROMPT_BASE,
    CONF_CUSTOM_PROMPTS_SECTION,
    CONF_INSTRUCTIONS_PROMPT,
    CONF_PROMPT_BASE,
    CONF_PROMPT_DEVICE_KNOWN_LOCATION,
    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
    CONF_PROMPT_EXPOSED_ENTITIES,
    CONF_PROMPT_NO_ENABLED_ENTITIES,
    CONF_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_API_PROMPT_BASE,
    DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
    DEFAULT_BASE_PROMPT,
    DEFAULT_INSTRUCTIONS_PROMPT,
    DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
    LOGGER,
)


@dataclass
class PromptContext:
    """Context for prompt generation."""

    hass: HomeAssistant
    ha_name: str
    user_name: str | None = None
    llm_context: Any | None = None
    location: str | None = None
    exposed_entities: dict | None = None
    supports_timers: bool = True


class PromptManager:
    """Manager for Custom Conversation prompts."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the prompt manager."""
        self.hass = hass

    def _get_config_entry_from_context(self, llm_context: Any) -> ConfigEntry | None:
        """Get config entry from LLM context."""
        if not llm_context or not llm_context.context.origin_event:
            return None

        originating_entity = llm_context.context.origin_event.data.get("entity_id")
        if not originating_entity:
            return None

        entity_registry = er.async_get(self.hass)
        entity_entry = entity_registry.async_get(originating_entity)
        if not entity_entry:
            return None

        return self.hass.config_entries.async_get_entry(entity_entry.config_entry_id)

    def _get_prompt_config(
        self, config_entry: ConfigEntry | None, key: str, default: str
    ) -> str:
        """Get prompt configuration with fallback to defaults."""
        if not config_entry:
            return default

        return config_entry.options.get(CONF_CUSTOM_PROMPTS_SECTION, {}).get(
            key, default
        )

    async def async_get_base_prompt(
        self, context: PromptContext, config_entry: ConfigEntry | None = None
    ) -> str:
        """Get the base prompt with rendered template."""
        try:
            base_prompt = self._get_prompt_config(
                config_entry, CONF_PROMPT_BASE, DEFAULT_BASE_PROMPT
            )
            instructions_prompt = self._get_prompt_config(
                config_entry, CONF_INSTRUCTIONS_PROMPT, DEFAULT_INSTRUCTIONS_PROMPT
            )

            return template.Template(
                base_prompt + "\n" + instructions_prompt,
                context.hass,
            ).async_render(
                {
                    "ha_name": context.ha_name,
                    "user_name": context.user_name,
                    "llm_context": context.llm_context,
                },
                parse_result=False,
            )
        except TemplateError as err:
            LOGGER.error("Error rendering base prompt: %s", err)
            raise

    def get_api_prompt(
        self, context: PromptContext, config_entry: ConfigEntry | None = None
    ) -> str:
        """Get the API prompt based on context."""
        prompt_parts = []

        if not context.exposed_entities:
            return self._get_prompt_config(
                config_entry,
                CONF_PROMPT_NO_ENABLED_ENTITIES,
                DEFAULT_PROMPT_NO_ENABLED_ENTITIES,
            )

        # Add base API prompt
        prompt_parts.append(
            self._get_prompt_config(
                config_entry, CONF_API_PROMPT_BASE, DEFAULT_API_PROMPT_BASE
            )
        )

        # Add location-specific prompt
        if context.location:
            location_prompt = self._get_prompt_config(
                config_entry,
                CONF_PROMPT_DEVICE_KNOWN_LOCATION,
                DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION,
            )
            prompt_parts.append(
                template.Template(location_prompt, context.hass).async_render(
                    {"location": context.location}, parse_result=False
                )
            )
        else:
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_DEVICE_UNKNOWN_LOCATION,
                    DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION,
                )
            )

        # Add timer capability prompt if needed
        if not context.supports_timers:
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_TIMERS_UNSUPPORTED,
                    DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED,
                )
            )

        # Add exposed entities prompt and data
        if context.exposed_entities:
            prompt_parts.append(
                self._get_prompt_config(
                    config_entry,
                    CONF_PROMPT_EXPOSED_ENTITIES,
                    DEFAULT_API_PROMPT_EXPOSED_ENTITIES,
                )
            )
            prompt_parts.append(yaml.dump(list(context.exposed_entities.values())))

        return "\n".join(prompt_parts)
