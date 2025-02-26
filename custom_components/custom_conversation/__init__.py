"""The Custom Conversation integration."""

from __future__ import annotations

from langfuse.openai import openai

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv, llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType

from .api import CustomLLMAPI
from .const import (
    CONF_BASE_URL,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_SCORE_ENABLED,
    CONF_LANGFUSE_SECTION,
    DOMAIN,
    LLM_API_ID,
    LOGGER,
)
from .prompt_manager import LangfuseClient, LangfuseError
from .service import async_setup_services

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

type CustomConversationConfigEntry = ConfigEntry[openai.AsyncClient]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Custom Conversation."""

    # Make sure the API is registered
    if not any(x.id == LLM_API_ID for x in llm.async_get_apis(hass)):
        llm.async_register_api(hass, CustomLLMAPI(hass))

    await async_setup_services(hass)

    return True


async def async_setup_entry(
    hass: HomeAssistant, entry: CustomConversationConfigEntry
) -> bool:
    """Set up a  Custom Conversation from a config entry."""
    client = openai.AsyncOpenAI(
        api_key=entry.data[CONF_API_KEY],
        http_client=get_async_client(hass),
        base_url=entry.data[CONF_BASE_URL],
    )

    # Cache current platform data which gets added to each request (caching done by library)
    _ = await hass.async_add_executor_job(client.platform_headers)

    try:
        await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)
    except openai.AuthenticationError as err:
        LOGGER.error("Invalid API key: %s", err)
        return False
    except openai.OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    entry.runtime_data = client

    langfuse_client = None
    # initialize Langfuse client if enabled
    try:
        langfuse_client = await LangfuseClient.create(hass, entry)
    except LangfuseError as err:
        LOGGER.error("Error initializing Langfuse client: %s", err)
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][entry.entry_id] = {
        "langfuse_client": langfuse_client,
    }
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Set up Langfuse trace config if enabled
    if entry.options.get(CONF_LANGFUSE_SECTION, {}).get(CONF_LANGFUSE_SCORE_ENABLED):
        # Get existing score configs
        langfuse_host = entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_LANGFUSE_HOST
        )
        if not langfuse_host:
            LOGGER.error(
                "Langfuse score enabled but no host provided in options: %s",
                entry.options,
            )
            return False
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Clean up clients."""
    # Clean up Langfuse client if it exists
    if (
        DOMAIN in hass.data
        and entry.entry_id in hass.data[DOMAIN]
        and hass.data[DOMAIN][entry.entry_id].get("langfuse_client")
    ):
        langfuse_client = hass.data[DOMAIN][entry.entry_id]["langfuse_client"]
        try:
            await langfuse_client.cleanup()
        except Exception as err:
            LOGGER.warning("Error cleaning up Langfuse client: %s", err)

    # Remove data
    if DOMAIN in hass.data and entry.entry_id in hass.data[DOMAIN]:
        hass.data[DOMAIN].pop(entry.entry_id)
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
