"""Constants for the Custom Conversation integration."""

import logging

DOMAIN = "custom_conversation"
LOGGER = logging.getLogger(__package__)

CONF_CHAT_MODEL = "chat_model"
CONF_ENABLE_HASS_AGENT = "enable_home_assistant_agent"
CONF_ENABLE_LLM_AGENT = "enable_llm_agent"
CONF_AGENTS_SECTION = "agents"
CONF_LLM_PARAMETERS_SECTION = "llm_parameters"
CONF_IGNORED_INTENTS_SECTION = "ignored_intents_section"
CONF_IGNORED_INTENTS = "ignored_intents"
RECOMMENDED_CHAT_MODEL = "gpt-4o-mini"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 150
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
CONF_BASE_URL = "base_url"
RECOMMENDED_BASE_URL = "https://api.openai.com/v1"
LLM_API_ID = "custom-conversation"
HOME_ASSISTANT_AGENT = "conversation.home_assistant"
CONVERSATION_STARTED_EVENT = f"{DOMAIN}_conversation_started"
CONVERSATION_ENDED_EVENT = f"{DOMAIN}_conversation_ended"

CONF_CUSTOM_PROMPTS_SECTION = "custom_prompts"
CONF_PROMPT_BASE = "prompt_base"
DEFAULT_BASE_PROMPT = (
    'Current time is {{ now().strftime("%H:%M:%S") }}. '
    'Today\'s date is {{ now().strftime("%Y-%m-%d") }}.\n'
)

CONF_INSTRUCTIONS_PROMPT = "instructions_prompt"
DEFAULT_INSTRUCTIONS_PROMPT = """You are a voice assistant for Home Assistant.
Answer questions about the world truthfully.
Answer in plain text. Keep it simple and to the point.
"""

CONF_PROMPT_NO_ENABLED_ENTITIES = "prompt_no_enabled_entities"
DEFAULT_PROMPT_NO_ENABLED_ENTITIES = (
    "Only if the user wants to control a device, tell them to expose entities "
    "to their voice assistant in Home Assistant."
)

CONF_API_PROMPT_BASE = "prompt_api_base"
DEFAULT_API_PROMPT_BASE = (
    "When controlling Home Assistant always call the intent tools. "
    "Use HassTurnOn to lock and HassTurnOff to unlock a lock. "
    "When controlling a device, prefer passing just name and domain. "
    "When controlling an area, prefer passing just area name and domain."
)

# If the user request came from a device, it may have a known location
CONF_PROMPT_DEVICE_KNOWN_LOCATION = "prompt_device_known_location"
DEFAULT_API_PROMPT_DEVICE_KNOWN_LOCATION = (
    "You are in area {{ location }} and all generic commands "
    "like 'turn on the lights' should target this area."
)
CONF_PROMPT_DEVICE_UNKNOWN_LOCATION = "prompt_device_unknown_location"
DEFAULT_API_PROMPT_DEVICE_UNKNOWN_LOCATION = (
    "When a user asks to turn on all devices of a specific type, "
    "ask user to specify an area, unless there is only one device of that type."
)

CONF_PROMPT_TIMERS_UNSUPPORTED = "prompt_timers_unsupported"
DEFAULT_API_PROMPT_TIMERS_UNSUPPORTED = "This device is not able to start timers."

CONF_PROMPT_EXPOSED_ENTITIES = "prompt_exposed_entities"
DEFAULT_API_PROMPT_EXPOSED_ENTITIES = (
    "An overview of the areas and the devices in this smart home:"
)

# Langfuse Constants
CONF_LANGFUSE_SECTION = "langfuse"
CONF_ENABLE_LANGFUSE = "enable_langfuse"
CONF_LANGFUSE_BASE_PROMPT_ID = "base_prompt_id"
CONF_LANGFUSE_BASE_PROMPT_LABEL = "base_prompt_label"
CONF_LANGFUSE_API_PROMPT_ID = "api_prompt_id"
CONF_LANGFUSE_API_PROMPT_LABEL = "api_prompt_label"
CONF_LANGFUSE_SECRET_KEY = "langfuse_secret_key"
CONF_LANGFUSE_PUBLIC_KEY = "langfuse_public_key"
CONF_LANGFUSE_HOST = "langfuse_host"
CONF_LANGFUSE_TRACING_ENABLED = "langfuse_tracing_enabled"
