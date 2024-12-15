"""Constants for the Custom Conversation integration."""

import logging

DOMAIN = "custom_conversation"
LOGGER = logging.getLogger(__package__)

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
CONF_ENABLE_HASS_AGENT = "enable_home_assistant_agent"
CONF_ENABLE_LLM_AGENT = "enable_llm_agent"
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
