"""Conversation support for Custom Conversation APIs."""

from collections.abc import Callable
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Union, cast

from langfuse.decorators import langfuse_context, observe

if TYPE_CHECKING:
    from langfuse.types import PromptClient
from litellm import OpenAIError, RateLimitError, completion
from litellm.types.completion import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
)
from litellm.types.llms.openai import ChatCompletionToolParam, Function
from litellm.types.utils import Message
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation.chat_log import async_get_chat_log
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_API_KEY, CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import chat_session, device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import CustomConversationConfigEntry
from .api import CustomLLMAPI, IntentTool
from .cc_llm import async_update_llm_data
from .const import (
    CONF_AGENTS_SECTION,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_ENABLE_HASS_AGENT,
    CONF_ENABLE_LLM_AGENT,
    CONF_LANGFUSE_HOST,
    CONF_LANGFUSE_PUBLIC_KEY,
    CONF_LANGFUSE_SECRET_KEY,
    CONF_LANGFUSE_SECTION,
    CONF_LANGFUSE_TAGS,
    CONF_LANGFUSE_TRACING_ENABLED,
    CONF_LLM_PARAMETERS_SECTION,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONVERSATION_ENDED_EVENT,
    CONVERSATION_ERROR_EVENT,
    CONVERSATION_STARTED_EVENT,
    DOMAIN,
    HOME_ASSISTANT_AGENT,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)
from .prompt_manager import PromptManager

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: CustomConversationConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    langfuse_client = None
    if DOMAIN in hass.data and config_entry.entry_id in hass.data[DOMAIN]:
        langfuse_client = hass.data[DOMAIN][config_entry.entry_id].get(
            "langfuse_client"
        )
    prompt_manager = PromptManager(hass)
    if langfuse_client:
        prompt_manager.set_langfuse_client(langfuse_client)
    agent = CustomConversationEntity(config_entry, prompt_manager, hass)
    async_add_entities([agent])


def _format_tool(
    tool: IntentTool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = {
        "name": tool.name,
        "parameters": convert(tool.parameters, custom_serializer=custom_serializer),
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


def _message_convert(message: Message) -> ChatCompletionMessageParam:
    """Convert from class to TypedDict."""
    tool_calls: list[ChatCompletionMessageToolCallParam] = []
    if message.tool_calls:
        tool_calls = [
            ChatCompletionMessageToolCallParam(
                id=tool_call.id,
                function=Function(
                    arguments=tool_call.function.arguments,
                    name=tool_call.function.name,
                ),
                type=tool_call.type,
            )
            for tool_call in message.tool_calls
        ]
    param = ChatCompletionAssistantMessageParam(
        role=message.role,
        content=message.content,
    )
    if tool_calls:
        param["tool_calls"] = tool_calls
    return param


def _chat_message_convert(
    message: conversation.Content
    | conversation.NativeContent[ChatCompletionMessageParam],
    agent_id: str | None,
) -> ChatCompletionMessageParam:
    """Convert any native chat message for this agent to the native format."""
    if message.role == "native":
        # mypy doesn't understand that checking role ensures content type
        return message.content  # type: ignore[return-value]
    return cast(
        ChatCompletionMessageParam,
        {
            "role": message.role,
            "content": message.content,
        },
    )


class CustomConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Custom conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self,
        entry: CustomConversationConfigEntry,
        prompt_manager: PromptManager,
        hass: HomeAssistant,
    ) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.hass = hass
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Custom",
            model="Custom Conversation",
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if self.entry.options.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )
        self.prompt_manager = prompt_manager
        if entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_LANGFUSE_TRACING_ENABLED, False
        ):
            try:
                hass.async_add_executor_job(
                    lambda: langfuse_context.configure(
                        host=entry.options[CONF_LANGFUSE_SECTION][CONF_LANGFUSE_HOST],
                        public_key=entry.options[CONF_LANGFUSE_SECTION][
                            CONF_LANGFUSE_PUBLIC_KEY
                        ],
                        secret_key=entry.options[CONF_LANGFUSE_SECTION][
                            CONF_LANGFUSE_SECRET_KEY
                        ],
                    )
                )
            except ValueError as e:
                LOGGER.error("Error configuring langfuse: %s", e)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    @observe(name="cc_async_process")
    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        return await self._async_handle_message(user_input)

    @observe(name="cc_handle_message")
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Process enabled agents started with the built in agent."""
        LOGGER.debug("Processing user input: %s", user_input)
        assert user_input.agent_id
        options = self.entry.options
        device_registry = dr.async_get(self.hass)
        device = device_registry.async_get(user_input.device_id)
        device_data = {
            "device_id": user_input.device_id,
            "device_name": device.name if device else "Unknown",
            "device_area": device.area_id if device else "Unknown",
        }
        device_tags = [
            f"device_id:{device_data['device_id']}",
            f"device_name:{device_data['device_name']}",
            f"device_area:{device_data['device_area']}",
        ]
        user_configured_tags = options.get(CONF_LANGFUSE_SECTION, {}).get(
            CONF_LANGFUSE_TAGS, []
        )
        new_tags = user_configured_tags + device_tags

        langfuse_context.update_current_trace(tags=new_tags)
        event_data = {
            "agent_id": user_input.agent_id,
            "conversation_id": user_input.conversation_id,
            "language": user_input.language,
            "device_id": user_input.device_id,
            "device_name": device_data["device_name"],
            "device_area": device_data["device_area"],
            "text": user_input.text,
            "user_id": user_input.context.user_id,
        }

        self.hass.bus.async_fire(CONVERSATION_STARTED_EVENT, event_data)
        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_error(
            intent.IntentResponseErrorCode.UNKNOWN,
            "Sorry, there are no enabled Agents",
        )
        result = conversation.ConversationResult(
            response=intent_response, conversation_id=user_input.conversation_id
        )

        if options.get(CONF_AGENTS_SECTION, {}).get(CONF_ENABLE_HASS_AGENT):
            LOGGER.debug("Processing with Home Assistant agent")
            async with (
                chat_session.async_get_chat_session(
                    self.hass, user_input
                ) as session,
                async_get_chat_log(self.hass, session, user_input) as chat_log,
            ):
                result = await self._async_handle_message_with_hass(user_input)
                LOGGER.debug("Received response: %s", result.response.speech)
                if result.response.error_code is None:
                    await self._async_fire_conversation_ended(
                        result,
                        HOME_ASSISTANT_AGENT,
                        user_input,
                        device_data=device_data,
                    )
                    new_tags = ["handling_agent:home_assistant"]
                    if result.response.intent.intent_type is not None:
                        new_tags.append(f"intent:{result.response.intent.intent_type}")
                    if len(result.response.success_results) > 0:
                        for success_result in result.response.success_results:
                            new_tags.append(f"affected_entity:{success_result.id}")
                    langfuse_context.update_current_observation(output=result.as_dict())
                    langfuse_context.update_current_trace(tags=new_tags)
                    chat_log.async_add_message(
                        conversation.Content(
                            role="assistant",
                            agent_id=user_input.agent_id,
                            content=result.response.speech,
                            native=result.response,
                        )
                    )
                    return conversation.ConversationResult(
                        response=result.response,
                        conversation_id=session.conversation_id,
                    )

        if options.get(CONF_AGENTS_SECTION, {}).get(CONF_ENABLE_LLM_AGENT):
            LOGGER.debug("Processing with LLM agent")
            try:
                async with (
                    chat_session.async_get_chat_session(
                        self.hass, user_input
                    ) as session,
                    async_get_chat_log(self.hass, session, user_input) as chat_log,
                ):
                    result, llm_data = await self._async_handle_message_with_llm(
                        user_input, session
                    )
                    LOGGER.debug("Received response: %s", result.response.speech)
                    if result.response.error_code is None:
                        await self._async_fire_conversation_ended(
                            result,
                            "LLM",
                            user_input,
                            llm_data=llm_data,
                            device_data=device_data,
                        )
                        langfuse_context.update_current_trace(
                            tags=["handling_agent:llm"]
                        )
                    else:
                        await self._async_fire_conversation_error(
                            result.response.error_code,
                            "LLM",
                            user_input,
                            device_data=device_data,
                        )
            except RateLimitError as err:
                error_message = getattr(err, "body", str(err))
                await self._async_fire_conversation_error(
                    error_message,
                    "LLM",
                    user_input,
                    device_data=device_data,
                )
                raise HomeAssistantError("Rate limited or insufficient funds") from err
            except OpenAIError as err:
                error_message = getattr(err, "body", str(err))
                await self._async_fire_conversation_error(
                    error_message,
                    "LLM",
                    user_input,
                    device_data=device_data,
                )
                raise HomeAssistantError("Error talking to OpenAI API") from err
        return result

    @observe(name="cc_handle_message_with_hass")
    async def _async_handle_message_with_hass(
        self,
        user_input: conversation.ConversationInput,
    ) -> conversation.ConversationResult:
        """Process a sentence with the Home Assistant agent."""
        hass_agent = conversation.async_get_agent(self.hass, HOME_ASSISTANT_AGENT)
        if hass_agent is None:
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I had a problem talking to Home Assistant",
            )
            langfuse_context.update_current_observation(
                output=intent_response.as_dict()
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=user_input.conversation_id
            )
        response = await hass_agent.async_process(user_input)
        if response.response.intent:
            if response.response.intent.intent_type is not None:
                LOGGER.debug(
                    "Hass agent handled intent_type: %s",
                    response.response.intent.intent_type,
                )
            if response.response.intent.slots is not None:
                LOGGER.debug(
                    "Hass agent handled intent with slots: %s",
                    response.response.intent.slots,
                )
        if response.response.response_type is not None:
            LOGGER.debug(
                "Hass agent returned response_type: %s",
                response.response.response_type,
            )
        if response.response.error_code is not None:
            LOGGER.debug(
                "Hass agent responded with error_code: %s", response.response.error_code
            )
        langfuse_context.update_current_observation(output=response.as_dict())
        return response

    @observe(name="cc_handle_message_with_llm")
    async def _async_handle_message_with_llm(
        self,
        user_input: conversation.ConversationInput,
        session: conversation.ChatLog[ChatCompletionMessageParam],
    ) -> tuple[conversation.ConversationResult, dict]:
        """Process a sentence with the llm."""

        try:
            prompt_object = await async_update_llm_data(
                self.hass,
                user_input,
                self.entry,
                session,
                self.prompt_manager,
                self.entry.options.get(CONF_LLM_HASS_API),
            )

        except conversation.ConverseError as err:
            return err.as_conversation_result()
        options = self.entry.options
        intent_response = intent.IntentResponse(language=user_input.language)
        llm_api: CustomLLMAPI | None = None
        tools: list[ChatCompletionToolParam] | None = None
        if session.llm_api:
            tools = [
                _format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
            ]
        messages = [
            _chat_message_convert(message) for message in session.async_get_messages()
        ]
        # To prevent infinite loops, we limit the number of iterations
        langfuse_context.update_current_observation(prompt=prompt_object)
        llm_details = {}
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                result = await self._async_generate_completion(
                    config_options=options,
                    messages=messages,
                    tools=tools,
                    conversation_id=session.conversation_id,
                    prompt=prompt_object,
                )
                # Assistant-role responses have the content field set to None, but Google's OpenAI compatible endpoint can't handle that
                # and returns an error. So we set it to an empty string.
                if (
                    result.choices[0].message.role == "assistant"
                    and result.choices[0].message.content is None
                ):
                    result.choices[0].message.content = ""

                LOGGER.debug("LLM API response: %s", result)
            except RateLimitError as err:
                LOGGER.error("Rate limit error: %s", err)
                # Re-raise the error so the caller can handle it and fire a message on the event bus
                raise
            except OpenAIError as err:
                LOGGER.error("Error talking to OpenAI: %s", err)
                # Re-raise the error so the caller can handle it and fire a message on the event bus
                raise

            LOGGER.debug("Response %s", result)
            response = result.choices[0].message
            messages.append(_message_convert(response))

            session.async_add_message(
                conversation.Content(
                    role=response.role,
                    agent_id=user_input.agent_id,
                    content=response.content or "",
                ),
            )

            if not response.tool_calls or not session.llm_api:
                break

            for tool_call in response.tool_calls:
                tool_input = llm.ToolInput(
                    tool_name=tool_call.function.name,
                    tool_args=json.loads(tool_call.function.arguments),
                )
                tool_call_data = {
                    "tool_name": tool_call.function.name,
                    "tool_args": tool_call.function.arguments,
                }
                LOGGER.debug(
                    "Tool call: %s(%s)", tool_input.tool_name, tool_input.tool_args
                )

                try:
                    tool_response = await session.async_call_tool(tool_input)
                    # Save a copy of the tool response before deleting any card data to save tokens
                    tool_call_data["tool_response"] = tool_response.copy()
                    # Tag langfuse traces with the intent as the tool call, and the success response as affected entities,
                    # matching the way it's done for hass-handled intents
                    new_tags = ["intent:" + tool_input.tool_name]
                    if tool_response.get("data", {}).get("success"):
                        for entity in tool_response["data"]["success"]:
                            new_tags.extend([f"affected_entity:{entity['id']}"])
                    langfuse_context.update_current_trace(tags=new_tags)
                    if tool_response.get("card"):
                        del tool_response["card"]
                except (HomeAssistantError, vol.Invalid) as e:
                    tool_response = {"error": type(e).__name__}
                    if str(e):
                        tool_response["error_text"] = str(e)

                LOGGER.debug("Tool response: %s", tool_response)
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=tool_call.id,
                        content=json.dumps(tool_response),
                    )
                )
                if "tool_calls" not in llm_details:
                    llm_details["tool_calls"] = []
                llm_details["tool_calls"].append(tool_call_data)

                session.async_add_message(
                    conversation.NativeContent(
                        agent_id=user_input.agent_id,
                        content=messages[-1],
                    )
                )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response.content or "")
        return conversation.ConversationResult(
            response=intent_response, conversation_id=session.conversation_id
        ), llm_details

    @observe(name="cc_generate_completion")
    async def _async_generate_completion(
        self,
        config_options: MappingProxyType[str, Any],
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None,
        conversation_id: str,
        prompt: Union["PromptClient", None] = None,
    ) -> Any:
        """Generate a completion from the LLM."""
        generation_id = langfuse_context.get_current_observation_id()
        existing_trace_id = langfuse_context.get_current_trace_id()
        return await self.hass.async_add_executor_job(
            lambda: completion(
                api_key=self.entry.data.get(CONF_API_KEY),
                base_url=self.entry.data.get(CONF_BASE_URL),
                model=f"openai/{config_options.get(CONF_LLM_PARAMETERS_SECTION, {}).get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)}",
                messages=messages,
                tools=tools,
                max_tokens=config_options.get(CONF_LLM_PARAMETERS_SECTION, {}).get(
                    CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                ),
                top_p=config_options.get(CONF_LLM_PARAMETERS_SECTION, {}).get(
                    CONF_TOP_P, RECOMMENDED_TOP_P
                ),
                temperature=config_options.get(CONF_LLM_PARAMETERS_SECTION, {}).get(
                    CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                ),
                user=conversation_id,
                metadata={
                    "generation_id": generation_id,
                    "existing_trace_id": existing_trace_id,
                    "generation_name": "cc_generate_completion",
                    "prompt": prompt.__dict__ if prompt else None,
                },
                langfuse_secret_key=config_options.get(CONF_LANGFUSE_SECTION, {}).get(
                    CONF_LANGFUSE_SECRET_KEY
                ),
                langfuse_public_key=config_options.get(CONF_LANGFUSE_SECTION, {}).get(
                    CONF_LANGFUSE_PUBLIC_KEY
                ),
                langfuse_host=config_options.get(CONF_LANGFUSE_SECTION, {}).get(
                    CONF_LANGFUSE_HOST
                ),
                callbacks=["langfuse"],
            )
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)

    async def _async_fire_card_requested_event(
        self, conversation_id: str, device_id: str, card: dict
    ) -> None:
        """Fire an event to request a card be displayed."""
        self.hass.bus.async_fire(
            "assistant_card_requested",
            {
                "conversation_id": conversation_id,
                "device_id": device_id,
                "card": card,
            },
        )

    async def _async_fire_conversation_error(
        self,
        error: str,
        agent: str,
        user_input: conversation.ConversationInput,
        device_data: dict | None = None,
    ) -> None:
        """Fire an event to notify that an error occurred."""
        event_data = {
            "agent_id": user_input.agent_id,
            "handling_agent": agent,
            "device_id": user_input.device_id,
            "device_name": device_data.get("device_name") if device_data else "Unknown",
            "device_area": device_data.get("device_area") if device_data else "Unknown",
            "request": user_input.text,
            "error": error,
        }
        self.hass.bus.async_fire(CONVERSATION_ERROR_EVENT, event_data)

    async def _async_fire_conversation_ended(
        self,
        result: dict,
        agent: str,
        user_input: conversation.ConversationInput,
        llm_data: dict | None = None,
        device_data: dict | None = None,
    ) -> None:
        """Fire an event to notify that a conversation has completed."""
        event_data = {
            "agent_id": user_input.agent_id,
            "handling_agent": agent,
            "device_id": user_input.device_id,
            "device_name": device_data.get("device_name") if device_data else "Unknown",
            "device_area": device_data.get("device_area") if device_data else "Unknown",
            "request": user_input.text,
            "result": result.as_dict(),
        }
        if llm_data:
            # If there's any card in the llm_data, we attach one to the response
            if any(
                "card" in tool_call.get("tool_response", {})
                for tool_call in llm_data.get("tool_calls", [])
            ):
                event_data["result"]["response"]["card"] = choose_card(
                    llm_data["tool_calls"]
                )
            event_data["llm_data"] = llm_data
            # If any of the tool calls has data matching intent entities, we attach it to the response
            data_dict = {"targets": [], "success": [], "failed": []}
            for tool_call in llm_data.get("tool_calls", []):
                tool_response = tool_call.get("tool_response", {}).get("data", {})
                for field in ("targets", "success", "failed"):
                    if values := tool_response.get(field, False):
                        data_dict[field].extend(values)
            event_data["result"]["response"]["data"].update(data_dict)
        self.hass.bus.async_fire(CONVERSATION_ENDED_EVENT, event_data)


def choose_card(tool_calls):
    """Choose the most likely card from the tool calls."""
    # It's possible that multiple tools have requested cards, but we only want to show one. For now, we'll choose the last tool call that has a card response.
    filtered_tool_calls = [
        tool_call
        for tool_call in tool_calls
        if isinstance(tool_call.get("tool_response"), dict)
        and "card" in tool_call["tool_response"]
    ]
    if filtered_tool_calls:
        return filtered_tool_calls[-1]["tool_response"]["card"]
    return None
