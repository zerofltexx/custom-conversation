"""Conversation support for Custom Conversation APIs."""

from collections.abc import Callable
import json
from typing import Any, Literal

from langfuse.decorators import langfuse_context, observe

# import openai
from langfuse.openai import openai
from openai._types import NOT_GIVEN
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params import FunctionDefinition
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components import assist_pipeline, conversation
from homeassistant.components.conversation import trace
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, TemplateError
from homeassistant.helpers import device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import ulid as ulid_util

from . import CustomConversationConfigEntry
from .api import CustomLLMAPI, IntentTool
from .const import (
    CONF_AGENTS_SECTION,
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
    CONVERSATION_STARTED_EVENT,
    DOMAIN,
    HOME_ASSISTANT_AGENT,
    LLM_API_ID,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)
from .prompt_manager import PromptContext, PromptManager

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
    agent = CustomConversationEntity(config_entry, prompt_manager)
    async_add_entities([agent])


def _format_tool(
    tool: IntentTool, custom_serializer: Callable[[Any], Any] | None
) -> ChatCompletionToolParam:
    """Format tool specification."""
    tool_spec = FunctionDefinition(
        name=tool.name,
        parameters=convert(tool.parameters, custom_serializer=custom_serializer),
    )
    if tool.description:
        tool_spec["description"] = tool.description
    return ChatCompletionToolParam(type="function", function=tool_spec)


class CustomConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Custom conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(
        self, entry: CustomConversationConfigEntry, prompt_manager: PromptManager
    ) -> None:
        """Initialize the agent."""
        self.entry = entry
        self.history: dict[str, list[ChatCompletionMessageParam]] = {}
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
                langfuse_context.configure(
                    host=entry.options[CONF_LANGFUSE_SECTION][CONF_LANGFUSE_HOST],
                    public_key=entry.options[CONF_LANGFUSE_SECTION][
                        CONF_LANGFUSE_PUBLIC_KEY
                    ],
                    secret_key=entry.options[CONF_LANGFUSE_SECTION][
                        CONF_LANGFUSE_SECRET_KEY
                    ],
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

    @observe(name="cc_process")
    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process enabled agents started with the built in agent."""
        LOGGER.debug("Processing user input: %s", user_input)
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
        user_configured_tags = self.entry.options.get(CONF_LANGFUSE_SECTION, {}).get(
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

        if self.entry.options.get(CONF_AGENTS_SECTION, {}).get(CONF_ENABLE_HASS_AGENT):
            LOGGER.debug("Processing with Home Assistant agent")
            result = await self._async_process_hass(user_input)
            LOGGER.debug("Received response: %s", result.response.speech)
            if result.response.error_code is None:
                await self._async_fire_conversation_ended(
                    result, HOME_ASSISTANT_AGENT, user_input, device_data=device_data
                )
                new_tags = ["handling_agent:home_assistant"]
                if result.response.intent.intent_type is not None:
                    new_tags.append(f"intent:{result.response.intent.intent_type}")
                if len(result.response.success_results) > 0:
                    for success_result in result.response.success_results:
                        new_tags.append(f"affected_entity:{success_result.id}")
                langfuse_context.update_current_observation(output=result.as_dict())
                langfuse_context.update_current_trace(tags=new_tags)
                return result

        if self.entry.options.get(CONF_AGENTS_SECTION, {}).get(CONF_ENABLE_LLM_AGENT):
            LOGGER.debug("Processing with LLM agent")
            result, llm_data = await self._async_process_llm(user_input)
            LOGGER.debug("Received response: %s", result.response.speech)
            if result.response.error_code is None:
                await self._async_fire_conversation_ended(
                    result,
                    "LLM",
                    user_input,
                    llm_data=llm_data,
                    device_data=device_data,
                )
                langfuse_context.update_current_trace(tags=["handling_agent:llm"])
        return result

    @observe(name="cc_process_hass")
    async def _async_process_hass(
        self, user_input: conversation.ConversationInput
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
        langfuse_context.update_current_observation(output=response.as_dict())
        return response

    @observe(name="cc_process_llm")
    async def _async_process_llm(
        self, user_input: conversation.ConversationInput
    ) -> tuple[conversation.ConversationResult, dict]:
        """Process a sentence with the llm."""
        options = self.entry.options
        intent_response = intent.IntentResponse(language=user_input.language)
        llm_api: CustomLLMAPI | None = None
        tools: list[ChatCompletionToolParam] | None = None
        user_name: str | None = None
        llm_context = llm.LLMContext(
            platform=DOMAIN,
            context=user_input.context,
            user_prompt=user_input.text,
            language=user_input.language,
            assistant=conversation.DOMAIN,
            device_id=user_input.device_id,
        )
        llm_details = {}
        if (
            user_input.context
            and user_input.context.user_id
            and (
                user := await self.hass.auth.async_get_user(user_input.context.user_id)
            )
        ):
            user_name = user.name
        if llm_api := options.get(CONF_LLM_HASS_API):
            try:
                if llm_api == LLM_API_ID:
                    api_instance = CustomLLMAPI(
                        self.hass, user_name, conversation_config_entry=self.entry
                    )
                    if (
                        langfuse_client := self.hass.data.get(DOMAIN, {})
                        .get(self.entry.entry_id, {})
                        .get("langfuse_client")
                    ):
                        api_instance.set_langfuse_client(langfuse_client)
                    llm_api = await api_instance.async_get_api_instance(llm_context)
                else:
                    llm_api = await llm.async_get_api(
                        self.hass,
                        llm_api,
                        llm_context,
                    )
            except HomeAssistantError as err:
                LOGGER.error("Error getting LLM API: %s", err)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    "Error preparing LLM API",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )
            tools = [
                _format_tool(tool, llm_api.custom_serializer) for tool in llm_api.tools
            ]

        if user_input.conversation_id is None:
            conversation_id = ulid_util.ulid_now()
            messages = []

        elif user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            messages = self.history[conversation_id]

        else:
            # Conversation IDs are ULIDs. We generate a new one if not provided.
            # If an old OLID is passed in, we will generate a new one to indicate
            # a new conversation was started. If the user picks their own, they
            # want to track a conversation and we respect it.
            try:
                ulid_util.ulid_to_bytes(user_input.conversation_id)
                conversation_id = ulid_util.ulid_now()
            except ValueError:
                conversation_id = user_input.conversation_id

            messages = []
        prompt_object = None
        try:
            prompt_context = PromptContext(
                hass=self.hass,
                ha_name=self.hass.config.location_name,
                user_name=user_name,
            )
            if isinstance(llm_api.api, CustomLLMAPI):
                # The LLM API is the CustomLLMAPI, so use its prompt. The prompt manager
                # will pull in the  base prompt if langfuse is disabled.
                prompt = await llm_api.api_prompt
                # If langfuse is successfully used, we'll get back a tuple that contains a prompt object as well
                if isinstance(prompt, tuple):
                    prompt_object, prompt = prompt
            elif not llm_api:
                # No API is enabled - just get the base prompt
                prompt = await self.prompt_manager.async_get_base_prompt(
                    prompt_context, self.entry
                )
            else:
                # We're using a different API, so we need to combine the base prompt with the API prompt
                base_prompt = await self.prompt_manager.async_get_base_prompt(
                    prompt_context, self.entry
                )
                prompt_parts = [base_prompt]
                prompt_parts.append(llm_api.api_prompt)
                prompt = "\n".join(prompt_parts)

        except TemplateError as err:
            LOGGER.error("Error rendering prompt: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Sorry, I had a problem with my template",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        # Create a copy of the variable because we attach it to the trace
        messages = [
            ChatCompletionSystemMessageParam(role="system", content=prompt),
            *messages[1:],
            ChatCompletionUserMessageParam(role="user", content=user_input.text),
        ]

        LOGGER.debug("Prompt: %s", messages)
        LOGGER.debug("Tools: %s", tools)
        trace.async_conversation_trace_append(
            trace.ConversationTraceEventType.AGENT_DETAIL,
            {"messages": messages, "tools": llm_api.tools if llm_api else None},
        )

        client = self.entry.runtime_data
        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            try:
                result = await client.chat.completions.create(
                    model=options.get(CONF_LLM_PARAMETERS_SECTION, {}).get(
                        CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
                    ),
                    messages=messages,
                    tools=tools or NOT_GIVEN,
                    max_tokens=options.get(CONF_LLM_PARAMETERS_SECTION, {}).get(
                        CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS
                    ),
                    top_p=options.get(CONF_LLM_PARAMETERS_SECTION, {}).get(
                        CONF_TOP_P, RECOMMENDED_TOP_P
                    ),
                    temperature=options.get(CONF_LLM_PARAMETERS_SECTION, {}).get(
                        CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE
                    ),
                    user=conversation_id,
                    langfuse_prompt=prompt_object,
                )
            except openai.OpenAIError as err:
                LOGGER.error("Error talking to OpenAI: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    "Sorry, I had a problem talking to OpenAI",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )

            LOGGER.debug("Response %s", result)
            response = result.choices[0].message

            def message_convert(
                message: ChatCompletionMessage,
            ) -> ChatCompletionMessageParam:
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

            messages.append(message_convert(response))
            tool_calls = response.tool_calls

            if not tool_calls or not llm_api:
                break

            for tool_call in tool_calls:
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
                    tool_response = await llm_api.async_call_tool(tool_input)
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

        self.history[conversation_id] = messages

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(response.content or "")
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id
        ), llm_details

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
                "card" in tool_call["tool_response"]
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
    # It's possible that multiple tools have requested cards, but we only want to show one. For now, we'll choose the last tool call that has a card respose.
    tool_calls = [
        tool_call for tool_call in tool_calls if "card" in tool_call["tool_response"]
    ]
    return tool_calls[-1]["tool_response"]["card"]
