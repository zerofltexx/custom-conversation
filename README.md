# Custom Conversation
A very customizable version of Conversation Agent for Home Assistant. Based on the original OpenAI component, this project provides a conversation component and an LLM API that differ from the original in a few key areas:
* It takes a baseurl, so that any OpenAI API-compatible LLM can be used.
* It retains the "card" parameter from intent scripts. The internal LLM API deletes this from tool call responses, in order to save on tokens sent to the LLM. This project removes it from future interactions with the LLM specifically, but retains it so that it can be published with events.
* It publishes conversation started and ended events, along with the card of any tools called, to the home assistant event bus.
* If the internal Home Assistant agent is enabled in the component's configuration, it will attempt to resolve requests using the local intent matching system first, and then fall back to the LLM. While Home Assistant does now provide the option for fallback itself, leaving that disabled and enabling the fallback in this component enables the same functionality while retaining equivalent behavior regardless of whether the request is handled by an LLM - an event is published either way, and it indicates how the request was handled.
* The list of built-in intents to not expose to the LLM is configurable - allowing users to bypass the built-in list (although keep in mind, those intents were probably ignored for a reason), or add additional ones.

Example event from a request handled by the LLM:
```
event_type: custom_conversation_conversation_ended
data:
  agent_id: conversation.customconversation
  handling_agent: LLM
  device_id: null
  request: Turn on the non-production switch
  result:
    response:
      speech:
        plain:
          speech: The non-production switch has been turned on.
          extra_data: null
      card: {}
      language: en
      response_type: action_done
      data:
        targets: []
        success: []
        failed: []
    conversation_id: 01JF1FRHX64C0D5XZB56G9BKC7
  llm_data:
    tool_calls:
      - tool_call_id: call_vhpicJtYThVOWHwmjUjNNaFF
        tool_args: "{\"name\":\"Test Switch\",\"domain\":\"input_boolean\"}"
        tool_response:
          speech: {}
          card: {}
          response_type: action_done
          data:
            targets: []
            success:
              - name: Test Switch
                type: entity
                id: input_boolean.test_switch
            failed: []
origin: LOCAL
time_fired: "2024-12-14T02:25:55.426573+00:00"
context:
  id: 01JF1FS772TCB99VDCCA14M10T
  parent_id: null
  user_id: null
```