# ViewAssist Dashboard Configuration for Conversation Rating

This file contains the configuration snippets that need to be added to your ViewAssist dashboard
to enable conversation rating functionality. Add these sections to your existing dashboard configuration.

## Variables Section

Add this to your dashboard's variables section:

```yaml
var_mic_device: |-
  [[[
    try
    {
      var micdevice_assistbid = hass.states[variables.assist_group].attributes.entity_id
        .find((eid) => hass.states[eid].attributes.browser_id === localStorage.getItem("browser_mod-browser-id")) ?? variables.default_satellite;
      var micdevice = hass.states[micdevice_assistbid].attributes.mic_device;
      return `${micdevice}`;
    } catch { return  ""}
  ]]]
```

## Icon Templates Section

Add these to your dashboard's icon templates section:

```yaml
thumbs_up: # Don't change this name, as it's referenced in the automation.
  type: custom:button-card
  template: icon_template
  icon: mdi:thumb-up # Can be changed to another icon if desired.
  tap_action:
    action: call-service
    service: script.score_conversation_from_va_device # This may be different depending on the name you gave the script when you saved it
    service_data:
      config_entry: YOUR_CONFIG_ENTRY_ID  # Replace with your actual config entry ID.
      assist_entity: '[[[ return variables.var_mic_device ]]]'
      score: positive
      entity_id: '[[[ return variables.var_assistsat_entity ]]]'

thumbs_down:
  type: custom:button-card
  template: icon_template
  icon: mdi:thumb-down
  tap_action:
    action: call-service
    service: script.score_conversation_from_va_device # This may be different depending on the name you gave the script when you saved it
    service_data:
      config_entry: YOUR_CONFIG_ENTRY_ID  # Replace with your actual config entry ID. 
      assist_entity: '[[[ return variables.var_mic_device ]]]'
      score: negative
      entity_id: '[[[ return variables.var_assistsat_entity ]]]'
```

Replace `YOUR_CONFIG_ENTRY_ID` with the config entry ID for your Custom Conversation integration.
This can be found in the URL after clicking "configure" on the Integration page. Make sure that the script in the service call matches
the name of your actual script.

## Notes

- The `var_mic_device` variable retrieves the microphone device associated with the current ViewAssist satellite.
- The `thumbs_up` and `thumbs_down` templates create buttons that call the conversation rating script with the appropriate parameters.
