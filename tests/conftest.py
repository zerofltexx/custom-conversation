"""Fixtures for Custom Conversation tests."""
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component, setup_component

from pytest_homeassistant_custom_component.common import MockConfigEntry, MockEntity, MockEntityPlatform

from custom_components.custom_conversation.const import DOMAIN

@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations):
    """Enable custom integrations for testing."""
    yield

@pytest.fixture(autouse=True)
async def mock_default_components(hass: HomeAssistant) -> None:
    """Fixture to setup required default components."""
    assert await async_setup_component(hass, "homeassistant", {})
    assert await async_setup_component(hass, "conversation", {})

@pytest.fixture
def hass(hass: HomeAssistant) -> HomeAssistant:
    """Fixture that provides a fully set up Home Assistant instance."""
    hass.data.update({DOMAIN: {}})    
    return hass

@pytest.fixture
def config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Create a mock config entry."""
    entry = MockConfigEntry(
        title="Test",
        domain=DOMAIN,
        data={
           "api_key": "test-api-key",
           "base_url": "https://api.openai.com/v1",
        },
        options={}
    )
    entry.add_to_hass(hass)
    return entry
