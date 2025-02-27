"""Test helpers that don't import the component."""
from unittest.mock import Mock, patch
import sys

def setup_mocks():
    """Set up all mocks before any component imports."""
    # Create mock modules
    mock_openai = Mock()
    
    # Create full langfuse structure
    mock_langfuse = Mock()
    mock_langfuse.openai = Mock(openai=mock_openai)
    mock_langfuse.model = Mock()
    mock_langfuse.model.Prompt = Mock
    mock_langfuse.decorators = Mock()
    mock_langfuse.api = Mock()
    # Mock the observe decorator to simply return the function
    mock_langfuse.decorators.observe = lambda *args, **kwargs: lambda f: f
    
    # Mock all required langfuse modules
    sys.modules['langfuse'] = mock_langfuse
    sys.modules['langfuse.openai'] = mock_langfuse.openai
    sys.modules['langfuse.openai.openai'] = mock_openai
    sys.modules['langfuse.model'] = mock_langfuse.model
    sys.modules['langfuse.decorators'] = mock_langfuse.decorators
    sys.modules['langfuse.api'] = mock_langfuse.api
    sys.modules['langfuse.api.resources'] = mock_langfuse.api.resources
    sys.modules['langfuse.api.resources.commons'] = mock_langfuse.api.resources.commons
    sys.modules['langfuse.api.resources.commons.types'] = mock_langfuse.api.resources.commons.types