"""
LLM module - Uses spoon-core SDK directly.

All LLM functionality is provided by spoon-core's ChatBot and LLMManager.
No local reimplementations - use spoon-core directly.
"""

_SPOON_CORE_AVAILABLE = False

try:
    from spoon_ai.chat import ChatBot
    from spoon_ai.llm import LLMManager
    from spoon_ai.llm.interface import LLMResponse
    from spoon_ai.schema import Message, ToolCall

    _SPOON_CORE_AVAILABLE = True

    __all__ = [
        "ChatBot",
        "LLMManager",
        "LLMResponse",
        "Message",
        "ToolCall",
        "is_available",
    ]

except ImportError:
    # Set placeholders when spoon-core is not available
    ChatBot = None
    LLMManager = None
    LLMResponse = None
    Message = None
    ToolCall = None

    __all__ = ["is_available"]


def is_available() -> bool:
    """Check if LLM functionality is available (spoon-core installed)."""
    return _SPOON_CORE_AVAILABLE
