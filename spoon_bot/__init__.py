"""
Spoon-Bot: Local-first AI agent with native OS tools.

A nanobot-style local agent product focused on OS-level interactions,
powered by spoon-core as the underlying engine.

spoon-bot requires spoon-core SDK for all LLM, MCP, and skill functionality.
Install with: pip install spoon-ai
"""

__version__ = "0.1.0"
__author__ = "XSpoon Team"

# Check for spoon-core SDK availability
_SPOON_CORE_AVAILABLE = False
_SPOON_CORE_ERROR = None

try:
    from spoon_ai.chat import ChatBot
    from spoon_ai.schema import Message
    from spoon_ai.llm.interface import LLMResponse
    from spoon_ai.agents.spoon_react_mcp import SpoonReactMCP
    from spoon_ai.agents.spoon_react_skill import SpoonReactSkill
    from spoon_ai.tools import BaseTool, ToolManager
    from spoon_ai.tools.mcp_tool import MCPTool
    from spoon_ai.skills import SkillManager

    _SPOON_CORE_AVAILABLE = True

    # Import spoon-bot components that require spoon-core
    from spoon_bot.core import SpoonBot, SpoonBotConfig, create_agent as create_spoon_bot
    from spoon_bot.agent.loop import AgentLoop, create_agent

except ImportError as e:
    _SPOON_CORE_ERROR = e
    # Set placeholders for spoon-core types when not available
    ChatBot = None
    Message = None
    LLMResponse = None
    SpoonReactMCP = None
    SpoonReactSkill = None
    BaseTool = None
    ToolManager = None
    MCPTool = None
    SkillManager = None
    SpoonBot = None
    SpoonBotConfig = None
    create_spoon_bot = None
    AgentLoop = None
    create_agent = None

# Import spoon-bot components that don't require spoon-core
from spoon_bot.agent.tools.base import Tool
from spoon_bot.exceptions import (
    SpoonBotError,
    ConfigurationError,
    APIKeyMissingError,
    ProviderNotAvailableError,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMResponseError,
    MCPError,
    MCPConnectionError,
    MCPServerNotFoundError,
    MCPToolExecutionError,
    SkillError,
    SkillNotFoundError,
    SkillActivationError,
    SkillPrerequisiteError,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolTimeoutError,
    DependencyError,
    user_friendly_error,
)


def require_spoon_core() -> None:
    """Check that spoon-core SDK is available. Raises ImportError if not."""
    if not _SPOON_CORE_AVAILABLE:
        raise ImportError(
            "spoon-bot requires spoon-core SDK. Install with: pip install spoon-ai"
        ) from _SPOON_CORE_ERROR


def is_spoon_core_available() -> bool:
    """Check if spoon-core SDK is available."""
    return _SPOON_CORE_AVAILABLE


__all__ = [
    # Version
    "__version__",
    # Availability checks
    "require_spoon_core",
    "is_spoon_core_available",
    # Main classes (spoon-bot) - may be None if spoon-core not installed
    "SpoonBot",
    "SpoonBotConfig",
    "AgentLoop",
    "Tool",
    # Factory functions - may be None if spoon-core not installed
    "create_agent",
    "create_spoon_bot",
    # spoon-core SDK types (re-exported for convenience) - may be None
    "ChatBot",
    "Message",
    "LLMResponse",
    "SpoonReactMCP",
    "SpoonReactSkill",
    "BaseTool",
    "ToolManager",
    "MCPTool",
    "SkillManager",
    # Exceptions
    "SpoonBotError",
    "ConfigurationError",
    "APIKeyMissingError",
    "ProviderNotAvailableError",
    "LLMError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "LLMResponseError",
    "MCPError",
    "MCPConnectionError",
    "MCPServerNotFoundError",
    "MCPToolExecutionError",
    "SkillError",
    "SkillNotFoundError",
    "SkillActivationError",
    "SkillPrerequisiteError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "DependencyError",
    "user_friendly_error",
]
