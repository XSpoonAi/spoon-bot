"""
Spoon-Bot: Local-first AI agent with native OS tools.

A nanobot-style local agent product focused on OS-level interactions,
powered by spoon-core as the underlying engine.
"""

__version__ = "0.1.0"
__author__ = "XSpoon Team"

from spoon_bot.agent.loop import AgentLoop
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

__all__ = [
    # Core classes
    "AgentLoop",
    "Tool",
    "__version__",
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
