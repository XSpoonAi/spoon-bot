"""Custom exceptions for spoon-bot.

This module provides a hierarchy of exception classes for different error scenarios,
enabling proper error handling and user-friendly error messages.
"""

from __future__ import annotations

from typing import Any


class SpoonBotError(Exception):
    """Base exception for all spoon-bot errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message

    def user_message(self) -> str:
        """Return a user-friendly error message."""
        return self.message


# === Configuration Errors ===


class ConfigurationError(SpoonBotError):
    """Error in configuration or setup."""

    pass


class APIKeyMissingError(ConfigurationError):
    """API key is missing or invalid."""

    def __init__(
        self,
        provider: str,
        env_var: str | None = None,
        message: str | None = None,
    ):
        self.provider = provider
        self.env_var = env_var
        msg = message or f"API key for {provider} is not configured"
        if env_var:
            msg += f". Set the {env_var} environment variable or pass api_key parameter"
        super().__init__(msg, {"provider": provider, "env_var": env_var})

    def user_message(self) -> str:
        if self.env_var:
            return (
                f"The {self.provider} API key is missing. "
                f"Please set the {self.env_var} environment variable."
            )
        return f"The {self.provider} API key is missing."


class ProviderNotAvailableError(ConfigurationError):
    """An LLM provider or service is not available."""

    def __init__(
        self,
        provider: str,
        reason: str | None = None,
        install_hint: str | None = None,
    ):
        self.provider = provider
        self.reason = reason
        self.install_hint = install_hint
        msg = f"Provider '{provider}' is not available"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, {"provider": provider})

    def user_message(self) -> str:
        msg = f"The {self.provider} provider is not available."
        if self.install_hint:
            msg += f" {self.install_hint}"
        return msg


# === LLM Errors ===


class LLMError(SpoonBotError):
    """Base class for LLM-related errors."""

    pass


class LLMConnectionError(LLMError):
    """Error connecting to an LLM provider."""

    def __init__(
        self,
        provider: str,
        status_code: int | None = None,
        response_text: str | None = None,
    ):
        self.provider = provider
        self.status_code = status_code
        self.response_text = response_text
        msg = f"Failed to connect to {provider}"
        if status_code:
            msg += f" (HTTP {status_code})"
        super().__init__(
            msg,
            {"provider": provider, "status_code": status_code},
        )

    def user_message(self) -> str:
        if self.status_code == 401:
            return f"Authentication failed with {self.provider}. Please check your API key."
        if self.status_code == 429:
            return f"Rate limit exceeded for {self.provider}. Please try again later."
        if self.status_code == 503:
            return f"The {self.provider} service is temporarily unavailable."
        if self.status_code and self.status_code >= 500:
            return f"The {self.provider} service encountered an error. Please try again."
        return f"Could not connect to {self.provider}. Please check your network connection."


class LLMResponseError(LLMError):
    """Error in LLM response processing."""

    def __init__(self, message: str, raw_response: Any = None):
        self.raw_response = raw_response
        super().__init__(message, {"has_raw_response": raw_response is not None})


class LLMTimeoutError(LLMError):
    """LLM request timed out."""

    def __init__(self, provider: str, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Request to {provider} timed out after {timeout_seconds}s",
            {"provider": provider, "timeout": timeout_seconds},
        )

    def user_message(self) -> str:
        return "The request took too long to complete. Please try again."


# === MCP Errors ===


class MCPError(SpoonBotError):
    """Base class for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """Error connecting to an MCP server."""

    def __init__(
        self,
        server_name: str,
        transport: str | None = None,
        cause: str | None = None,
    ):
        self.server_name = server_name
        self.transport = transport
        msg = f"Failed to connect to MCP server '{server_name}'"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, {"server": server_name, "transport": transport})

    def user_message(self) -> str:
        return f"Could not connect to the MCP server '{self.server_name}'."


class MCPServerNotFoundError(MCPError):
    """MCP server not found or not configured."""

    def __init__(self, server_name: str):
        self.server_name = server_name
        super().__init__(
            f"MCP server '{server_name}' is not configured",
            {"server": server_name},
        )


class MCPToolExecutionError(MCPError):
    """Error executing an MCP tool."""

    def __init__(
        self,
        tool_name: str,
        server_name: str | None = None,
        cause: str | None = None,
    ):
        self.tool_name = tool_name
        self.server_name = server_name
        msg = f"Error executing MCP tool '{tool_name}'"
        if server_name:
            msg += f" on server '{server_name}'"
        if cause:
            msg += f": {cause}"
        super().__init__(
            msg,
            {"tool": tool_name, "server": server_name},
        )

    def user_message(self) -> str:
        return f"The tool '{self.tool_name}' encountered an error."


# === Skill Errors ===


class SkillError(SpoonBotError):
    """Base class for skill-related errors."""

    pass


class SkillNotFoundError(SkillError):
    """Skill not found."""

    def __init__(self, skill_name: str, available_skills: list[str] | None = None):
        self.skill_name = skill_name
        self.available_skills = available_skills
        msg = f"Skill '{skill_name}' not found"
        if available_skills:
            msg += f". Available skills: {', '.join(available_skills[:5])}"
            if len(available_skills) > 5:
                msg += f" (and {len(available_skills) - 5} more)"
        super().__init__(msg, {"skill": skill_name})


class SkillActivationError(SkillError):
    """Error activating a skill."""

    def __init__(self, skill_name: str, cause: str | None = None):
        self.skill_name = skill_name
        msg = f"Failed to activate skill '{skill_name}'"
        if cause:
            msg += f": {cause}"
        super().__init__(msg, {"skill": skill_name})


class SkillPrerequisiteError(SkillError):
    """Skill prerequisites not met."""

    def __init__(self, skill_name: str, missing_prereqs: list[str]):
        self.skill_name = skill_name
        self.missing_prereqs = missing_prereqs
        super().__init__(
            f"Skill '{skill_name}' requires: {', '.join(missing_prereqs)}",
            {"skill": skill_name, "missing": missing_prereqs},
        )


# === Tool Errors ===


class ToolError(SpoonBotError):
    """Base class for tool-related errors."""

    pass


class ToolNotFoundError(ToolError):
    """Tool not found in registry."""

    def __init__(self, tool_name: str, available_tools: list[str] | None = None):
        self.tool_name = tool_name
        self.available_tools = available_tools
        msg = f"Tool '{tool_name}' not found"
        super().__init__(msg, {"tool": tool_name})


class ToolExecutionError(ToolError):
    """Error during tool execution."""

    def __init__(self, tool_name: str, cause: str):
        self.tool_name = tool_name
        super().__init__(
            f"Tool '{tool_name}' execution failed: {cause}",
            {"tool": tool_name},
        )

    def user_message(self) -> str:
        return f"The tool '{self.tool_name}' encountered an error."


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""

    def __init__(self, tool_name: str, timeout_seconds: float):
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_seconds}s",
            {"tool": tool_name, "timeout": timeout_seconds},
        )


# === File Operation Errors ===


class FileOperationError(SpoonBotError):
    """Base class for file operation errors."""

    pass


class FileNotFoundError_(FileOperationError):
    """File not found (custom to avoid shadowing builtin)."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"File not found: {path}", {"path": path})


class FilePermissionError(FileOperationError):
    """Permission denied for file operation."""

    def __init__(self, path: str, operation: str = "access"):
        self.path = path
        self.operation = operation
        super().__init__(
            f"Permission denied to {operation} file: {path}",
            {"path": path, "operation": operation},
        )


class DirectoryNotFoundError(FileOperationError):
    """Directory not found."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Directory not found: {path}", {"path": path})


# === Session Errors ===


class SessionError(SpoonBotError):
    """Base class for session-related errors."""

    pass


class SessionNotFoundError(SessionError):
    """Session not found."""

    def __init__(self, session_key: str):
        self.session_key = session_key
        super().__init__(
            f"Session '{session_key}' not found",
            {"session": session_key},
        )


# === Dependency Errors ===


class DependencyError(SpoonBotError):
    """Optional dependency not available."""

    def __init__(
        self,
        package: str,
        feature: str | None = None,
        install_command: str | None = None,
    ):
        self.package = package
        self.feature = feature
        self.install_command = install_command
        msg = f"Optional dependency '{package}' is not installed"
        if feature:
            msg += f" (required for {feature})"
        super().__init__(msg, {"package": package})

    def user_message(self) -> str:
        msg = f"The '{self.package}' package is not installed"
        if self.feature:
            msg += f" (needed for {self.feature})"
        if self.install_command:
            msg += f". Install it with: {self.install_command}"
        return msg


# === Utility Functions ===


def format_exception_chain(exc: Exception) -> str:
    """Format an exception chain for logging."""
    messages = []
    current: Exception | None = exc
    while current is not None:
        if isinstance(current, SpoonBotError):
            messages.append(str(current))
        else:
            messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__
    return " -> ".join(messages)


def user_friendly_error(exc: Exception) -> str:
    """Get a user-friendly error message from any exception."""
    if isinstance(exc, SpoonBotError):
        return exc.user_message()
    # Map common exceptions to friendly messages
    if isinstance(exc, ConnectionError):
        return "Network connection error. Please check your internet connection."
    if isinstance(exc, TimeoutError):
        return "The operation timed out. Please try again."
    if isinstance(exc, PermissionError):
        return "Permission denied. Please check file or directory permissions."
    if isinstance(exc, FileNotFoundError):
        return "The requested file was not found."
    # Generic fallback
    return "An unexpected error occurred. Please try again."
