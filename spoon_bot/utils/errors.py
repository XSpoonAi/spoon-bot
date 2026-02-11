"""User-friendly error handling for spoon-bot.

This module provides:
- Custom exception classes for different error types
- Functions to format errors for user display (no stack traces)
- Error categorization for appropriate user feedback
"""

import re
import traceback
from typing import Any


class SpoonBotError(Exception):
    """Base exception for spoon-bot errors."""

    def __init__(
        self,
        message: str,
        user_message: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize SpoonBotError.

        Args:
            message: Technical error message (for logs).
            user_message: User-friendly message (for display).
            details: Additional error details.
        """
        super().__init__(message)
        self.user_message = user_message or message
        self.details = details or {}

    def __str__(self) -> str:
        return self.user_message


class ConfigurationError(SpoonBotError):
    """Error in configuration or setup."""

    pass


class APIError(SpoonBotError):
    """Error communicating with external APIs."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        provider: str | None = None,
        user_message: str | None = None,
    ):
        user_msg = user_message
        if not user_msg:
            if status_code == 401:
                user_msg = f"Authentication failed for {provider or 'API'}. Please check your API key."
            elif status_code == 429:
                user_msg = f"Rate limit exceeded for {provider or 'API'}. Please wait and try again."
            elif status_code == 500:
                user_msg = f"The {provider or 'API'} service is experiencing issues. Please try again later."
            elif status_code == 503:
                user_msg = f"The {provider or 'API'} service is temporarily unavailable."
            else:
                user_msg = f"Error communicating with {provider or 'API'}. Please try again."

        super().__init__(
            message,
            user_message=user_msg,
            details={"status_code": status_code, "provider": provider},
        )
        self.status_code = status_code
        self.provider = provider


class ToolExecutionError(SpoonBotError):
    """Error executing a tool."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        user_message: str | None = None,
    ):
        user_msg = user_message or f"The {tool_name} tool encountered an error. Please try again."
        super().__init__(
            message,
            user_message=user_msg,
            details={"tool_name": tool_name},
        )
        self.tool_name = tool_name


class RateLimitExceeded(SpoonBotError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        resource: str,
        limit: int,
        window: float,
        retry_after: float | None = None,
    ):
        user_msg = f"Too many requests to {resource}. Please wait a moment before trying again."
        if retry_after:
            user_msg += f" (retry in {retry_after:.1f}s)"

        super().__init__(
            f"Rate limit exceeded for {resource}: {limit} requests per {window}s",
            user_message=user_msg,
            details={
                "resource": resource,
                "limit": limit,
                "window": window,
                "retry_after": retry_after,
            },
        )
        self.resource = resource
        self.limit = limit
        self.window = window
        self.retry_after = retry_after


# Common error patterns and their user-friendly messages
ERROR_PATTERNS = [
    # API key errors
    (r"ANTHROPIC_API_KEY", "API key not configured. Please set your ANTHROPIC_API_KEY environment variable."),
    (r"OPENAI_API_KEY", "API key not configured. Please set your OPENAI_API_KEY environment variable."),
    (r"api.?key.*required", "API key is required. Please configure your API key."),
    (r"Invalid API key|api_key.*invalid", "Invalid API key. Please check your API key configuration."),
    # Network errors
    (r"ConnectionError|ConnectTimeout|ConnectionRefused", "Unable to connect to the server. Please check your internet connection."),
    (r"ReadTimeout|TimeoutError", "The request timed out. Please try again."),
    (r"SSLError|SSL.*certificate", "SSL certificate error. Please check your network configuration."),
    # Permission errors
    (r"PermissionError|Permission denied", "Permission denied. Please check file or directory permissions."),
    (r"Access denied", "Access denied. You may not have permission for this operation."),
    # File errors
    (r"FileNotFoundError|No such file", "File or directory not found. Please check the path."),
    (r"IsADirectoryError", "Expected a file but found a directory."),
    (r"NotADirectoryError", "Expected a directory but found a file."),
    # Resource errors
    (r"MemoryError|Out of memory", "Out of memory. Please try with a smaller request."),
    (r"DiskQuotaExceeded", "Disk quota exceeded. Please free up disk space."),
    # JSON errors
    (r"JSONDecodeError|Invalid JSON", "Invalid data format received. Please try again."),
    # Rate limiting
    (r"rate.?limit|too many requests|429", "Rate limit exceeded. Please wait before trying again."),
]


def format_user_error(
    error: Exception,
    include_type: bool = False,
    log_full_error: bool = True,
) -> str:
    """
    Format an exception into a user-friendly error message.

    This function:
    - Converts technical exceptions to readable messages
    - Hides stack traces from users
    - Provides actionable feedback when possible

    Args:
        error: The exception to format.
        include_type: Include error type prefix (e.g., "Configuration Error:").
        log_full_error: Whether to log the full traceback (for debugging).

    Returns:
        User-friendly error message string.
    """
    # Handle our custom exceptions
    if isinstance(error, SpoonBotError):
        prefix = ""
        if include_type:
            if isinstance(error, ConfigurationError):
                prefix = "Configuration Error: "
            elif isinstance(error, APIError):
                prefix = "API Error: "
            elif isinstance(error, ToolExecutionError):
                prefix = "Tool Error: "
            elif isinstance(error, RateLimitExceeded):
                prefix = "Rate Limit: "
            else:
                prefix = "Error: "
        return f"{prefix}{error.user_message}"

    # Get error message
    error_str = str(error)
    error_type = type(error).__name__

    # Check against known patterns
    for pattern, user_message in ERROR_PATTERNS:
        if re.search(pattern, error_str, re.IGNORECASE) or re.search(pattern, error_type, re.IGNORECASE):
            return user_message

    # Default messages for common exception types
    type_messages = {
        "ValueError": "Invalid value provided. Please check your input.",
        "TypeError": "Invalid operation. Please check your input.",
        "KeyError": "Required information is missing.",
        "AttributeError": "An internal error occurred. Please try again.",
        "ImportError": "A required component is not installed.",
        "ModuleNotFoundError": "A required module is not installed.",
        "RuntimeError": "A runtime error occurred. Please try again.",
        "OSError": "A system error occurred. Please try again.",
        "IOError": "An input/output error occurred. Please try again.",
    }

    if error_type in type_messages:
        return type_messages[error_type]

    # For unknown errors, provide a generic message
    # but include a sanitized version of the error for context
    sanitized = _sanitize_error_message(error_str)
    if sanitized and len(sanitized) < 200:
        return f"An error occurred: {sanitized}"

    return "An unexpected error occurred. Please try again or contact support."


def _sanitize_error_message(message: str) -> str:
    """
    Sanitize an error message for user display.

    Removes:
    - File paths (potential security info)
    - Line numbers
    - Technical jargon
    - Long tracebacks

    Args:
        message: Raw error message.

    Returns:
        Sanitized message.
    """
    if not message:
        return ""

    # Remove file paths
    message = re.sub(r'[A-Za-z]:\\[^\s:]+', '[path]', message)  # Windows paths
    message = re.sub(r'/[^\s:]+/[^\s:]+', '[path]', message)    # Unix paths

    # Remove line numbers
    message = re.sub(r', line \d+', '', message)
    message = re.sub(r'line \d+:', '', message)

    # Remove "Traceback" and everything after
    if "Traceback" in message:
        message = message.split("Traceback")[0].strip()

    # Remove stack frame references
    message = re.sub(r'File "[^"]+",', '', message)
    message = re.sub(r'in \w+\(\)', '', message)

    # Clean up whitespace
    message = re.sub(r'\s+', ' ', message).strip()

    return message


def get_error_suggestions(error: Exception) -> list[str]:
    """
    Get actionable suggestions for resolving an error.

    Args:
        error: The exception to analyze.

    Returns:
        List of suggestion strings.
    """
    suggestions = []
    error_str = str(error).lower()
    error_type = type(error).__name__

    # API key suggestions
    if "api_key" in error_str or "api key" in error_str:
        suggestions.append("Check that your API key is correctly set in the environment")
        suggestions.append("Verify the API key has not expired")
        suggestions.append("Ensure there are no extra spaces in the API key")

    # Connection suggestions
    if any(word in error_str for word in ["connection", "timeout", "network"]):
        suggestions.append("Check your internet connection")
        suggestions.append("Try again in a few moments")
        suggestions.append("Check if the service is available")

    # Permission suggestions
    if "permission" in error_str or error_type == "PermissionError":
        suggestions.append("Check file/directory permissions")
        suggestions.append("Try running with appropriate permissions")

    # File not found suggestions
    if "not found" in error_str or error_type == "FileNotFoundError":
        suggestions.append("Verify the file path is correct")
        suggestions.append("Check if the file exists")

    # Rate limit suggestions
    if "rate" in error_str or "limit" in error_str or "429" in error_str:
        suggestions.append("Wait a few minutes before trying again")
        suggestions.append("Consider reducing request frequency")

    # Generic suggestions if nothing specific
    if not suggestions:
        suggestions.append("Try the operation again")
        suggestions.append("Check the input parameters")

    return suggestions
