"""Base LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypedDict, NotRequired


class MessageDict(TypedDict, total=False):
    """Type for a message in OpenAI format."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]] | None
    name: NotRequired[str]
    tool_calls: NotRequired[list[dict[str, Any]]]
    tool_call_id: NotRequired[str]


class UsageDict(TypedDict, total=False):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: NotRequired[int]


@dataclass(frozen=True)
class ToolCall:
    """
    Represents a tool call from the LLM.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool to execute.
        arguments: Arguments to pass to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate tool call fields."""
        if not self.id:
            raise ValueError("ToolCall id cannot be empty")
        if not self.name:
            raise ValueError("ToolCall name cannot be empty")
        if not isinstance(self.arguments, dict):
            raise ValueError("ToolCall arguments must be a dictionary")


@dataclass
class LLMResponse:
    """
    Response from an LLM call.

    Attributes:
        content: Text content of the response (may be None if only tool calls).
        tool_calls: List of tool calls requested by the LLM.
        model: Model used for this response.
        usage: Token usage statistics.
        finish_reason: Why the response ended (e.g., "stop", "tool_calls").
    """

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str | None = None
    usage: UsageDict | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate response fields."""
        if self.content is not None and not isinstance(self.content, str):
            raise ValueError("LLMResponse content must be a string or None")
        if not isinstance(self.tool_calls, list):
            raise ValueError("LLMResponse tool_calls must be a list")

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        if self.usage is None:
            return 0
        return self.usage.get("total_tokens", 0) or (
            self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)
        )

    def get_content_or_empty(self) -> str:
        """Get content, returning empty string if None."""
        return self.content or ""


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Subclasses must implement:
    - chat: Send a chat completion request
    - get_default_model: Return the default model name

    Example:
        class MyProvider(LLMProvider):
            async def chat(
                self,
                messages: list[MessageDict],
                tools: list[dict[str, Any]] | None = None,
                model: str | None = None,
                **kwargs: Any,
            ) -> LLMResponse:
                # Implementation here
                ...

            def get_default_model(self) -> str:
                return "my-model-v1"
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[MessageDict],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of messages in OpenAI format.
            tools: Optional list of tool definitions.
            model: Optional model override.
            **kwargs: Additional provider-specific arguments.

        Returns:
            LLMResponse with content and/or tool calls.

        Raises:
            ValueError: If messages is empty or invalid.
            httpx.HTTPStatusError: If the API request fails.
        """
        ...

    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get the default model for this provider.

        Returns:
            Model identifier string.
        """
        ...

    def validate_messages(self, messages: list[MessageDict]) -> list[str]:
        """
        Validate messages before sending to the API.

        Args:
            messages: List of messages to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        if not messages:
            errors.append("Messages list cannot be empty")
            return errors

        valid_roles = {"system", "user", "assistant", "tool"}

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"Message {i} must be a dictionary")
                continue

            role = msg.get("role")
            if not role:
                errors.append(f"Message {i} missing 'role' field")
            elif role not in valid_roles:
                errors.append(f"Message {i} has invalid role: {role}")

            # Tool messages must have tool_call_id
            if role == "tool" and not msg.get("tool_call_id"):
                errors.append(f"Tool message {i} missing 'tool_call_id'")

        return errors
