"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str | None = None
    usage: dict[str, int] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
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
        """
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get the default model for this provider."""
        pass
