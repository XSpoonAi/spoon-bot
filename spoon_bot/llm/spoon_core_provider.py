"""LLM provider using spoon-core's ChatBot.

This module provides a bridge between spoon-bot's LLM interface and spoon-core's
ChatBot/LLMManager system, enabling access to multiple providers (anthropic, openai,
deepseek, ollama, gemini, etc.) through a unified interface.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from spoon_bot.llm.base import LLMProvider, LLMResponse, ToolCall

# Flag to track if spoon-core is available
_SPOON_CORE_AVAILABLE = False

try:
    from spoon_ai.chat import ChatBot
    from spoon_ai.schema import Message as CoreMessage, ToolCall as CoreToolCall, Function
    from spoon_ai.llm.interface import LLMResponse as CoreLLMResponse

    _SPOON_CORE_AVAILABLE = True
except ImportError:
    logger.warning(
        "spoon-core (spoon_ai) not installed. SpoonCoreProvider will use fallback mode."
    )
    ChatBot = None
    CoreMessage = None
    CoreToolCall = None
    Function = None
    CoreLLMResponse = None


# Default models for each provider
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "deepseek": "deepseek-chat",
    "ollama": "llama3.2",
    "gemini": "gemini-2.0-flash",
    "openrouter": "anthropic/claude-3.5-sonnet",
}


class SpoonCoreProvider(LLMProvider):
    """Wraps spoon-core's ChatBot for spoon-bot compatibility.

    This provider acts as a bridge between spoon-bot's LLM interface and
    spoon-core's ChatBot system, enabling:
    - Multi-provider support (anthropic, openai, deepseek, ollama, gemini, etc.)
    - Tool calling via ChatBot.ask_tool()
    - Memory management features from spoon-core
    - Graceful fallback if spoon-core is not installed

    Example usage:
        >>> provider = SpoonCoreProvider(provider="anthropic")
        >>> response = await provider.chat(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.content)

        # With tools
        >>> tools = [
        ...     {
        ...         "type": "function",
        ...         "function": {
        ...             "name": "get_weather",
        ...             "description": "Get weather for a location",
        ...             "parameters": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "location": {"type": "string"}
        ...                 }
        ...             }
        ...         }
        ...     }
        ... ]
        >>> response = await provider.chat(messages, tools=tools)
    """

    def __init__(
        self,
        provider: str = "anthropic",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        enable_short_term_memory: bool = False,
        enable_long_term_memory: bool = False,
        **kwargs: Any,
    ):
        """Initialize SpoonCoreProvider.

        Args:
            provider: LLM provider name (anthropic, openai, deepseek, ollama, gemini, openrouter).
            api_key: Optional API key override. If not provided, uses environment variables
                     or spoon-core's configuration system.
            model: Optional model name override. If not provided, uses provider's default.
            base_url: Optional base URL override for the API.
            enable_short_term_memory: Enable spoon-core's short-term memory management.
            enable_long_term_memory: Enable spoon-core's long-term memory (Mem0).
            **kwargs: Additional parameters passed to ChatBot.

        Raises:
            RuntimeError: If spoon-core is not installed and no fallback is available.
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model or DEFAULT_MODELS.get(self.provider)
        self.base_url = base_url
        self._chatbot: Any = None
        self._kwargs = kwargs
        self._enable_short_term_memory = enable_short_term_memory
        self._enable_long_term_memory = enable_long_term_memory

        if not _SPOON_CORE_AVAILABLE:
            logger.warning(
                f"SpoonCoreProvider initialized for '{provider}' but spoon-core "
                "is not available. Operations will raise RuntimeError."
            )
        else:
            self._initialize_chatbot()

    def _initialize_chatbot(self) -> None:
        """Initialize the underlying ChatBot instance."""
        if not _SPOON_CORE_AVAILABLE or ChatBot is None:
            return

        try:
            chatbot_kwargs: dict[str, Any] = {
                "use_llm_manager": True,
                "llm_provider": self.provider,
                "enable_short_term_memory": self._enable_short_term_memory,
                "enable_long_term_memory": self._enable_long_term_memory,
            }

            if self.api_key:
                chatbot_kwargs["api_key"] = self.api_key

            if self.model:
                chatbot_kwargs["model_name"] = self.model

            if self.base_url:
                chatbot_kwargs["base_url"] = self.base_url

            # Merge additional kwargs
            chatbot_kwargs.update(self._kwargs)

            self._chatbot = ChatBot(**chatbot_kwargs)
            logger.info(
                f"SpoonCoreProvider initialized with provider='{self.provider}', "
                f"model='{self.model}'"
            )

        except Exception as e:
            logger.error(f"Failed to initialize ChatBot: {e}")
            raise RuntimeError(f"Failed to initialize SpoonCoreProvider: {e}") from e

    def get_default_model(self) -> str:
        """Get the default model for this provider.

        Returns:
            Default model name string.
        """
        return self.model or DEFAULT_MODELS.get(self.provider, "")

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: List of messages in OpenAI format (role, content).
            tools: Optional list of tool definitions in OpenAI format.
            model: Optional model override for this request.
            **kwargs: Additional provider-specific arguments.

        Returns:
            LLMResponse with content and/or tool calls.

        Raises:
            RuntimeError: If spoon-core is not available.
            Exception: If the chat request fails.
        """
        if not _SPOON_CORE_AVAILABLE or self._chatbot is None:
            raise RuntimeError(
                "spoon-core is not available. Install it with: pip install spoon-ai"
            )

        # Convert messages to spoon-core format
        core_messages = self._convert_messages_to_core(messages)

        # Convert tools to spoon-core format if provided
        core_tools = self._convert_tools_to_core(tools) if tools else None

        try:
            if tools:
                # Use ask_tool for tool-enabled requests
                tool_choice = kwargs.pop("tool_choice", None)
                core_response = await self._chatbot.ask_tool(
                    messages=core_messages,
                    tools=core_tools,
                    tool_choice=tool_choice,
                    **kwargs,
                )
                return self._convert_tool_response_to_spoon_bot(core_response, model)
            else:
                # Use simple ask for regular chat
                content = await self._chatbot.ask(messages=core_messages, **kwargs)
                return LLMResponse(
                    content=content,
                    tool_calls=[],
                    model=model or self.model,
                    usage=None,
                )

        except Exception as e:
            logger.error(f"SpoonCoreProvider chat error: {e}")
            raise

    def _convert_messages_to_core(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to spoon-core compatible format.

        spoon-core's ChatBot accepts messages as dicts that will be converted
        to Message objects internally.

        Args:
            messages: List of messages in OpenAI format.

        Returns:
            List of message dicts compatible with spoon-core.
        """
        converted = []

        for msg in messages:
            converted_msg: dict[str, Any] = {
                "role": msg.get("role", "user"),
                "content": msg.get("content"),
            }

            # Handle tool calls in assistant messages
            if msg.get("tool_calls"):
                converted_msg["tool_calls"] = [
                    self._convert_tool_call_to_core(tc) for tc in msg["tool_calls"]
                ]

            # Handle tool results
            if msg.get("tool_call_id"):
                converted_msg["tool_call_id"] = msg["tool_call_id"]

            # Handle name field (for tool messages)
            if msg.get("name"):
                converted_msg["name"] = msg["name"]

            converted.append(converted_msg)

        return converted

    def _convert_tool_call_to_core(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Convert an OpenAI-format tool call to spoon-core format.

        Args:
            tool_call: Tool call in OpenAI format.

        Returns:
            Tool call dict compatible with spoon-core's ToolCall model.
        """
        func = tool_call.get("function", {})
        arguments = func.get("arguments", "{}")

        # Ensure arguments is a string (JSON)
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)

        return {
            "id": tool_call.get("id", ""),
            "type": tool_call.get("type", "function"),
            "function": {
                "name": func.get("name", ""),
                "arguments": arguments,
            },
        }

    def _convert_tools_to_core(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format tools to spoon-core format.

        Args:
            tools: List of tools in OpenAI format.

        Returns:
            List of tools compatible with spoon-core.
        """
        # spoon-core accepts tools in OpenAI format, so minimal conversion needed
        return tools

    def _convert_tool_response_to_spoon_bot(
        self, core_response: Any, model: str | None = None
    ) -> LLMResponse:
        """Convert spoon-core's LLMResponse to spoon-bot's LLMResponse.

        Args:
            core_response: Response from ChatBot.ask_tool().
            model: Optional model name override.

        Returns:
            spoon-bot LLMResponse.
        """
        tool_calls: list[ToolCall] = []

        # Extract tool calls from core response
        if hasattr(core_response, "tool_calls") and core_response.tool_calls:
            for tc in core_response.tool_calls:
                tool_calls.append(self._convert_core_tool_call_to_spoon_bot(tc))

        # Extract content
        content = None
        if hasattr(core_response, "content"):
            content = core_response.content

        # Extract usage if available
        usage = None
        if hasattr(core_response, "usage") and core_response.usage:
            usage = core_response.usage

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=model or self.model,
            usage=usage,
        )

    def _convert_core_tool_call_to_spoon_bot(self, tc: Any) -> ToolCall:
        """Convert a spoon-core ToolCall to spoon-bot ToolCall.

        Args:
            tc: spoon-core ToolCall object.

        Returns:
            spoon-bot ToolCall.
        """
        # Handle both object and dict formats
        if hasattr(tc, "function"):
            func = tc.function
            name = func.name if hasattr(func, "name") else func.get("name", "")
            arguments_str = (
                func.arguments if hasattr(func, "arguments") else func.get("arguments", "{}")
            )
            tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "")
        elif isinstance(tc, dict):
            func = tc.get("function", {})
            name = func.get("name", "")
            arguments_str = func.get("arguments", "{}")
            tc_id = tc.get("id", "")
        else:
            # Fallback for unexpected formats
            logger.warning(f"Unexpected tool call format: {type(tc)}")
            return ToolCall(id="", name="", arguments={})

        # Parse arguments JSON string to dict
        try:
            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str) if arguments_str else {}
            else:
                arguments = arguments_str if isinstance(arguments_str, dict) else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call arguments: {arguments_str}")
            arguments = {}

        return ToolCall(
            id=tc_id,
            name=name,
            arguments=arguments,
        )

    @property
    def chatbot(self) -> Any:
        """Access the underlying ChatBot instance.

        Returns:
            The ChatBot instance, or None if not initialized.
        """
        return self._chatbot

    @property
    def is_available(self) -> bool:
        """Check if spoon-core is available.

        Returns:
            True if spoon-core is installed and the provider is initialized.
        """
        return _SPOON_CORE_AVAILABLE and self._chatbot is not None

    async def close(self) -> None:
        """Clean up resources.

        This method is provided for compatibility but spoon-core's ChatBot
        handles cleanup internally.
        """
        # ChatBot handles cleanup via LLMManager's cleanup
        pass


def is_spoon_core_available() -> bool:
    """Check if spoon-core is available for use.

    Returns:
        True if spoon-core (spoon_ai) is installed.
    """
    return _SPOON_CORE_AVAILABLE
