"""Anthropic Claude provider implementation with rate limiting."""

import json
import os
from typing import Any

import httpx
from loguru import logger

from spoon_bot.llm.base import LLMProvider, LLMResponse, ToolCall
from spoon_bot.exceptions import (
    APIKeyMissingError,
    LLMConnectionError,
    LLMTimeoutError,
    LLMResponseError,
)
from spoon_bot.utils.rate_limit import (
    RateLimitConfig,
    get_rate_limiter,
)


class AnthropicProvider(LLMProvider):
    """
    Anthropic Claude API provider.

    Uses the Anthropic Messages API for chat completions with tool use.
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    API_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        rate_limit_config: RateLimitConfig | None = None,
    ):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
            model: Default model to use.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            rate_limit_config: Configuration for rate limiting API calls.

        Raises:
            APIKeyMissingError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise APIKeyMissingError(
                provider="Anthropic",
                env_var="ANTHROPIC_API_KEY",
            )

        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

        # Initialize rate limiter
        self._rate_limit_config = rate_limit_config or RateLimitConfig.for_llm_api()
        self._rate_limiter = get_rate_limiter("anthropic_api", self._rate_limit_config)
        logger.debug(
            f"AnthropicProvider initialized with rate limit: "
            f"{self._rate_limit_config.requests_per_second}/s, "
            f"{self._rate_limit_config.requests_per_minute}/min"
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        """Lazily create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def get_default_model(self) -> str:
        return self.model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request to Anthropic with rate limiting."""
        # Apply rate limiting
        if self._rate_limit_config.enabled:
            wait_time = await self._rate_limiter.wait_and_acquire()
            if wait_time > 0.1:
                logger.info(f"Anthropic API rate limited, waited {wait_time:.2f}s")

        model = model or self.model

        # Convert messages to Anthropic format
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Build request
        request_body: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": anthropic_messages,
        }

        if system_prompt:
            request_body["system"] = system_prompt

        # Convert tools to Anthropic format
        if tools:
            anthropic_tools = self._convert_tools(tools)
            if anthropic_tools:
                request_body["tools"] = anthropic_tools

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }

        client = self._ensure_client()

        try:
            response = await client.post(
                self.API_URL,
                json=request_body,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data, model)

        except httpx.TimeoutException as e:
            logger.error(f"Anthropic API timeout: {e}")
            raise LLMTimeoutError(
                provider="Anthropic",
                timeout_seconds=self._timeout,
            ) from e
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            logger.error(f"Anthropic API error: {e.response.status_code} - {error_text}")
            raise LLMConnectionError(
                provider="Anthropic",
                status_code=e.response.status_code,
                response_text=error_text,
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Anthropic API connection error: {e}")
            raise LLMConnectionError(
                provider="Anthropic",
                status_code=None,
                response_text=str(e),
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Anthropic response: {e}")
            raise LLMResponseError(
                "Failed to parse API response",
                raw_response=getattr(response, "text", None),
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error calling Anthropic API: {e}")
            raise LLMConnectionError(
                provider="Anthropic",
                status_code=None,
                response_text=str(e),
            ) from e

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """
        Convert OpenAI-format messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages).
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "system":
                # Anthropic uses separate system parameter
                system_prompt = content
            elif role == "user":
                anthropic_messages.append({
                    "role": "user",
                    "content": self._convert_content(content),
                })
            elif role == "assistant":
                assistant_msg: dict[str, Any] = {"role": "assistant"}

                # Handle tool calls
                if msg.get("tool_calls"):
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})

                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}

                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": func.get("name"),
                            "input": args,
                        })

                    assistant_msg["content"] = content_blocks
                else:
                    assistant_msg["content"] = content or ""

                anthropic_messages.append(assistant_msg)
            elif role == "tool":
                # Tool result - append to previous user message or create new
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": content or "",
                }

                # Find or create user message for tool results
                if anthropic_messages and anthropic_messages[-1]["role"] == "user":
                    last_msg = anthropic_messages[-1]
                    if isinstance(last_msg["content"], list):
                        last_msg["content"].append(tool_result)
                    else:
                        last_msg["content"] = [
                            {"type": "text", "text": last_msg["content"]},
                            tool_result,
                        ]
                else:
                    anthropic_messages.append({
                        "role": "user",
                        "content": [tool_result],
                    })

        return system_prompt, anthropic_messages

    def _convert_content(self, content: Any) -> str | list[dict[str, Any]]:
        """Convert message content to Anthropic format."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Handle multimodal content
            result = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        result.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            # Parse data URL
                            parts = url.split(",", 1)
                            if len(parts) == 2:
                                mime_parts = parts[0].replace("data:", "").split(";")
                                media_type = mime_parts[0] if mime_parts else "image/jpeg"
                                result.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": parts[1],
                                    }
                                })
                else:
                    result.append({"type": "text", "text": str(item)})
            return result
        return str(content)

    def _convert_tools(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format tools to Anthropic format."""
        anthropic_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append({
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object"}),
                })

        return anthropic_tools

    def _parse_response(
        self,
        data: dict[str, Any],
        model: str,
    ) -> LLMResponse:
        """Parse Anthropic API response."""
        content_blocks = data.get("content", [])
        tool_calls = []
        text_content = []

        for block in content_blocks:
            block_type = block.get("type")

            if block_type == "text":
                text_content.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.get("id"),
                    name=block.get("name"),
                    arguments=block.get("input", {}),
                ))

        usage = data.get("usage", {})

        return LLMResponse(
            content="\n".join(text_content) if text_content else None,
            tool_calls=tool_calls,
            model=data.get("model", model),
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
            },
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
