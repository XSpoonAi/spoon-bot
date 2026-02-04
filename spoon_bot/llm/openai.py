"""OpenAI GPT provider implementation."""

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


class OpenAIProvider(LLMProvider):
    """
    OpenAI GPT API provider.

    Uses the OpenAI Chat Completions API for chat completions with tool use.
    """

    DEFAULT_MODEL = "gpt-4o"
    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        base_url: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            model: Default model to use.
            max_tokens: Maximum tokens in response.
            base_url: Custom API base URL (for compatible APIs).
            timeout: Request timeout in seconds.

        Raises:
            APIKeyMissingError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise APIKeyMissingError(
                provider="OpenAI",
                env_var="OPENAI_API_KEY",
            )

        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.api_url = (base_url.rstrip("/") + "/chat/completions") if base_url else self.API_URL
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

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
        """Send a chat completion request to OpenAI."""
        model = model or self.model

        # Build request
        request_body: dict[str, Any] = {
            "model": model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": messages,
        }

        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        client = self._ensure_client()

        try:
            response = await client.post(
                self.api_url,
                json=request_body,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_response(data, model)

        except httpx.TimeoutException as e:
            logger.error(f"OpenAI API timeout: {e}")
            raise LLMTimeoutError(
                provider="OpenAI",
                timeout_seconds=self._timeout,
            ) from e
        except httpx.HTTPStatusError as e:
            error_text = e.response.text
            logger.error(f"OpenAI API error: {e.response.status_code} - {error_text}")
            raise LLMConnectionError(
                provider="OpenAI",
                status_code=e.response.status_code,
                response_text=error_text,
            ) from e
        except httpx.RequestError as e:
            logger.error(f"OpenAI API connection error: {e}")
            raise LLMConnectionError(
                provider="OpenAI",
                status_code=None,
                response_text=str(e),
            ) from e
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response: {e}")
            raise LLMResponseError(
                "Failed to parse API response",
                raw_response=getattr(response, "text", None),
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {e}")
            raise LLMConnectionError(
                provider="OpenAI",
                status_code=None,
                response_text=str(e),
            ) from e

    def _parse_response(
        self,
        data: dict[str, Any],
        model: str,
    ) -> LLMResponse:
        """Parse OpenAI API response."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        content = message.get("content")
        tool_calls = []

        # Parse tool calls
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                tool_calls.append(ToolCall(
                    id=tc.get("id"),
                    name=func.get("name"),
                    arguments=args,
                ))

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
