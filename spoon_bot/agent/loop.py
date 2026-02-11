"""Agent loop module — delegates to spoon_bot.core.

This module bridges the gateway's expected ``AgentLoop`` / ``create_agent``
imports to the canonical implementation in :mod:`spoon_bot.core`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncGenerator

from spoon_bot.core import SpoonBot, SpoonBotConfig


class AgentLoop(SpoonBot):
    """Gateway-compatible agent that adds the interface methods the WS
    handler and REST endpoints expect on top of :class:`SpoonBot`.

    Extra methods provided:
    * ``process(message)`` — non-streaming call
    * ``process_with_thinking(message)`` — non-streaming + thinking
    * ``stream(message, **kw)`` — yields ``dict`` chunks expected by the handler
    """

    # --- helpers expected by the WebSocket handler & REST endpoints ---

    async def process(self, message: str, **kwargs: Any) -> str:
        """Non-streaming agent call (used by REST non-stream & WS non-stream)."""
        return await self.chat(message, **kwargs)

    async def process_with_thinking(
        self, message: str, **kwargs: Any
    ) -> tuple[str, str | None]:
        """Non-streaming call that also returns thinking content.

        Attempts to extract thinking content from the result object's
        ``thinking_content``, ``thinking``, or ``metadata`` attributes.
        """
        if not self._initialized:
            await self.initialize()

        try:
            result = await self._agent.run(message, **kwargs)
        except Exception as exc:
            return f"Error processing request: {exc}", None

        # Extract response text
        if hasattr(result, "content"):
            response = result.content
        else:
            response = str(result)

        # Extract thinking content (try multiple attribute names)
        thinking: str | None = None
        if hasattr(result, "thinking_content") and result.thinking_content:
            thinking = result.thinking_content
        elif hasattr(result, "thinking") and result.thinking:
            thinking = result.thinking
        elif hasattr(result, "metadata") and isinstance(result.metadata, dict):
            thinking = result.metadata.get("thinking")

        return response, thinking

    async def stream(self, message: str, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming agent call that yields chunk dicts.

        The WS handler and SSE generator expect dicts with keys:
        ``type`` (content | thinking | done), ``delta``, ``metadata``.

        Extra kwargs like ``thinking`` are consumed here and not passed
        to the underlying spoon-core ChatBot (which doesn't support them).
        """
        # Strip kwargs not understood by ChatBot.astream()
        kwargs.pop("thinking", None)
        full_content = ""
        try:
            async for chunk in super().stream(message, **kwargs):
                full_content += chunk
                yield {"type": "content", "delta": chunk, "metadata": {}}
            yield {"type": "done", "delta": "", "metadata": {"content": full_content}}
        except Exception as exc:
            yield {"type": "done", "delta": "", "metadata": {"content": full_content, "error": str(exc)}}


async def create_agent(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    api_key: str | None = None,
    base_url: str | None = None,
    mcp_servers: dict[str, dict[str, Any]] | None = None,
    enable_skills: bool = True,
    skill_paths: list[str] | None = None,
    workspace: str | Path | None = None,
    **kwargs: Any,
) -> AgentLoop:
    """Create and initialise an :class:`AgentLoop`.

    Signature intentionally mirrors :func:`spoon_bot.core.create_agent`
    but returns an ``AgentLoop`` so the gateway gets the ``process()`` /
    ``stream()`` helpers it relies on.

    ``api_key`` and ``base_url`` are optional overrides.  When *None*,
    spoon-core's ``ConfigurationManager`` automatically resolves them
    from standard env vars (e.g. ``OPENROUTER_API_KEY``).
    """
    config = SpoonBotConfig(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        mcp_servers=mcp_servers or {},
        enable_skills=enable_skills,
        skill_paths=skill_paths or [],
        workspace=Path(workspace) if workspace else Path.home() / ".spoon-bot" / "workspace",
    )

    agent = AgentLoop(config)
    await agent.initialize()
    return agent


__all__ = ["AgentLoop", "create_agent"]
