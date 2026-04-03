"""Channel webhook dispatch endpoints."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from spoon_bot.channels.base import ChannelMode
from spoon_bot.gateway import app as gateway_app

router = APIRouter(include_in_schema=False)


def _normalize_webhook_path(raw: str | None) -> str | None:
    """Normalize configured webhook URLs or raw paths to a comparable route path."""
    if not raw:
        return None
    value = str(raw).strip()
    if not value:
        return None
    parsed = urlparse(value)
    path = parsed.path if (parsed.scheme or parsed.netloc) else value
    if not path.startswith("/"):
        path = f"/{path}"
    path = path.rstrip("/")
    return path or "/"


def _default_webhook_path(channel: Any) -> str | None:
    """Return a fallback webhook path for channels that define an implicit route."""
    if getattr(channel, "name", "") == "feishu":
        return "/feishu/events"
    return None


def _match_webhook_channel(request_path: str) -> Any | None:
    """Return the unique channel configured for the given webhook path."""
    channel_manager = gateway_app._channel_manager
    if channel_manager is None:
        return None

    normalized_request = _normalize_webhook_path(request_path)
    if normalized_request is None:
        return None

    matches: list[Any] = []
    for name in channel_manager.channel_names:
        channel = channel_manager.get_channel(name)
        if channel is None or channel.config.mode != ChannelMode.WEBHOOK:
            continue
        configured = _normalize_webhook_path(channel.config.webhook_path)
        if configured is None:
            configured = _default_webhook_path(channel)
        if configured == normalized_request:
            matches.append(channel)

    if not matches:
        return None
    if len(matches) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Multiple webhook channels configured for {normalized_request}",
        )
    return matches[0]


@router.post("/{webhook_path:path}")
async def dispatch_channel_webhook(webhook_path: str, request: Request) -> JSONResponse:
    """Dispatch incoming channel webhooks to the matching webhook-mode channel."""
    channel = _match_webhook_channel(request.url.path)
    if channel is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook endpoint not found",
        )

    payload = await channel.handle_webhook(request)
    status_code = status.HTTP_200_OK
    if isinstance(payload, dict):
        payload = dict(payload)
        raw_status = payload.pop("status_code", None)
        if isinstance(raw_status, int):
            status_code = raw_status
        return JSONResponse(status_code=status_code, content=payload)

    return JSONResponse(status_code=status_code, content={"ok": True, "result": payload})
