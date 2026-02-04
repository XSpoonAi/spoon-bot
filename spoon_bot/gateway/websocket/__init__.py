"""WebSocket module for real-time communication."""

from spoon_bot.gateway.websocket.manager import ConnectionManager
from spoon_bot.gateway.websocket.protocol import (
    WSMessage,
    WSRequest,
    WSResponse,
    WSEvent,
    WSError,
)

__all__ = [
    "ConnectionManager",
    "WSMessage",
    "WSRequest",
    "WSResponse",
    "WSEvent",
    "WSError",
]
