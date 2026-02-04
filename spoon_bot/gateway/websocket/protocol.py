"""WebSocket message protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from enum import Enum


class MessageType(str, Enum):
    """WebSocket message types."""

    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    EVENT = "event"
    STREAM = "stream"
    PING = "ping"
    PONG = "pong"


@dataclass
class WSMessage:
    """Base WebSocket message."""

    type: MessageType
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {"type": self.type.value}
        if self.id:
            result["id"] = self.id
        return result


@dataclass
class WSRequest(WSMessage):
    """WebSocket request message (client -> server)."""

    method: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    type: MessageType = field(default=MessageType.REQUEST)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["method"] = self.method
        result["params"] = self.params
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WSRequest":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            method=data.get("method", ""),
            params=data.get("params", {}),
        )


@dataclass
class WSResponse(WSMessage):
    """WebSocket response message (server -> client)."""

    result: Any = None
    type: MessageType = field(default=MessageType.RESPONSE)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["result"] = self.result
        return result


@dataclass
class WSError(WSMessage):
    """WebSocket error message."""

    code: str = "UNKNOWN_ERROR"
    message: str = "An unknown error occurred"
    details: dict[str, Any] | None = None
    type: MessageType = field(default=MessageType.ERROR)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["error"] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            result["error"]["details"] = self.details
        return result


@dataclass
class WSEvent(WSMessage):
    """WebSocket event message (server -> client, no ID)."""

    event: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    type: MessageType = field(default=MessageType.EVENT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "event": self.event,
            "data": self.data,
        }


@dataclass
class WSStreamChunk(WSMessage):
    """WebSocket streaming chunk."""

    chunk: str = ""
    done: bool = False
    type: MessageType = field(default=MessageType.STREAM)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["chunk"] = self.chunk
        result["done"] = self.done
        return result


def parse_message(data: dict[str, Any]) -> WSMessage:
    """
    Parse a raw WebSocket message.

    Args:
        data: Raw message dictionary.

    Returns:
        Parsed WSMessage subclass.
    """
    msg_type = data.get("type", "request")

    if msg_type == "request":
        return WSRequest.from_dict(data)
    elif msg_type == "ping":
        return WSMessage(type=MessageType.PING, id=data.get("id"))
    else:
        return WSMessage(type=MessageType(msg_type), id=data.get("id"))
