"""WebSocket message protocol.

Protocol design based on: docs/plans/2025-02-05-spoon-bot-api-design.md

Message Format:
- Client → Server (Request): {"id": "...", "method": "...", "params": {...}}
- Server → Client (Response): {"id": "...", "result": {...}, "error": null}
- Server → Client (Event): {"event": "...", "data": {...}, "timestamp": "..."}
"""

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


class ClientMethod(str, Enum):
    """Client-to-server method types."""

    # Chat methods
    CHAT_SEND = "chat.send"
    CHAT_CANCEL = "chat.cancel"

    # Confirmation methods
    CONFIRM_RESPOND = "confirm.respond"

    # Subscription methods
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"

    # Session methods
    SESSION_SWITCH = "session.switch"
    SESSION_LIST = "session.list"
    SESSION_CLEAR = "session.clear"
    SESSION_EXPORT = "session.export"
    SESSION_IMPORT = "session.import"
    SESSION_SEARCH = "session.search"

    # Agent methods
    AGENT_STATUS = "agent.status"

    # Workspace methods
    WORKSPACE_TREE = "workspace.tree"
    FS_LIST = "fs.list"
    FS_STAT = "fs.stat"
    FS_READ = "fs.read"
    FS_WRITE = "fs.write"
    FS_MKDIR = "fs.mkdir"
    FS_RENAME = "fs.rename"
    FS_REMOVE = "fs.remove"
    FS_WATCH = "fs.watch"
    FS_UNWATCH = "fs.unwatch"

    # Terminal methods
    TERM_OPEN = "term.open"
    TERM_INPUT = "term.input"
    TERM_RESIZE = "term.resize"
    TERM_CLOSE = "term.close"

    # Audio streaming methods
    AUDIO_STREAM_START = "audio.stream.start"
    AUDIO_STREAM_END = "audio.stream.end"

    # Heartbeat
    PING = "ping"


class ServerEvent(str, Enum):
    """Server-to-client event types."""

    # Agent state events
    AGENT_THINKING = "agent.thinking"
    AGENT_STEP = "agent.step"
    AGENT_STREAMING = "agent.streaming"
    AGENT_STREAM_CHUNK = "agent.stream.chunk"
    AGENT_STREAM_DONE = "agent.stream.done"
    AGENT_TOOL_CALL = "agent.tool_call"
    AGENT_TOOL_RESULT = "agent.tool_result"
    AGENT_COMPLETE = "agent.complete"
    AGENT_ERROR = "agent.error"
    AGENT_CANCELLED = "agent.cancelled"
    AGENT_IDLE = "agent.idle"

    # Confirmation events
    CONFIRM_REQUEST = "confirm.request"
    CONFIRM_TIMEOUT = "confirm.timeout"
    CONFIRM_RESPONSE = "confirm.response"

    # Resource events
    METRICS_UPDATE = "metrics.update"
    RESOURCE_TOKEN_LIMIT = "resource.token_limit"
    RESOURCE_TIME_LIMIT = "resource.time_limit"

    # Audio streaming events
    AUDIO_STREAM_STARTED = "audio.stream.started"
    AUDIO_STREAM_ERROR = "audio.stream.error"
    AUDIO_STREAM_TRANSCRIPTION = "audio.stream.transcription"

    # Connection events
    CONNECTION_ESTABLISHED = "connection.established"
    CONNECTION_READY = "connection.ready"
    CONNECTION_ERROR = "connection.error"

    # Sandbox events
    SANDBOX_STDOUT = "sandbox.stdout"
    SANDBOX_FILE_CHANGED = "sandbox.file.changed"
    TERM_CLOSED = "term.closed"


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
        raw_params = data.get("params", {})
        # Ensure params is always a dict – reject non-dict values (#14)
        if not isinstance(raw_params, dict):
            raise ValueError(
                f"'params' must be a JSON object, got {type(raw_params).__name__}"
            )
        return cls(
            id=data.get("id"),
            method=data.get("method", ""),
            params=raw_params,
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
