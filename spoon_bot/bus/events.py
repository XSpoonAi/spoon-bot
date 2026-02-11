"""Message events for the bus system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4


@dataclass
class InboundMessage:
    """
    A message received from a channel.

    Represents user input from any channel (CLI, Telegram, Discord, etc.)
    """
    content: str
    channel: str  # e.g., "cli", "telegram", "discord"
    session_key: str = "default"
    sender_id: str | None = None
    sender_name: str | None = None
    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_media(self) -> bool:
        """Check if message has media attachments."""
        return len(self.media) > 0


@dataclass
class OutboundMessage:
    """
    A message to send to a channel.

    Represents agent response to be delivered to user.
    """
    content: str
    channel: str | None = None  # Target channel, None = reply to source
    reply_to: str | None = None  # Message ID to reply to
    message_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_media(self) -> bool:
        """Check if message has media attachments."""
        return len(self.media) > 0
