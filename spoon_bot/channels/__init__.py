"""Channel system for multi-platform communication."""

from spoon_bot.channels.base import (
    BaseChannel,
    ChannelConfig,
    ChannelMode,
    ChannelStatus,
)

try:
    from spoon_bot.channels.manager import ChannelManager
except Exception:  # pragma: no cover - optional import for lightweight consumers
    ChannelManager = None  # type: ignore[assignment]

__all__ = [
    "BaseChannel",
    "ChannelConfig",
    "ChannelMode",
    "ChannelStatus",
    "ChannelManager",
]
