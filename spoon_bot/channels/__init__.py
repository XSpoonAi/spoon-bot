"""Channel system for multi-platform communication."""

from spoon_bot.channels.base import (
    BaseChannel,
    ChannelConfig,
    ChannelMode,
    ChannelStatus,
)
from spoon_bot.channels.manager import ChannelManager

__all__ = [
    "BaseChannel",
    "ChannelConfig",
    "ChannelMode",
    "ChannelStatus",
    "ChannelManager",
]
