"""Base channel interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spoon_bot.bus.events import InboundMessage, OutboundMessage
    from spoon_bot.bus.queue import MessageBus


class BaseChannel(ABC):
    """
    Abstract base class for communication channels.

    Channels handle:
    - Receiving messages from external sources
    - Converting to InboundMessage
    - Sending OutboundMessage back to source
    """

    def __init__(self, name: str):
        """
        Initialize channel.

        Args:
            name: Unique channel identifier.
        """
        self.name = name
        self._bus: MessageBus | None = None

    def attach_bus(self, bus: MessageBus) -> None:
        """
        Attach to message bus.

        Args:
            bus: MessageBus instance.
        """
        self._bus = bus
        bus.register_outbound_handler(self.name, self.send)

    @abstractmethod
    async def start(self) -> None:
        """Start the channel (begin listening for messages)."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the channel."""
        pass

    @abstractmethod
    async def send(self, message: OutboundMessage) -> None:
        """
        Send a message through this channel.

        Args:
            message: Outbound message to send.
        """
        pass

    async def publish(self, message: InboundMessage) -> None:
        """
        Publish a message to the bus.

        Args:
            message: Inbound message to publish.
        """
        if self._bus:
            await self._bus.publish(message)
        else:
            raise RuntimeError(f"Channel {self.name} not attached to bus")

    @property
    def is_attached(self) -> bool:
        """Check if channel is attached to bus."""
        return self._bus is not None
