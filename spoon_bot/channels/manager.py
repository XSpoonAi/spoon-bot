"""Channel manager for coordinating multiple channels."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.bus.queue import MessageBus
from spoon_bot.bus.events import InboundMessage, OutboundMessage
from spoon_bot.channels.base import BaseChannel

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop


class ChannelManager:
    """
    Manages multiple communication channels.

    Coordinates:
    - Channel lifecycle (start/stop)
    - Message routing via MessageBus
    - Agent integration
    """

    def __init__(self):
        """Initialize channel manager."""
        self._bus = MessageBus()
        self._channels: dict[str, BaseChannel] = {}
        self._agent: AgentLoop | None = None

    def set_agent(self, agent: AgentLoop) -> None:
        """
        Set the agent for handling messages.

        Args:
            agent: AgentLoop instance.
        """
        self._agent = agent
        self._bus.set_handler(self._handle_message)

    def add_channel(self, channel: BaseChannel) -> None:
        """
        Add a channel to the manager.

        Args:
            channel: Channel to add.
        """
        channel.attach_bus(self._bus)
        self._channels[channel.name] = channel
        logger.info(f"Added channel: {channel.name}")

    def remove_channel(self, name: str) -> bool:
        """
        Remove a channel.

        Args:
            name: Channel name to remove.

        Returns:
            True if channel was removed.
        """
        if name in self._channels:
            del self._channels[name]
            logger.info(f"Removed channel: {name}")
            return True
        return False

    async def start(self) -> None:
        """Start all channels and the message bus."""
        if not self._agent:
            raise RuntimeError("No agent set. Call set_agent() first.")

        # Start message bus
        await self._bus.start()

        # Start all channels
        for channel in self._channels.values():
            try:
                await channel.start()
            except Exception as e:
                logger.error(f"Failed to start channel {channel.name}: {e}")

        logger.info(f"ChannelManager started with {len(self._channels)} channels")

    async def stop(self) -> None:
        """Stop all channels and the message bus."""
        # Stop all channels
        for channel in self._channels.values():
            try:
                await channel.stop()
            except Exception as e:
                logger.error(f"Failed to stop channel {channel.name}: {e}")

        # Stop message bus
        await self._bus.stop()

        logger.info("ChannelManager stopped")

    async def _handle_message(self, message: InboundMessage) -> OutboundMessage | None:
        """
        Handle incoming message by routing to agent.

        Args:
            message: Inbound message to process.

        Returns:
            Agent's response as OutboundMessage.
        """
        if not self._agent:
            logger.error("No agent set")
            return None

        try:
            # Process with agent
            response_text = await self._agent.process(
                message=message.content,
                media=message.media if message.has_media else None,
            )

            # Create outbound message
            return OutboundMessage(
                content=response_text,
                channel=message.channel,
                reply_to=message.message_id,
                metadata=message.metadata.copy(),
            )

        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return OutboundMessage(
                content=f"Sorry, I encountered an error: {str(e)}",
                channel=message.channel,
                reply_to=message.message_id,
                metadata=message.metadata.copy(),
            )

    @property
    def channel_names(self) -> list[str]:
        """Get list of channel names."""
        return list(self._channels.keys())

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._bus.is_running
