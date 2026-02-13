"""Enhanced channel manager for coordinating multiple channels."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.bus.events import InboundMessage, OutboundMessage
from spoon_bot.bus.queue import MessageBus
from spoon_bot.channels.base import BaseChannel, ChannelStatus
from spoon_bot.channels.config import ChannelsConfig, load_channels_config

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop


class ChannelManager:
    """
    Enhanced manager for multiple communication channels.

    Features:
    - Load channels from configuration
    - Hot reload configuration
    - Health monitoring
    - Selective start/stop
    - Status reporting
    """

    def __init__(self, config: ChannelsConfig | None = None, bus: MessageBus | None = None):
        """
        Initialize channel manager.

        Args:
            config: Channels configuration (loaded from file if None)
            bus: Message bus (creates new if None)
        """
        self._config = config
        self._bus = bus or MessageBus()
        self._channels: dict[str, BaseChannel] = {}
        self._agent: AgentLoop | None = None
        self._running = False
        self._health_check_task: asyncio.Task | None = None

    def set_agent(self, agent: AgentLoop) -> None:
        """
        Set the agent for handling messages.

        Args:
            agent: AgentLoop instance.
        """
        self._agent = agent
        self._bus.set_handler(self._handle_message)
        logger.info("Agent attached to ChannelManager")

    def add_channel(self, channel: BaseChannel) -> None:
        """
        Add a channel to the manager.

        Args:
            channel: Channel to add.
        """
        channel.attach_bus(self._bus)
        self._channels[channel.full_name] = channel
        logger.info(f"Added channel: {channel.full_name}")

    def remove_channel(self, name: str) -> bool:
        """
        Remove a channel.

        Args:
            name: Channel name to remove (full_name format: "type:account")

        Returns:
            True if channel was removed.
        """
        if name in self._channels:
            del self._channels[name]
            logger.info(f"Removed channel: {name}")
            return True
        return False

    def get_channel(self, name: str) -> BaseChannel | None:
        """
        Get a channel by name.

        Args:
            name: Channel full name

        Returns:
            Channel instance or None
        """
        return self._channels.get(name)

    async def load_from_config(self, config_path: str | Path | None = None) -> None:
        """
        Load channels from configuration file.

        Args:
            config_path: Path to config file (uses default locations if None)
        """
        self._config = load_channels_config(config_path)
        logger.info("Configuration loaded, creating channels...")

        # Telegram channels
        for config, account_id in self._config.get_telegram_configs():
            try:
                from spoon_bot.channels.telegram.channel import TelegramChannel

                channel = TelegramChannel(config, account_id)
                self.add_channel(channel)
            except ImportError as e:
                logger.warning(f"Failed to load Telegram channel: {e}")

        # Discord channels
        for config, account_id in self._config.get_discord_configs():
            try:
                from spoon_bot.channels.discord.channel import DiscordChannel

                channel = DiscordChannel(config, account_id)
                self.add_channel(channel)
            except ImportError as e:
                logger.warning(f"Failed to load Discord channel: {e}")

        # Feishu channels
        for config, account_id in self._config.get_feishu_configs():
            try:
                from spoon_bot.channels.feishu.channel import FeishuChannel

                channel = FeishuChannel(config, account_id)
                self.add_channel(channel)
            except ImportError as e:
                logger.warning(f"Failed to load Feishu channel: {e}")

        # CLI channel (if enabled)
        if self._config.is_cli_enabled():
            try:
                from spoon_bot.channels.cli_channel import CLIChannel

                cli_channel = CLIChannel()
                self.add_channel(cli_channel)
            except ImportError as e:
                logger.warning(f"Failed to load CLI channel: {e}")

        logger.info(f"Loaded {len(self._channels)} channels from configuration")

    async def reload_config(self, config_path: str | Path | None = None) -> None:
        """
        Hot reload configuration.

        Stops existing channels, reloads config, and starts new channels.

        Args:
            config_path: Path to config file
        """
        logger.info("Reloading configuration...")

        # Stop all channels
        await self.stop_all()

        # Clear channels
        self._channels.clear()

        # Reload
        await self.load_from_config(config_path)

        # Restart if was running
        if self._running:
            await self.start_all()

        logger.info("Configuration reloaded")

    async def start_all(self) -> None:
        """Start all channels and the message bus."""
        if not self._agent:
            raise RuntimeError("No agent set. Call set_agent() first.")

        if self._running:
            logger.warning("ChannelManager already running")
            return

        # Start message bus
        await self._bus.start()

        # Start all channels
        started = 0
        for channel in self._channels.values():
            if not channel.config.enabled:
                logger.debug(f"Skipping disabled channel: {channel.full_name}")
                continue

            try:
                await channel.start()
                started += 1
            except Exception as e:
                logger.error(f"Failed to start channel {channel.full_name}: {e}")

        self._running = True

        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            f"ChannelManager started: {started}/{len(self._channels)} channels running"
        )

    async def start_channels(self, channel_names: list[str]) -> None:
        """
        Start specific channels.

        Args:
            channel_names: List of channel names to start (e.g., ["telegram", "discord"])
        """
        if not self._agent:
            raise RuntimeError("No agent set. Call set_agent() first.")

        # Start message bus if not running
        if not self._bus.is_running:
            await self._bus.start()

        for name in channel_names:
            # Find channels matching this name (may be multiple accounts)
            matching = [
                ch for ch in self._channels.values() if ch.name == name or ch.full_name == name
            ]

            for channel in matching:
                if not channel.config.enabled:
                    logger.warning(f"Channel {channel.full_name} is disabled in config")
                    continue

                try:
                    await channel.start()
                    logger.info(f"Started channel: {channel.full_name}")
                except Exception as e:
                    logger.error(f"Failed to start channel {channel.full_name}: {e}")

        self._running = True

        # Start health check if not already running
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop_all(self) -> None:
        """Stop all channels and the message bus."""
        if not self._running:
            return

        # Stop health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop all channels
        for channel in self._channels.values():
            try:
                await channel.stop()
            except Exception as e:
                logger.error(f"Failed to stop channel {channel.full_name}: {e}")

        # Stop message bus
        await self._bus.stop()

        self._running = False
        logger.info("ChannelManager stopped")

    async def stop(self) -> None:
        """Stop all channels and the message bus (alias for stop_all)."""
        await self.stop_all()

    async def stop_channels(self, channel_names: list[str]) -> None:
        """
        Stop specific channels.

        Args:
            channel_names: List of channel names to stop
        """
        for name in channel_names:
            matching = [
                ch for ch in self._channels.values() if ch.name == name or ch.full_name == name
            ]

            for channel in matching:
                try:
                    await channel.stop()
                    logger.info(f"Stopped channel: {channel.full_name}")
                except Exception as e:
                    logger.error(f"Failed to stop channel {channel.full_name}: {e}")

    async def health_check_all(self) -> dict[str, Any]:
        """
        Get health status of all channels.

        Returns:
            Dictionary with health information for each channel
        """
        health = {
            "manager_running": self._running,
            "bus_running": self._bus.is_running,
            "total_channels": len(self._channels),
            "channels": {},
        }

        for name, channel in self._channels.items():
            try:
                channel_health = await channel.health_check()
                health["channels"][name] = channel_health
            except Exception as e:
                health["channels"][name] = {
                    "status": "error",
                    "error": str(e),
                }

        # Count by status
        running = sum(
            1
            for ch in health["channels"].values()
            if ch.get("status") == ChannelStatus.RUNNING.value
        )
        health["running_channels"] = running

        return health

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                health = await self.health_check_all()
                logger.debug(
                    f"Health: {health['running_channels']}/{health['total_channels']} channels running"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

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
            logger.debug(
                f"[{message.channel}] Processing message from {message.sender_id}: "
                f"{message.content[:50]}..."
            )

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
        """Get list of channel full names."""
        return list(self._channels.keys())

    @property
    def channels_by_type(self) -> dict[str, list[str]]:
        """Get channels grouped by type."""
        by_type: dict[str, list[str]] = {}
        for channel in self._channels.values():
            if channel.name not in by_type:
                by_type[channel.name] = []
            by_type[channel.name].append(channel.full_name)
        return by_type

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    @property
    def running_channels_count(self) -> int:
        """Get count of running channels."""
        return sum(1 for ch in self._channels.values() if ch.is_running)
