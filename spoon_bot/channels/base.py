"""Base channel interface with enhanced features."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

from loguru import logger

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop
    from spoon_bot.bus.events import InboundMessage, OutboundMessage
    from spoon_bot.bus.queue import MessageBus


class ChannelMode(Enum):
    """Channel operation mode."""

    POLLING = "polling"  # Active polling for messages
    WEBHOOK = "webhook"  # Passive webhook receiving
    GATEWAY = "gateway"  # Persistent gateway connection (e.g., Discord)


class ChannelStatus(Enum):
    """Channel health status."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    RECONNECTING = "reconnecting"


class ChannelConfig:
    """Base configuration for channels."""

    def __init__(
        self,
        name: str,
        mode: ChannelMode = ChannelMode.POLLING,
        enabled: bool = True,
        retry_max_attempts: int = 3,
        retry_delay: float = 1.0,
        health_check_interval: float = 60.0,
        webhook_path: str | None = None,
        webhook_secret: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize channel configuration.

        Args:
            name: Channel identifier
            mode: Operation mode (polling/webhook/gateway)
            enabled: Whether channel is enabled
            retry_max_attempts: Max retry attempts for failed operations
            retry_delay: Delay between retries (seconds)
            health_check_interval: Health check interval (seconds)
            webhook_path: Webhook URL path (for webhook mode)
            webhook_secret: Webhook secret for verification
            **kwargs: Additional channel-specific config
        """
        self.name = name
        self.mode = mode
        self.enabled = enabled
        self.retry_max_attempts = retry_max_attempts
        self.retry_delay = retry_delay
        self.health_check_interval = health_check_interval
        self.webhook_path = webhook_path
        self.webhook_secret = webhook_secret
        self.extra = kwargs


class BaseChannel(ABC):
    """
    Enhanced abstract base class for communication channels.

    Features:
    - Webhook/Polling/Gateway modes
    - Automatic retry with exponential backoff
    - Health checking
    - Status tracking
    - Multi-account support (via account_id)
    """

    def __init__(self, config: ChannelConfig, account_id: str | None = None):
        """
        Initialize channel.

        Args:
            config: Channel configuration
            account_id: Optional account identifier for multi-account support
        """
        self.config = config
        self.name = config.name
        self.account_id = account_id or "default"
        self.full_name = f"{self.name}:{self.account_id}"

        # State
        self._bus: MessageBus | None = None
        self._status = ChannelStatus.STOPPED
        self._error: Exception | None = None
        self._last_heartbeat: datetime | None = None
        self._running = False

        # Agent reference (set by ChannelManager, used by subclasses)
        self._agent_loop: AgentLoop | None = None

        # Tasks
        self._health_check_task: asyncio.Task | None = None
        self._message_tasks: set[asyncio.Task] = set()

    def set_agent(self, agent: AgentLoop) -> None:
        """Set agent reference. Override in subclasses to use it.

        Args:
            agent: AgentLoop instance.
        """
        self._agent_loop = agent

    def attach_bus(self, bus: MessageBus) -> None:
        """
        Attach to message bus.

        Args:
            bus: MessageBus instance.
        """
        self._bus = bus
        bus.register_outbound_handler(self.full_name, self.send)
        logger.debug(f"Channel {self.full_name} attached to bus")

    @abstractmethod
    async def start(self) -> None:
        """
        Start the channel (begin listening for messages).

        Implementations should:
        1. Set self._status = ChannelStatus.STARTING
        2. Initialize connections/listeners
        3. Set self._status = ChannelStatus.RUNNING
        4. Set self._running = True
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        Stop the channel.

        Implementations should:
        1. Stop all listeners
        2. Clean up resources
        3. Set self._running = False
        4. Set self._status = ChannelStatus.STOPPED
        """
        pass

    @abstractmethod
    async def send(self, message: OutboundMessage) -> None:
        """
        Send a message through this channel.

        Args:
            message: Outbound message to send.

        Raises:
            Exception: If send fails after retries
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
            self._update_heartbeat()
        else:
            raise RuntimeError(f"Channel {self.full_name} not attached to bus")

    async def send_with_retry(
        self,
        send_func: Callable,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute send operation with retry logic.

        Args:
            send_func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from send_func

        Raises:
            Exception: If all retries fail
        """
        max_attempts = self.config.retry_max_attempts
        delay = self.config.retry_delay

        for attempt in range(1, max_attempts + 1):
            try:
                result = await send_func(*args, **kwargs)
                if attempt > 1:
                    logger.info(
                        f"[{self.full_name}] Send succeeded on attempt {attempt}"
                    )
                return result
            except Exception as e:
                if attempt == max_attempts:
                    logger.error(
                        f"[{self.full_name}] Send failed after {max_attempts} attempts: {e}"
                    )
                    raise
                else:
                    wait_time = delay * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(
                        f"[{self.full_name}] Send failed (attempt {attempt}/{max_attempts}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check.

        Returns:
            Dictionary with health status information
        """
        return {
            "channel": self.full_name,
            "status": self._status.value,
            "running": self._running,
            "last_heartbeat": (
                self._last_heartbeat.isoformat() if self._last_heartbeat else None
            ),
            "error": str(self._error) if self._error else None,
            "mode": self.config.mode.value,
            "account_id": self.account_id,
        }

    async def _start_health_check_loop(self) -> None:
        """Start periodic health check loop."""
        if self.config.health_check_interval <= 0:
            return

        while self._running:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                health = await self.health_check()
                logger.debug(f"[{self.full_name}] Health check: {health['status']}")

                # Auto-reconnect on error (can be overridden)
                if self._status == ChannelStatus.ERROR:
                    logger.warning(f"[{self.full_name}] Detected error, attempting reconnect...")
                    await self._attempt_reconnect()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.full_name}] Health check failed: {e}")

    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect the channel."""
        try:
            self._status = ChannelStatus.RECONNECTING
            logger.info(f"[{self.full_name}] Reconnecting...")
            await self.stop()
            await asyncio.sleep(2)
            await self.start()
            logger.info(f"[{self.full_name}] Reconnected successfully")
        except Exception as e:
            logger.error(f"[{self.full_name}] Reconnect failed: {e}")
            self._status = ChannelStatus.ERROR
            self._error = e

    def _update_heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self._last_heartbeat = datetime.now()

    def _set_status(self, status: ChannelStatus, error: Exception | None = None) -> None:
        """
        Update channel status.

        Args:
            status: New status
            error: Optional error if status is ERROR
        """
        self._status = status
        self._error = error
        if error:
            logger.error(f"[{self.full_name}] Status changed to {status.value}: {error}")
        else:
            logger.info(f"[{self.full_name}] Status changed to {status.value}")

    # Processing hooks (overridable by subclasses)

    async def on_processing_start(self, message: "InboundMessage") -> None:
        """Called by ChannelManager when message processing actually begins.

        Override in subclasses to show typing indicators, status updates, etc.
        This is intentionally separate from message receipt -- it fires only
        after the bus semaphore is acquired, so the agent is truly working.
        """

    async def on_processing_end(self, message: "InboundMessage") -> None:
        """Called by ChannelManager when message processing ends (success or error).

        Override in subclasses to stop typing indicators, clean up state, etc.
        Always called via ``finally``, guaranteed to run even on exceptions.
        """

    @property
    def is_attached(self) -> bool:
        """Check if channel is attached to bus."""
        return self._bus is not None

    @property
    def is_running(self) -> bool:
        """Check if channel is running."""
        return self._running and self._status == ChannelStatus.RUNNING

    @property
    def status(self) -> ChannelStatus:
        """Get current channel status."""
        return self._status

    # Webhook support (optional, for webhook-mode channels)
    async def handle_webhook(self, request: Any) -> dict[str, Any]:
        """
        Handle incoming webhook request.

        Args:
            request: HTTP request object (framework-specific)

        Returns:
            Response dictionary

        Note:
            Override this method in webhook-mode channels
        """
        raise NotImplementedError(f"{self.full_name} does not support webhook mode")

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: Request payload
            signature: Signature to verify

        Returns:
            True if signature is valid

        Note:
            Override this method to implement platform-specific signature verification
        """
        raise NotImplementedError(f"{self.full_name} does not implement signature verification")
