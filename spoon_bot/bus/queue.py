"""Message bus for routing messages between channels and agent."""

from __future__ import annotations

import asyncio
from typing import Callable, Any, Awaitable

from loguru import logger

from spoon_bot.bus.events import InboundMessage, OutboundMessage


MessageHandler = Callable[[InboundMessage], Awaitable[OutboundMessage | None]]


class MessageBus:
    """
    Central message bus for routing messages.

    Features:
    - Async message queue
    - Handler registration
    - Channel routing
    - Error handling
    """

    def __init__(self, max_queue_size: int = 100):
        """
        Initialize message bus.

        Args:
            max_queue_size: Maximum messages in queue.
        """
        self._queue: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=max_queue_size)
        self._handler: MessageHandler | None = None
        self._outbound_handlers: dict[str, Callable[[OutboundMessage], Awaitable[None]]] = {}
        self._running = False
        self._task: asyncio.Task | None = None

    def set_handler(self, handler: MessageHandler) -> None:
        """
        Set the message handler (typically the agent).

        Args:
            handler: Async function that processes InboundMessage and returns OutboundMessage.
        """
        self._handler = handler

    def register_outbound_handler(
        self,
        channel: str,
        handler: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        """
        Register an outbound handler for a channel.

        Args:
            channel: Channel name.
            handler: Async function to send OutboundMessage to channel.
        """
        self._outbound_handlers[channel] = handler
        logger.debug(f"Registered outbound handler for channel: {channel}")

    async def publish(self, message: InboundMessage) -> None:
        """
        Publish a message to the bus.

        Args:
            message: Inbound message from a channel.
        """
        await self._queue.put(message)
        logger.debug(f"Published message from {message.channel}: {message.content[:50]}...")

    async def _process_message(self, message: InboundMessage) -> None:
        """Process a single message."""
        if not self._handler:
            logger.warning("No handler registered, dropping message")
            return

        try:
            response = await self._handler(message)

            if response:
                # Route to appropriate channel
                target_channel = response.channel or message.channel
                handler = self._outbound_handlers.get(target_channel)

                if handler:
                    await handler(response)
                else:
                    logger.warning(f"No outbound handler for channel: {target_channel}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _run_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Wait for message with timeout to allow graceful shutdown
                try:
                    message = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                await self._process_message(message)
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Bus loop error: {e}")

    async def start(self) -> None:
        """Start the message bus."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Message bus started")

    async def stop(self) -> None:
        """Stop the message bus."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Message bus stopped")

    @property
    def is_running(self) -> bool:
        """Check if bus is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
