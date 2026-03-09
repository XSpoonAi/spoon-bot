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

    def __init__(self, max_queue_size: int = 100, max_concurrency: int = 4):
        """
        Initialize message bus.

        Args:
            max_queue_size: Maximum messages in queue.
            max_concurrency: Maximum number of messages processed concurrently.
        """
        self._queue: asyncio.Queue[InboundMessage] = asyncio.Queue(maxsize=max_queue_size)
        self._handler: MessageHandler | None = None
        self._outbound_handlers: dict[str, Callable[[OutboundMessage], Awaitable[None]]] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._active_tasks: set[asyncio.Task] = set()

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

    async def publish(self, message: InboundMessage) -> bool:
        """
        Publish a message to the bus (non-blocking).

        Uses ``put_nowait`` to avoid blocking the caller's event loop
        when the queue is full.  A blocked ``put()`` on a bounded queue
        would freeze the channel's event handler (e.g. Discord gateway
        heartbeats), potentially causing a disconnect.

        Args:
            message: Inbound message from a channel.

        Returns:
            True if the message was enqueued, False if the queue is full.
        """
        try:
            self._queue.put_nowait(message)
            logger.debug(f"Published message from {message.channel}: {message.content[:50]}...")
            return True
        except asyncio.QueueFull:
            logger.warning(
                f"Message bus queue full ({self._queue.maxsize}), "
                f"dropping message from {message.channel}"
            )
            return False

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
            # Send error response to ensure channel cleanup (typing, reactions).
            # Without this, an unhandled exception leaves typing indicators
            # running indefinitely and acknowledgment reactions stuck.
            # Use a generic user-facing message to avoid leaking internal details.
            error_response = OutboundMessage(
                content="Sorry, an unexpected error occurred. Please try again.",
                channel=message.channel,
                reply_to=message.message_id,
                metadata=message.metadata.copy() if message.metadata else {},
            )
            handler = self._outbound_handlers.get(message.channel)
            if handler:
                try:
                    await handler(error_response)
                except Exception as send_err:
                    logger.error(f"Failed to send error response: {send_err}")

    async def _process_with_semaphore(self, message: InboundMessage) -> None:
        """Process a single message under the concurrency semaphore."""
        try:
            async with self._semaphore:
                await self._process_message(message)
        finally:
            self._queue.task_done()

    async def _run_loop(self) -> None:
        """Main processing loop — dispatches messages concurrently."""
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

                task = asyncio.create_task(
                    self._process_with_semaphore(message),
                    name=f"bus-msg-{message.channel}",
                )
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)

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
        """Stop the message bus and wait for in-flight messages to finish."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Wait for all active processing tasks to complete
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        logger.info("Message bus stopped")

    @property
    def is_running(self) -> bool:
        """Check if bus is running."""
        return self._running

    @property
    def max_concurrency(self) -> int:
        """Return the configured concurrency limit."""
        return self._max_concurrency

    def set_max_concurrency(self, value: int) -> None:
        """Replace the concurrency semaphore with a new limit.

        Safe to call before :meth:`start` or while running — active
        in-flight tasks will finish under the old semaphore; new tasks
        will use the replacement.
        """
        self._max_concurrency = value
        self._semaphore = asyncio.Semaphore(value)

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
