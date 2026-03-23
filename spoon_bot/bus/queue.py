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
    - **Latest-wins per session**: when a new message arrives for a session
      that already has an in-flight or queued message, the older message is
      cancelled/skipped and only the newest message is processed.
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

        # Message coalescing: when multiple messages arrive for the same
        # session before processing begins, they are merged into a single
        # message so the agent sees the full context (e.g. a follow-up
        # clarification is kept together with the original request).
        # If a task is already in-flight, it is cancelled and the new
        # (merged) message is processed instead.
        self._seq_counter: int = 0
        self._latest_seq: dict[str, int] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_tasks: dict[str, asyncio.Task] = {}
        # Per-session accumulator: messages are buffered here on publish()
        # and drained at processing time so that all pending messages for a
        # session are coalesced into one.
        self._session_buffer: dict[str, list[InboundMessage]] = {}

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

        **Message coalescing**: the message is added to a per-session
        buffer *and* enqueued.  When processing starts, all buffered
        messages for the session are merged into one so the agent sees
        the full context (follow-ups, corrections, etc.).  If a task is
        already running for this session, it is cancelled and the new
        (coalesced) message takes over.

        Args:
            message: Inbound message from a channel.

        Returns:
            True if the message was enqueued, False if the queue is full.
        """
        # Assign a sequence number for ordering
        self._seq_counter += 1
        message._bus_seq = self._seq_counter
        session_key = message.session_key or message.channel
        self._latest_seq[session_key] = self._seq_counter

        # Accumulate in per-session buffer for coalescing at processing time
        self._session_buffer.setdefault(session_key, []).append(message)

        # Cancel the currently running task for this session (if any).
        # The cancelled task will release its session lock, allowing the
        # new (coalesced) message to proceed once it is dequeued.
        existing_task = self._session_tasks.get(session_key)
        if existing_task and not existing_task.done():
            existing_task.cancel()
            logger.info(
                f"Cancelling in-flight task for session {session_key} — "
                f"newer message arrived: {message.content[:50]}..."
            )

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

        except asyncio.CancelledError:
            # Let CancelledError propagate — the caller handles it.
            raise
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

    def _get_session_lock(self, session_key: str) -> asyncio.Lock:
        """Return (or create) a per-session lock for serialised processing."""
        lock = self._session_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_key] = lock
        return lock

    @staticmethod
    def _coalesce_messages(messages: list[InboundMessage]) -> InboundMessage:
        """Merge a list of messages into one, preserving the latest metadata.

        The content of all messages is joined with newlines so the agent
        sees the full context.  Media attachments are concatenated.  All
        other fields (channel, session_key, metadata, …) are taken from
        the **last** message since it is the most recent user intent.
        """
        if len(messages) == 1:
            return messages[0]

        base = messages[-1]  # newest message is the base
        merged_content = "\n".join(m.content for m in messages)

        # Merge media from all messages (deduplicated, order preserved)
        seen: set[str] = set()
        merged_media: list[str] = []
        for m in messages:
            for path in m.media:
                if path not in seen:
                    seen.add(path)
                    merged_media.append(path)

        # Build the coalesced message from the newest, replacing content/media
        coalesced = InboundMessage(
            content=merged_content,
            channel=base.channel,
            session_key=base.session_key,
            sender_id=base.sender_id,
            sender_name=base.sender_name,
            message_id=base.message_id,
            timestamp=base.timestamp,
            media=merged_media,
            metadata=base.metadata.copy() if base.metadata else {},
        )
        coalesced._bus_seq = base._bus_seq
        return coalesced

    async def _process_with_semaphore(self, message: InboundMessage) -> None:
        """Process a single message under the concurrency semaphore.

        **Per-session serialisation**: messages belonging to the same
        ``session_key`` are processed one at a time via a per-session lock
        so that a fast second message cannot run concurrently with the
        first.

        **Message coalescing**: before starting actual work, all pending
        messages for this session are drained from ``_session_buffer``
        and merged into one.  This means follow-up messages ("also use
        TypeScript") are kept together with the original request.

        If this trigger message is not the latest for its session (i.e.
        a newer trigger was already enqueued), it yields to the newer
        trigger which will perform the coalescing instead.

        **Cancellation-safe**: if this task is cancelled (because a newer
        message triggered cancellation via ``publish()``), the session
        lock and semaphore are properly released and the queue bookkeeping
        is maintained.
        """
        session_key = message.session_key or message.channel
        session_lock = self._get_session_lock(session_key)

        try:
            # Acquire per-session lock first (no semaphore slot consumed
            # while waiting, so other sessions are not starved).
            async with session_lock:
                # Only the trigger with the highest seq should coalesce
                # and process.  Earlier triggers for the same session
                # exit here — the latest trigger will pick up all
                # buffered messages.
                msg_seq = message._bus_seq
                latest = self._latest_seq.get(session_key, 0)
                if msg_seq < latest:
                    logger.info(
                        f"Skipping earlier trigger (seq={msg_seq}, "
                        f"latest={latest}) for session {session_key}"
                    )
                    return

                # Drain the per-session buffer and coalesce
                buffered = self._session_buffer.pop(session_key, [])
                if buffered:
                    message = self._coalesce_messages(buffered)
                    if len(buffered) > 1:
                        logger.info(
                            f"Coalesced {len(buffered)} messages for "
                            f"session {session_key}"
                        )

                # Register as the active task for this session
                current_task = asyncio.current_task()
                self._session_tasks[session_key] = current_task  # type: ignore[assignment]

                try:
                    async with self._semaphore:
                        await self._process_message(message)
                except asyncio.CancelledError:
                    logger.info(
                        f"Task cancelled for session {session_key}: "
                        f"{message.content[:50]}..."
                    )
                    # Do NOT re-raise inside the session_lock context —
                    # we want to release the lock cleanly so the next
                    # message can proceed.
                    return
                finally:
                    # Only clear if we are still the registered task
                    if self._session_tasks.get(session_key) is current_task:
                        self._session_tasks.pop(session_key, None)
        except asyncio.CancelledError:
            # Cancelled while waiting for the session lock — nothing to
            # clean up, just exit silently.
            logger.debug(
                f"Task cancelled while waiting for lock, "
                f"session {session_key}: {message.content[:50]}..."
            )
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
