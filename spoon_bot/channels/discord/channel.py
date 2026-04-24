"""Discord channel implementation using discord.py."""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.bus.events import InboundMessage, OutboundMessage
from spoon_bot.channels.base import BaseChannel, ChannelConfig, ChannelStatus
from spoon_bot.channels.discord.constants import DEFAULT_INTENTS, SAFE_MESSAGE_LENGTH

try:
    import discord

    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    logger.warning(
        "discord.py not installed. "
        "Install with: uv sync --extra discord"
    )

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop


class DiscordChannel(BaseChannel):
    """
    Discord bot channel.

    Features:
    - Gateway (WebSocket) mode via discord.py
    - Guild channel and DM message handling
    - User / guild allowlist access control
    - Mention requirement for guild messages
    - Automatic message splitting at the 2000-char limit
    - Proxy support (via discord.Client proxy parameter)
    """

    def __init__(self, config: ChannelConfig, account_id: str | None = None):
        if not DISCORD_AVAILABLE:
            raise ImportError(
                "discord.py is required for Discord channel. "
                "Install with: uv sync --extra discord"
            )

        super().__init__(config, account_id)

        # Discord-specific config from config.extra
        self.token: str = config.extra.get("token", "")
        if not self.token:
            raise ValueError(f"[{self.full_name}] Discord bot token is required")

        self.allowed_guilds: set[int] = set(int(g) for g in config.extra.get("allowed_guilds", []))
        self.allowed_users: set[int] = set(int(u) for u in config.extra.get("allowed_users", []))
        self.proxy_url: str | None = config.extra.get("proxy_url")
        self.require_mention: bool = config.extra.get("require_mention", True)
        self.allow_dm: bool = config.extra.get("allow_dm", True)

        intent_names: list[str] = config.extra.get("intents", DEFAULT_INTENTS)
        self._intents = self._build_intents(intent_names)

        # Runtime state
        self._client: discord.Client | None = None
        self._client_task: asyncio.Task | None = None
        self._ready_event: asyncio.Event = asyncio.Event()

        # Typing indicator tasks: key -> asyncio.Task
        # Key is message_id (str) for per-message typing from processing hooks,
        # or channel_id (int) for legacy _start_typing calls.
        self._typing_tasks: dict[int | str, asyncio.Task] = {}

        # Pending reactions: message_id -> discord.Message (for removal after reply)
        self._pending_reactions: dict[str, tuple["discord.Message", float]] = {}

        # Message deduplication — LRU cache of recently processed message IDs.
        # Discord may replay on_message events during reconnection.
        self._seen_message_ids: OrderedDict[str, None] = OrderedDict()
        self._seen_max_size: int = 1000

        # Per-user rate limiting: user_id -> list of timestamps
        self._user_rate_limits: dict[int, list[float]] = {}
        self._rate_limit_max: int = 5  # max messages per window
        self._rate_limit_window: float = 60.0  # window in seconds

        # Periodic cleanup interval for stuck pending reactions (seconds)
        self._reaction_max_age: float = 600.0  # 10 minutes

    # ------------------------------------------------------------------
    # Agent lifecycle integration
    # ------------------------------------------------------------------

    def set_agent(self, agent: Any) -> None:
        """Override to register sub-agent lifecycle event listener."""
        super().set_agent(agent)
        _sm = getattr(agent, "subagent_manager", None)
        if _sm is not None:
            _sm.add_event_listener(self._on_subagent_event)
            logger.debug(f"[{self.full_name}] Registered sub-agent event listener")

    def _on_subagent_event(self, event: Any) -> None:
        """Handle sub-agent lifecycle events for Discord notifications."""
        event_type = getattr(event, "event_type", "")
        agent_id = getattr(event, "agent_id", "?")
        label = getattr(event, "label", "?")
        model = getattr(event, "model_name", None)

        if event_type == "spawning":
            model_str = f" [{model}]" if model else ""
            logger.info(
                f"[{self.full_name}] Sub-agent spawned: {label!r} ({agent_id}){model_str}"
            )
        elif event_type == "completed":
            elapsed = getattr(event, "elapsed_seconds", None)
            elapsed_str = f" in {elapsed}s" if elapsed else ""
            logger.info(
                f"[{self.full_name}] Sub-agent completed: {label!r} ({agent_id}){elapsed_str}"
            )
        elif event_type in ("failed", "cancelled"):
            error = getattr(event, "error", None)
            error_str = f": {error}" if error else ""
            logger.warning(
                f"[{self.full_name}] Sub-agent {event_type}: "
                f"{label!r} ({agent_id}){error_str}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_intents(intent_names: list[str]) -> discord.Intents:
        """Build discord.Intents from a list of intent name strings."""
        intents = discord.Intents.default()
        for name in intent_names:
            normalized = name.lower().replace("-", "_").replace(" ", "_")
            if hasattr(intents, normalized):
                setattr(intents, normalized, True)
            else:
                logger.warning(f"Unknown Discord intent: {name!r}")
        # Always enable message_content so we can read message text
        intents.message_content = True
        return intents

    @staticmethod
    def _split_message(content: str, max_length: int) -> list[str]:
        """Split a long message into chunks no larger than *max_length*.

        Improvements over the naive approach:
        - Never splits inside a Markdown fenced code block (````` ``` ...
          ``` `````).  When a code block would be torn in two, the current
          chunk is closed with a matching fence and the next chunk reopens it.
        - Preserves leading whitespace on continuation lines (no ``.lstrip()``
          that would destroy code indentation).
        """
        if len(content) <= max_length:
            return [content]

        chunks: list[str] = []
        _fence_re = re.compile(r"^(`{3,})")

        while content:
            if len(content) <= max_length:
                chunks.append(content)
                break

            # Prefer splitting at a newline, then a space, then hard-cut
            pos = content.rfind("\n", 0, max_length)
            if pos == -1:
                pos = content.rfind(" ", 0, max_length)
            if pos == -1:
                pos = max_length

            head = content[:pos]
            tail = content[pos:]
            # Strip only the split-point newline/space, NOT all whitespace
            if tail and tail[0] in ("\n", " "):
                tail = tail[1:]

            # Check if we are splitting inside an open code fence.
            # Count fences in *head*: odd count means the fence is still open.
            open_fence: str | None = None
            for line in head.split("\n"):
                m = _fence_re.match(line.strip())
                if m:
                    if open_fence is None:
                        open_fence = m.group(1)
                    else:
                        open_fence = None

            if open_fence:
                # Close the fence in this chunk and reopen in the next
                head += f"\n{open_fence}"
                tail = f"{open_fence}\n{tail}"

            chunks.append(head)
            content = tail

        return chunks

    # ------------------------------------------------------------------
    # Typing indicator
    # ------------------------------------------------------------------

    async def _typing_loop(self, channel: Any) -> None:
        """Send typing indicator every 8 seconds until cancelled.

        Args:
            channel: The discord channel object (TextChannel / DMChannel).
        """
        try:
            while True:
                try:
                    await channel.typing()
                except Exception as e:
                    logger.warning(f"[{self.full_name}] typing failed: {e}")
                await asyncio.sleep(8)  # Discord typing indicator lasts ~10 s
        except asyncio.CancelledError:
            pass

    async def _start_typing(self, channel: Any) -> None:
        """Start typing indicator entirely in background (fire-and-forget).

        Does NOT await the first ``channel.typing()`` synchronously — the
        entire sequence runs in a background task so the processing pipeline
        is not delayed by Discord API latency or rate limits.
        """
        channel_id = channel.id
        self._stop_typing(channel_id)

        async def _initial_then_loop() -> None:
            try:
                await channel.typing()
            except Exception as e:
                logger.warning(f"[{self.full_name}] initial typing failed: {e}")
            await self._typing_loop(channel)

        self._typing_tasks[channel_id] = asyncio.create_task(
            _initial_then_loop(), name=f"discord-typing-{channel_id}"
        )

    def _stop_typing(self, channel_id: int) -> None:
        """Stop the typing indicator for a channel."""
        task = self._typing_tasks.pop(channel_id, None)
        if task and not task.done():
            task.cancel()

    # ------------------------------------------------------------------
    # Reaction emoji
    # ------------------------------------------------------------------

    async def _add_reaction(self, message: "discord.Message", emoji: str) -> None:
        """Add a reaction emoji to a user message.

        Note: the caller is responsible for registering the message in
        ``_pending_reactions`` *before* calling this method so that
        ``_remove_reaction`` can always find it even if this API call
        hasn't completed yet.
        """
        try:
            await message.add_reaction(emoji)
        except Exception as e:
            logger.debug(f"[{self.full_name}] add_reaction failed: {e}")

    async def _safe_add_reaction(self, message: "discord.Message", emoji: str) -> None:
        """Fire-and-forget add_reaction with timeout to prevent hanging."""
        try:
            await asyncio.wait_for(self._add_reaction(message, emoji), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"[{self.full_name}] add_reaction timed out (5s)")
        except Exception as e:
            logger.debug(f"[{self.full_name}] safe_add_reaction failed: {e}")

    async def _remove_reaction(self, message_id: str, emoji: str) -> None:
        """Remove our reaction emoji from the original user message."""
        entry = self._pending_reactions.pop(message_id, None)
        if entry and self._client and self._client.user:
            orig_msg = entry[0]
            try:
                await asyncio.wait_for(
                    orig_msg.remove_reaction(emoji, self._client.user),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[{self.full_name}] remove_reaction timed out (5s)")
            except Exception as e:
                logger.debug(f"[{self.full_name}] remove_reaction failed: {e}")

    async def _safe_channel_send(self, channel: Any, content: str) -> None:
        """Best-effort channel.send() with timeout — never raises."""
        try:
            await asyncio.wait_for(channel.send(content), timeout=10.0)
        except Exception as e:
            logger.debug(f"[{self.full_name}] safe_channel_send failed: {e}")

    # ------------------------------------------------------------------
    # Processing hooks (called by ChannelManager)
    # ------------------------------------------------------------------

    async def on_processing_start(self, message: "InboundMessage") -> None:
        """Start typing indicator when agent actually begins processing.

        Uses ``message_id`` as key for the typing task (not ``channel_id``)
        so that two consecutive messages from the same channel don't cancel
        each other's typing indicators.
        """
        channel_id = message.metadata.get("channel_id")
        msg_id = message.message_id
        if channel_id and self._client and msg_id:
            try:
                ch = self._client.get_channel(int(channel_id))
                if ch is None:
                    ch = await self._client.fetch_channel(int(channel_id))
                self._start_typing_for_message(ch, msg_id)
            except Exception as e:
                logger.debug(f"[{self.full_name}] Failed to start typing: {e}")

    async def on_processing_end(self, message: "InboundMessage") -> None:
        """Stop typing indicator when processing ends (success or error)."""
        msg_id = message.message_id
        if msg_id:
            self._stop_typing_for_message(msg_id)

    def _start_typing_for_message(self, channel: Any, msg_id: str) -> None:
        """Start a typing indicator keyed by message_id (fire-and-forget)."""
        channel_id = channel.id

        async def _initial_then_loop() -> None:
            try:
                await channel.typing()
            except Exception as e:
                logger.warning(f"[{self.full_name}] initial typing failed: {e}")
            await self._typing_loop(channel)

        task = asyncio.create_task(
            _initial_then_loop(), name=f"discord-typing-{channel_id}-{msg_id}"
        )
        # Store by msg_id so concurrent messages don't clobber each other
        self._typing_tasks[msg_id] = task  # type: ignore[assignment]

    def _stop_typing_for_message(self, msg_id: str) -> None:
        """Stop the typing indicator for a specific message."""
        task = self._typing_tasks.pop(msg_id, None)  # type: ignore[arg-type]
        if task and not task.done():
            task.cancel()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Discord bot and connect to the gateway."""
        self._set_status(ChannelStatus.STARTING)
        self._ready_event.clear()

        try:
            if self.proxy_url:
                logger.info(f"[{self.full_name}] Using proxy: {self.proxy_url}")

            channel_ref = self  # closure reference for the inner class

            class _BotClient(discord.Client):
                async def on_ready(bot_self) -> None:
                    guild_names = [g.name for g in bot_self.guilds]
                    logger.info(
                        f"[{channel_ref.full_name}] Logged in as "
                        f"{bot_self.user} (ID: {bot_self.user.id}), "
                        f"guilds: {guild_names}"
                    )
                    channel_ref._set_status(ChannelStatus.RUNNING)
                    channel_ref._ready_event.set()

                async def on_message(bot_self, message: discord.Message) -> None:
                    try:
                        await channel_ref._handle_message(message)
                    except Exception as e:
                        logger.error(
                            f"[{channel_ref.full_name}] Unhandled error in "
                            f"_handle_message: {e}",
                            exc_info=True,
                        )

                async def on_disconnect(bot_self) -> None:
                    logger.warning(f"[{channel_ref.full_name}] Discord gateway disconnected")

            client_kwargs: dict[str, Any] = {"intents": self._intents}
            if self.proxy_url:
                client_kwargs["proxy"] = self.proxy_url

            self._client = _BotClient(**client_kwargs)

            # client.start() is a coroutine that runs until disconnected;
            # wrap it in a task so we don't block the event loop.
            self._client_task = asyncio.create_task(
                self._client.start(self.token),
                name=f"discord-{self.account_id}",
            )

            # Wait for on_ready (up to 30 s)
            try:
                await asyncio.wait_for(self._ready_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"[{self.full_name}] Discord client did not connect within 30 seconds"
                )

            # Set _running BEFORE starting the health check loop, because
            # _start_health_check_loop() uses ``while self._running:`` as its
            # main loop condition.
            self._running = True

            # Start health check (must come after _running = True)
            self._health_check_task = asyncio.create_task(
                self._start_health_check_loop()
            )

            logger.info(f"[{self.full_name}] Started in gateway mode")

        except Exception as e:
            self._set_status(ChannelStatus.ERROR, e)
            raise

    async def stop(self) -> None:
        """Disconnect and clean up the Discord bot."""
        if not self._running:
            return

        try:
            # Cancel all typing indicator tasks
            for key in list(self._typing_tasks):
                task = self._typing_tasks.pop(key, None)
                if task and not task.done():
                    task.cancel()

            # Drop any pending reaction references
            self._pending_reactions.clear()

            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except (asyncio.CancelledError, Exception):
                    pass

            if self._client:
                try:
                    await asyncio.wait_for(self._client.close(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self.full_name}] client.close() timed out after 10s, "
                        "forcing cleanup"
                    )
                except Exception as e:
                    logger.warning(f"[{self.full_name}] client.close() error: {e}")

            if self._client_task:
                self._client_task.cancel()
                try:
                    await asyncio.wait_for(self._client_task, timeout=5.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass

            self._running = False
            self._set_status(ChannelStatus.STOPPED)
            logger.info(f"[{self.full_name}] Stopped")

        except Exception as e:
            logger.error(f"[{self.full_name}] Error during stop: {e}")
            self._running = False
            self._set_status(ChannelStatus.ERROR, e)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _check_rate_limit(self, user_id: int) -> bool:
        """Return True if the user is within rate limits, False if throttled."""
        now = time.monotonic()
        timestamps = self._user_rate_limits.get(user_id, [])
        # Purge expired entries
        cutoff = now - self._rate_limit_window
        timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= self._rate_limit_max:
            self._user_rate_limits[user_id] = timestamps
            return False
        timestamps.append(now)
        self._user_rate_limits[user_id] = timestamps
        return True

    def _cleanup_stale_reactions(self) -> None:
        """Remove pending reaction entries older than ``_reaction_max_age``."""
        if not self._pending_reactions:
            return
        now = time.monotonic()
        stale_keys = [
            k for k, (msg, ts) in self._pending_reactions.items()
            if now - ts > self._reaction_max_age
        ]
        for k in stale_keys:
            self._pending_reactions.pop(k, None)
        if stale_keys:
            logger.debug(
                f"[{self.full_name}] Cleaned up {len(stale_keys)} stale pending reactions"
            )

    async def _handle_message(self, message: discord.Message) -> None:
        """Process an incoming Discord message."""
        # Ignore own messages and other bots
        if self._client and message.author == self._client.user:
            return
        if message.author.bot:
            return

        # Filter out system messages (pin notifications, joins, etc.)
        if message.type != discord.MessageType.default and message.type != discord.MessageType.reply:
            logger.debug(
                f"[{self.full_name}] Ignoring system message type={message.type} "
                f"from {message.author}"
            )
            return

        # Message deduplication — Discord may replay events during reconnection
        msg_id = str(message.id)
        if msg_id in self._seen_message_ids:
            logger.debug(f"[{self.full_name}] Duplicate message {msg_id}, skipping")
            return
        self._seen_message_ids[msg_id] = None
        if len(self._seen_message_ids) > self._seen_max_size:
            self._seen_message_ids.popitem(last=False)

        if not self._check_access(message):
            return

        # Per-user rate limiting
        if not self._check_rate_limit(message.author.id):
            logger.debug(
                f"[{self.full_name}] Rate limited user {message.author.id}"
            )
            try:
                await message.add_reaction("\u23f3")  # hourglass
            except Exception:
                pass
            return

        # Periodic cleanup of stale pending reactions
        self._cleanup_stale_reactions()

        # Strip @bot mention from content
        content = message.content
        if self._client and self._client.user:
            for mention_fmt in (
                f"<@{self._client.user.id}>",
                f"<@!{self._client.user.id}>",
            ):
                content = content.replace(mention_fmt, "")
        # Collapse multiple spaces left after mention removal (preserve newlines)
        content = re.sub(r" {2,}", " ", content).strip()

        if not content:
            logger.debug(
                f"[{self.full_name}] Empty content after mention removal, "
                f"skipping message from {message.author}"
            )
            return

        logger.info(
            f"[{self.full_name}] Processing message from {message.author} "
            f"(id={message.author.id}): {content[:80]}..."
        )

        is_dm = isinstance(message.channel, discord.DMChannel)
        session_key = f"discord_{self.account_id}_{message.channel.id}"

        inbound = InboundMessage(
            content=content,
            channel=self.full_name,
            session_key=session_key,
            sender_id=str(message.author.id),
            sender_name=str(message.author.display_name),
            message_id=str(message.id),
            metadata={
                "channel_id": str(message.channel.id),
                "guild_id": str(message.guild.id) if message.guild else None,
                "is_dm": is_dm,
                "chat_type": "dm" if is_dm else "guild",
                "think_level": "off",
                "verbose": False,
            },
        )

        # Register the message in _pending_reactions synchronously so that
        # _remove_reaction (called by send()) can always find it, even if the
        # actual Discord API call hasn't completed yet.
        self._pending_reactions[str(message.id)] = (message, time.monotonic())

        # Publish to bus FIRST — this is the critical path.  Discord REST
        # API calls (_add_reaction) can hang in poor-network environments
        # (e.g. going through a proxy), so they must not block publishing.
        logger.debug(f"[{self.full_name}] Publishing message {msg_id} to bus")
        published = await self.publish(inbound)
        logger.debug(f"[{self.full_name}] Publish result for {msg_id}: {published}")
        if not published:
            # Queue full — notify user (best-effort, don't block)
            self._pending_reactions.pop(str(message.id), None)
            asyncio.create_task(
                self._safe_channel_send(
                    message.channel,
                    "\u26a0\ufe0f Too many requests, please try again later.",
                )
            )
            return

        # Acknowledge receipt with 👀 reaction — fire-and-forget with timeout
        # so a stalled REST call never blocks the gateway event loop.
        asyncio.create_task(self._safe_add_reaction(message, "👀"))

    def _check_access(self, message: discord.Message) -> bool:
        """Return True if this message should be processed."""
        # User allowlist
        if self.allowed_users and message.author.id not in self.allowed_users:
            logger.debug(
                f"[{self.full_name}] Unauthorized user: {message.author.id} ({message.author})"
            )
            return False

        # DM handling
        if isinstance(message.channel, discord.DMChannel):
            if not self.allow_dm:
                logger.debug(f"[{self.full_name}] DMs not enabled")
                return False
            return True

        # Group DM — treat as DM for access control purposes.
        # Guard with ``isinstance(cls, type)`` so mocked environments
        # (where ``discord.GroupChannel`` is a MagicMock) don't crash.
        _group_channel_cls = getattr(discord, "GroupChannel", None)
        if isinstance(_group_channel_cls, type) and isinstance(message.channel, _group_channel_cls):
            if not self.allow_dm:
                logger.debug(f"[{self.full_name}] Group DMs not enabled (allow_dm=False)")
                return False
            return True

        # Guild message handling
        if message.guild:
            if self.allowed_guilds and message.guild.id not in self.allowed_guilds:
                logger.debug(
                    f"[{self.full_name}] Guild not in allowlist: {message.guild.id}"
                )
                return False

            if self.require_mention:
                if not self._client or not self._client.user:
                    return False
                if not self._client.user.mentioned_in(message):
                    return False

            return True

        # Unknown channel type with no guild — reject by default
        logger.debug(
            f"[{self.full_name}] Unknown channel type without guild: "
            f"{type(message.channel).__name__}"
        )
        return False

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(self, message: OutboundMessage) -> None:
        """Send a response message to Discord."""
        if not self._client:
            logger.error(f"[{self.full_name}] Client not initialized")
            return

        channel_id = message.metadata.get("channel_id")
        if not channel_id:
            logger.error(f"[{self.full_name}] No channel_id in message metadata")
            return

        # Stop typing indicator — response is ready.
        # Try both message-id key (new) and channel-id key (legacy).
        if message.reply_to:
            self._stop_typing_for_message(message.reply_to)
        self._stop_typing(int(channel_id))

        # Remove 👀 reaction from the original user message
        if message.reply_to:
            await self._remove_reaction(message.reply_to, "👀")

        try:
            ch = self._client.get_channel(int(channel_id))
            if ch is None:
                ch = await self._client.fetch_channel(int(channel_id))
        except Exception as e:
            logger.error(f"[{self.full_name}] Could not fetch channel {channel_id}: {e}")
            return

        chunks = self._split_message(message.content, SAFE_MESSAGE_LENGTH)

        for i, chunk in enumerate(chunks):
            async def _send(text: str = chunk) -> None:
                await ch.send(text)  # type: ignore[union-attr]

            await self.send_with_retry(_send)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)
