"""Feishu/Lark channel implementation using lark-oapi."""

from __future__ import annotations

import asyncio
import json
import re
import threading
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.bus.events import InboundMessage, OutboundMessage
from spoon_bot.channels.base import BaseChannel, ChannelConfig, ChannelMode, ChannelStatus
from spoon_bot.channels.feishu.cards import build_markdown_card, should_use_card
from spoon_bot.channels.feishu.constants import (
    CHAT_TYPE_GROUP,
    CHAT_TYPE_P2P,
    EMOJI_ONIT,
    MESSAGE_DEDUP_MAX,
    MESSAGE_DEDUP_TTL,
    MSG_TYPE_AUDIO,
    MSG_TYPE_FILE,
    MSG_TYPE_IMAGE,
    MSG_TYPE_INTERACTIVE,
    MSG_TYPE_POST,
    MSG_TYPE_STICKER,
    MSG_TYPE_TEXT,
    MSG_TYPE_VIDEO,
    RENDER_MODE_AUTO,
    RENDER_MODE_CARD,
    RENDER_MODE_RAW,
    SAFE_MESSAGE_LENGTH,
    SENDER_NAME_CACHE_MAX,
    SENDER_NAME_TTL,
    SUPPORTED_MSG_TYPES,
)
from spoon_bot.channels.feishu.media import FeishuMedia

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        DeleteMessageReactionRequest,
        PatchMessageRequest,
        PatchMessageRequestBody,
        ReplyMessageRequest,
        ReplyMessageRequestBody,
    )
    from lark_oapi.api.im.v1.model import Emoji
    from lark_oapi.api.contact.v3 import GetUserRequest

    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    logger.warning(
        "lark-oapi not installed. "
        "Install with: uv sync --extra feishu"
    )

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop

# Domain map: config string -> lark SDK domain constant
_DOMAIN_MAP = {
    "feishu": "https://open.feishu.cn",
    "lark": "https://open.larksuite.com",
}


class FeishuChannel(BaseChannel):
    """
    Feishu/Lark bot channel.

    Features:
    - WebSocket long-connection mode (default) or Webhook mode
    - DM (p2p) and group chat support
    - Mention requirement for group chats
    - Automatic message splitting at 4000-char limit
    - Feishu (China) and Lark (international) domain support
    - Rich text / Markdown card rendering (P0-1)
    - Multi-type message reception: text, post, image, file, audio, video, sticker (P0-2)
    - Image/file upload and download via FeishuMedia (P0-3)
    - Emoji reactions / typing indicator (P0-4)
    - Message editing via PatchMessage (P0-5)
    - Sender name resolution with 10-min cache (P0-6)

    Known limitation:
    - The lark-oapi WebSocket client does not support HTTP proxy natively.
      Users who need a proxy for Feishu WebSocket should use Webhook mode
      or set the http_proxy / https_proxy environment variables (respected
      by the underlying websocket-client library on some versions).
    """

    def __init__(self, config: ChannelConfig, account_id: str | None = None):
        if not FEISHU_AVAILABLE:
            raise ImportError(
                "lark-oapi is required for Feishu channel. "
                "Install with: uv sync --extra feishu"
            )

        super().__init__(config, account_id)

        # Feishu-specific config from config.extra
        self.app_id: str = config.extra.get("app_id", "")
        self.app_secret: str = config.extra.get("app_secret", "")
        if not self.app_id or not self.app_secret:
            raise ValueError(
                f"[{self.full_name}] Feishu app_id and app_secret are required"
            )

        self.verification_token: str = config.extra.get("verification_token", "")
        self.encrypt_key: str = config.extra.get("encrypt_key", "")
        self.domain: str = config.extra.get("domain", "feishu")
        self.allowed_chats: set[str] = set(config.extra.get("allowed_chats", []))
        self.allowed_users: set[str] = set(config.extra.get("allowed_users", []))
        self.require_mention: bool = config.extra.get("require_mention", True)
        self.render_mode: str = config.extra.get("render_mode", RENDER_MODE_AUTO)

        # Runtime state
        self._api_client: Any | None = None  # lark.Client for sending
        self._ws_client: Any | None = None   # lark.ws.Client for receiving
        self._ws_thread: threading.Thread | None = None
        self._bot_open_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # P0-3: Media helper (initialised in start())
        self._media: FeishuMedia | None = None

        # P0-4: Typing indicator — maps message_id -> reaction_id
        # Empty string "" is used as a sentinel meaning "reaction pending"
        self._typing_reactions: dict[str, str] = {}

        # P0-6: Sender name cache — maps open_id -> (name, monotonic_ts)
        # Fix #1: Protected by _cache_lock for thread safety
        self._sender_name_cache: dict[str, tuple[str, float]] = {}
        self._cache_lock = threading.Lock()

        # Fix #6: Message deduplication — maps message_id -> monotonic_ts
        self._processed_messages: dict[str, float] = {}
        self._dedup_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_message(content: str, max_length: int) -> list[str]:
        """Split a long message into chunks no larger than max_length."""
        if not content:  # Fix #4: guard empty content
            return []
        chunks: list[str] = []
        while content:
            if len(content) <= max_length:
                chunks.append(content)
                break
            pos = content.rfind("\n", 0, max_length)
            if pos == -1:
                pos = content.rfind(" ", 0, max_length)
            if pos == -1:
                pos = max_length
            chunks.append(content[:pos])
            content = content[pos:].lstrip()
        return chunks

    @staticmethod
    def _strip_mention_tokens(text: str) -> str:
        """Remove Feishu @mention tokens (@_user_N) from message text."""
        return re.sub(r"@_user_\d+", "", text).strip()

    def _is_bot_mentioned(self, mentions: list[Any] | None) -> bool:
        """Return True if the bot is mentioned in the message mentions list."""
        if not mentions:
            return False
        if not self._bot_open_id:
            # Bot open_id not yet known — accept all to avoid dropping early messages
            return True
        for m in mentions:
            try:
                if m.id.open_id == self._bot_open_id:
                    return True
            except AttributeError:
                pass
        return False

    def _check_access(
        self,
        sender_id: str,
        chat_id: str,
        chat_type: str,
        mentions: list[Any] | None,
    ) -> bool:
        """Return True if this message should be processed."""
        if self.allowed_users and sender_id not in self.allowed_users:
            logger.debug(f"[{self.full_name}] Unauthorized user: {sender_id}")
            return False

        if self.allowed_chats and chat_id not in self.allowed_chats:
            logger.debug(f"[{self.full_name}] Chat not in allowlist: {chat_id}")
            return False

        if chat_type == CHAT_TYPE_GROUP and self.require_mention:
            if not self._is_bot_mentioned(mentions):
                logger.debug(f"[{self.full_name}] Bot not mentioned in group message")
                return False

        return True

    def _is_duplicate_message(self, message_id: str) -> bool:
        """Check if message was already processed. Records it if new.

        Fix #6: Uses a monotonic-time based TTL to prune stale entries.
        Prevents duplicate processing on WS reconnects.
        """
        now = time.monotonic()
        with self._dedup_lock:
            if message_id in self._processed_messages:
                return True
            self._processed_messages[message_id] = now
            # Prune old entries when exceeding max
            if len(self._processed_messages) > MESSAGE_DEDUP_MAX:
                cutoff = now - MESSAGE_DEDUP_TTL
                self._processed_messages = {
                    k: v for k, v in self._processed_messages.items()
                    if v > cutoff
                }
        return False

    # ------------------------------------------------------------------
    # P0-2: Multi-type message parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_post_content(content: dict, bot_open_id: str | None = None) -> str:
        """Extract plain text from a Feishu 'post' rich-text message.

        Post structure: {"<locale>": {"title": "...", "content": [[...elements...]]}}
        Element types: text, a (link), at (mention), img.

        Args:
            content: Parsed post content dict.
            bot_open_id: If provided, skip @mention elements for the bot (Fix #9).
        """
        try:
            # Fix #14: Try common locales first, then fall back to any available
            locale_content: dict = {}
            for key in ("zh_cn", "en_us"):
                if key in content and isinstance(content[key], dict):
                    locale_content = content[key]
                    break
            if not locale_content:
                for value in content.values():
                    if isinstance(value, dict) and "content" in value:
                        locale_content = value
                        break

            title: str = locale_content.get("title", "")
            blocks: list[list[dict]] = locale_content.get("content", [])

            parts: list[str] = []
            if title:
                parts.append(title)

            for block in blocks:
                line_parts: list[str] = []
                for element in block:
                    tag = element.get("tag", "")
                    if tag == "text":
                        line_parts.append(element.get("text", ""))
                    elif tag == "a":
                        text = element.get("text", "")
                        href = element.get("href", "")
                        line_parts.append(f"[{text}]({href})" if href else text)
                    elif tag == "at":
                        user_id = element.get("user_id", "")
                        # Fix #9: Skip bot mention to avoid residual @bot text
                        if bot_open_id and user_id == bot_open_id:
                            continue
                        name = element.get("user_name", user_id)
                        line_parts.append(f"@{name}")
                    # img elements are skipped (handled via _extract_post_image_keys)
                if line_parts:
                    parts.append("".join(line_parts))

            return "\n".join(parts).strip()
        except Exception as e:
            logger.debug(f"_parse_post_content error: {e}")
            return ""

    @staticmethod
    def _extract_post_image_keys(content: dict) -> list[str]:
        """Extract image_key values from img elements in a post message.

        Fix #16: Post messages can embed images that were previously lost.

        Args:
            content: Parsed post content dict.

        Returns:
            List of image_key strings found in the post.
        """
        keys: list[str] = []
        try:
            # Use same locale iteration as _parse_post_content
            locale_content: dict = {}
            for key in ("zh_cn", "en_us"):
                if key in content and isinstance(content[key], dict):
                    locale_content = content[key]
                    break
            if not locale_content:
                for value in content.values():
                    if isinstance(value, dict) and "content" in value:
                        locale_content = value
                        break

            for block in locale_content.get("content", []):
                for element in block:
                    if element.get("tag") == "img":
                        image_key = element.get("image_key", "")
                        if image_key:
                            keys.append(image_key)
        except Exception as e:
            logger.debug(f"_extract_post_image_keys error: {e}")
        return keys

    @staticmethod
    def _parse_media_keys(content: dict, msg_type: str) -> list[str]:
        """Extract media keys from message content by message type.

        Returns a list of key strings (image_key or file_key).
        """
        try:
            if msg_type == MSG_TYPE_IMAGE:
                key = content.get("image_key", "")
                return [key] if key else []
            elif msg_type in (MSG_TYPE_FILE, MSG_TYPE_AUDIO):
                key = content.get("file_key", "")
                return [key] if key else []
            elif msg_type == MSG_TYPE_VIDEO:
                keys = []
                if content.get("file_key"):
                    keys.append(content["file_key"])
                if content.get("image_key"):
                    keys.append(content["image_key"])  # thumbnail
                return keys
            elif msg_type == MSG_TYPE_STICKER:
                key = content.get("file_key", "")
                return [key] if key else []
        except Exception as e:
            logger.debug(f"_parse_media_keys error: {e}")
        return []

    # ------------------------------------------------------------------
    # P0-4: Emoji reactions
    # ------------------------------------------------------------------

    async def _add_reaction(self, message_id: str, emoji_type: str = EMOJI_ONIT) -> str | None:
        """Add an emoji reaction to a message. Returns reaction_id or None on failure."""
        if not self._api_client:
            return None
        try:
            request = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(
                        Emoji.builder().emoji_type(emoji_type).build()
                    )
                    .build()
                )
                .build()
            )
            response = await asyncio.to_thread(
                self._api_client.im.v1.message_reaction.create, request
            )
            if response.success():
                return response.data.reaction_id
            logger.debug(
                f"[{self.full_name}] Add reaction failed: code={response.code}, msg={response.msg}"
            )
        except Exception as e:
            logger.debug(f"[{self.full_name}] Add reaction error: {e}")
        return None

    async def _remove_reaction(self, message_id: str, reaction_id: str) -> None:
        """Remove an emoji reaction from a message.

        Fix #8: Retries once on failure to avoid leaked reactions.
        """
        if not self._api_client:
            return
        for attempt in range(2):
            try:
                request = (
                    DeleteMessageReactionRequest.builder()
                    .message_id(message_id)
                    .reaction_id(reaction_id)
                    .build()
                )
                await asyncio.to_thread(
                    self._api_client.im.v1.message_reaction.delete, request
                )
                return  # Success
            except Exception as e:
                if attempt == 0:
                    logger.debug(f"[{self.full_name}] Remove reaction retry after: {e}")
                    await asyncio.sleep(0.5)
                else:
                    logger.warning(
                        f"[{self.full_name}] Remove reaction failed permanently "
                        f"(msg={message_id}, reaction={reaction_id}): {e}"
                    )

    async def on_processing_start(self, message: InboundMessage) -> None:
        """Add typing-indicator reaction when agent begins processing.

        Fix #2: Pre-registers a sentinel ("") to prevent race condition with
        on_processing_end. If end fires before the reaction is created, the
        sentinel is popped and the reaction is cleaned up immediately after
        creation.
        """
        msg_id: str | None = message.metadata.get("message_id")
        if not msg_id:
            return
        # Pre-register sentinel before the await
        self._typing_reactions[msg_id] = ""
        reaction_id = await self._add_reaction(msg_id, EMOJI_ONIT)
        if reaction_id:
            if msg_id in self._typing_reactions:
                # on_processing_end hasn't run yet — store the real id
                self._typing_reactions[msg_id] = reaction_id
            else:
                # on_processing_end already popped the sentinel — clean up now
                await self._remove_reaction(msg_id, reaction_id)
        else:
            # Failed to add reaction — remove sentinel
            self._typing_reactions.pop(msg_id, None)

    async def on_processing_end(self, message: InboundMessage) -> None:
        """Remove typing-indicator reaction when agent finishes.

        Fix #2: Only removes if there's an actual reaction_id (truthy).
        An empty sentinel means the reaction is still being created and
        on_processing_start will handle cleanup.
        """
        msg_id: str | None = message.metadata.get("message_id")
        if not msg_id:
            return
        reaction_id = self._typing_reactions.pop(msg_id, None)
        if reaction_id:  # Has actual reaction_id (empty sentinel is falsy)
            await self._remove_reaction(msg_id, reaction_id)

    # ------------------------------------------------------------------
    # P0-5: Message editing
    # ------------------------------------------------------------------

    async def _edit_message(
        self, message_id: str, content: str, msg_type: str = MSG_TYPE_TEXT
    ) -> bool:
        """Edit an already-sent message (Feishu allows edits within 24 hours).

        Args:
            message_id: ID of the message to edit.
            content: New message content (plain text or card JSON string).
            msg_type: "text" or "interactive".

        Returns:
            True if edit succeeded, False otherwise.
        """
        if not self._api_client:
            return False
        try:
            if msg_type == MSG_TYPE_TEXT:
                content_str = json.dumps({"text": content})
            else:
                content_str = content  # already a JSON string for interactive

            request = (
                PatchMessageRequest.builder()
                .message_id(message_id)
                .request_body(
                    PatchMessageRequestBody.builder()
                    .content(content_str)
                    .build()
                )
                .build()
            )
            response = await asyncio.to_thread(
                self._api_client.im.v1.message.patch, request
            )
            if not response.success():
                logger.warning(
                    f"[{self.full_name}] Edit message failed: "
                    f"code={response.code}, msg={response.msg}"
                )
                return False
            return True
        except Exception as e:
            logger.debug(f"[{self.full_name}] Edit message error: {e}")
            return False

    # ------------------------------------------------------------------
    # P0-6: Sender name resolution
    # ------------------------------------------------------------------

    def _resolve_sender_name_sync(self, open_id: str) -> str:
        """Resolve a Feishu open_id to a display name (synchronous, best-effort).

        Checks a 10-minute in-memory cache before calling the contact API.
        Falls back to open_id if resolution fails.

        Fix #1: Thread-safe via _cache_lock.
        Fix #12: Enforces max cache size with LRU-style eviction.
        """
        # Thread-safe cache read
        with self._cache_lock:
            cached = self._sender_name_cache.get(open_id)
            if cached and (time.monotonic() - cached[1]) < SENDER_NAME_TTL:
                return cached[0]

        if not self._api_client:
            return open_id

        try:
            request = (
                GetUserRequest.builder()
                .user_id(open_id)
                .user_id_type("open_id")
                .build()
            )
            response = self._api_client.contact.v3.user.get(request)
            if response.success() and response.data and response.data.user:
                user = response.data.user
                name = (
                    getattr(user, "name", None)
                    or getattr(user, "en_name", None)
                    or open_id
                )
                # Thread-safe cache write with max size enforcement
                with self._cache_lock:
                    self._sender_name_cache[open_id] = (name, time.monotonic())
                    if len(self._sender_name_cache) > SENDER_NAME_CACHE_MAX:
                        oldest_key = min(
                            self._sender_name_cache,
                            key=lambda k: self._sender_name_cache[k][1],
                        )
                        del self._sender_name_cache[oldest_key]
                return name
        except Exception as e:
            logger.debug(f"[{self.full_name}] Sender name resolution failed for {open_id}: {e}")

        return open_id

    # ------------------------------------------------------------------
    # Fix #5: Bot info fetch
    # ------------------------------------------------------------------

    def _fetch_bot_open_id(self) -> str | None:
        """Fetch bot open_id via Feishu /bot/v3/info API (synchronous, best-effort).

        Uses urllib to avoid dependency on lark-oapi's raw request API
        which may vary across SDK versions.
        """
        domain_url = _DOMAIN_MAP.get(self.domain, _DOMAIN_MAP["feishu"])
        try:
            # Step 1: Get tenant access token
            token_data = json.dumps({
                "app_id": self.app_id,
                "app_secret": self.app_secret,
            }).encode()
            token_req = urllib.request.Request(
                f"{domain_url}/open-apis/auth/v3/tenant_access_token/internal",
                data=token_data,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            with urllib.request.urlopen(token_req, timeout=5) as resp:
                token_resp = json.loads(resp.read())
            tenant_token = token_resp.get("tenant_access_token", "")
            if not tenant_token:
                return None

            # Step 2: Get bot info
            info_req = urllib.request.Request(
                f"{domain_url}/open-apis/bot/v3/info",
                headers={"Authorization": f"Bearer {tenant_token}"},
            )
            with urllib.request.urlopen(info_req, timeout=5) as resp:
                info_resp = json.loads(resp.read())
            return info_resp.get("bot", {}).get("open_id")
        except Exception as e:
            logger.debug(f"[{self.full_name}] Failed to fetch bot open_id: {e}")
            return None

    # ------------------------------------------------------------------
    # Fix #3: Publish callback for error logging
    # ------------------------------------------------------------------

    @staticmethod
    def _on_publish_done(future: Any) -> None:
        """Log errors from publish() futures dispatched from WS thread."""
        try:
            future.result()
        except Exception as e:
            logger.error(f"Feishu publish failed: {e}")

    # ------------------------------------------------------------------
    # Event handler factory
    # ------------------------------------------------------------------

    def _build_event_handler(self) -> Any:
        """Build the lark EventDispatcherHandler that receives messages."""
        channel_ref = self

        def on_message_receive(data: Any) -> None:
            """Callback for im.message.receive_v1 events."""
            try:
                event = data.event
                message = event.message
                sender = event.sender

                msg_type: str = message.message_type
                message_id: str = message.message_id

                # Fix #6: Deduplicate messages (WS reconnect can replay)
                if channel_ref._is_duplicate_message(message_id):
                    logger.debug(
                        f"[{channel_ref.full_name}] Duplicate message {message_id}, skipping"
                    )
                    return

                # P0-2: Accept all supported types; skip unsupported ones
                if msg_type not in SUPPORTED_MSG_TYPES:
                    logger.debug(
                        f"[{channel_ref.full_name}] Unsupported message type: {msg_type}"
                    )
                    return

                content_json = json.loads(message.content)

                chat_id: str = message.chat_id
                chat_type: str = message.chat_type  # "p2p" or "group"
                sender_id: str = sender.sender_id.open_id
                mentions = getattr(message, "mentions", None)

                if not channel_ref._check_access(sender_id, chat_id, chat_type, mentions):
                    return

                # P0-2: Parse content by type
                if msg_type == MSG_TYPE_TEXT:
                    text = content_json.get("text", "").strip()
                    if chat_type == CHAT_TYPE_GROUP:
                        text = channel_ref._strip_mention_tokens(text)
                    media_keys: list[str] = []

                elif msg_type == MSG_TYPE_POST:
                    # Fix #9: Pass bot_open_id to skip bot @mention residue
                    text = channel_ref._parse_post_content(
                        content_json, channel_ref._bot_open_id
                    )
                    if chat_type == CHAT_TYPE_GROUP:
                        text = channel_ref._strip_mention_tokens(text)
                    # Fix #16: Extract image keys from post elements
                    media_keys = channel_ref._extract_post_image_keys(content_json)

                else:
                    # image / file / audio / video / sticker
                    text = f"[{msg_type}]"
                    media_keys = channel_ref._parse_media_keys(content_json, msg_type)

                if not text and not media_keys:
                    return

                # Fix #11: Resolve sender name directly (we're already in a thread,
                # no need to bridge through the async event loop)
                try:
                    sender_name = channel_ref._resolve_sender_name_sync(sender_id)
                except Exception:
                    sender_name = sender_id

                session_key = f"feishu_{channel_ref.account_id}_{chat_id}"

                inbound = InboundMessage(
                    content=text,
                    channel=channel_ref.full_name,
                    session_key=session_key,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    message_id=message_id,
                    media=media_keys,
                    metadata={
                        "chat_id": chat_id,
                        "chat_type": chat_type,
                        "message_id": message_id,
                        "msg_type": msg_type,
                        "think_level": "off",
                        "verbose": False,
                    },
                )

                # Fix #3: Bridge to async event loop with error callback
                loop = channel_ref._loop
                if loop and loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        channel_ref.publish(inbound), loop
                    )
                    future.add_done_callback(channel_ref._on_publish_done)
                else:
                    logger.error(f"[{channel_ref.full_name}] Event loop not available")

            except Exception as e:
                logger.error(f"[{channel_ref.full_name}] Error handling message: {e}")

        handler = (
            lark.EventDispatcherHandler.builder(
                self.encrypt_key, self.verification_token, lark.LogLevel.DEBUG
            )
            .register_p2_im_message_receive_v1(on_message_receive)
            .build()
        )
        return handler

    # ------------------------------------------------------------------
    # WebSocket thread
    # ------------------------------------------------------------------

    def _run_ws_in_thread(self) -> None:
        """Run the lark WebSocket client in its own event loop.

        lark_oapi.ws.client stores a module-level ``loop`` variable that is
        set to ``asyncio.get_event_loop()`` at import time.  When ``start()``
        is later called from a background thread, it invokes
        ``loop.run_until_complete()``, which fails because the main loop is
        already running.

        The fix: create a fresh event loop for this thread **before**
        constructing the ws.Client (so the module-level ``loop`` is patched
        to the new one), then call ``start()``.
        """
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        # Patch the module-level loop so lark's ws.Client uses this thread's loop
        import lark_oapi.ws.client as ws_module
        ws_module.loop = new_loop

        try:
            self._ws_client = lark.ws.Client(
                self.app_id,
                self.app_secret,
                event_handler=self._ws_event_handler,
                log_level=lark.LogLevel.DEBUG,
            )
            self._ws_client.start()
        except Exception as e:
            logger.error(f"[{self.full_name}] WebSocket client error: {e}")
        finally:
            new_loop.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Feishu bot (WebSocket or Webhook mode)."""
        self._set_status(ChannelStatus.STARTING)
        self._loop = asyncio.get_running_loop()

        try:
            domain_url = _DOMAIN_MAP.get(self.domain, _DOMAIN_MAP["feishu"])

            # API client for sending messages
            self._api_client = (
                lark.Client.builder()
                .app_id(self.app_id)
                .app_secret(self.app_secret)
                .domain(domain_url)
                .log_level(lark.LogLevel.INFO)
                .build()
            )

            # P0-3: Initialise media helper
            self._media = FeishuMedia(self._api_client)

            # Fix #5: Fetch bot open_id for accurate group mention filtering
            try:
                bot_id = await asyncio.to_thread(self._fetch_bot_open_id)
                if bot_id:
                    self._bot_open_id = bot_id
                    logger.info(f"[{self.full_name}] Bot open_id: {bot_id}")
                else:
                    logger.debug(
                        f"[{self.full_name}] Could not resolve bot open_id; "
                        "will accept any mention in groups until learned"
                    )
            except Exception as e:
                logger.debug(f"[{self.full_name}] Bot info fetch failed: {e}")

            event_handler = self._build_event_handler()

            if self.config.mode == ChannelMode.GATEWAY:
                # WebSocket mode: lark.ws.Client.start() is blocking and
                # uses a module-level event loop (lark_oapi.ws.client.loop).
                # We must run it in a daemon thread with its OWN event loop
                # to avoid "This event loop is already running" errors.
                self._ws_event_handler = event_handler  # store for thread
                self._ws_thread = threading.Thread(
                    target=self._run_ws_in_thread,
                    daemon=True,
                    name=f"feishu-ws-{self.account_id}",
                )
                self._ws_thread.start()
                logger.info(f"[{self.full_name}] WebSocket client started in background thread")

            else:
                # Webhook mode: store handler for external routing
                self._event_handler = event_handler
                logger.info(
                    f"[{self.full_name}] Webhook mode: handler registered, "
                    f"awaiting requests at {self.config.webhook_path or '/feishu/events'}"
                )

            self._running = True
            self._set_status(ChannelStatus.RUNNING)

            # Start health check
            self._health_check_task = asyncio.create_task(
                self._start_health_check_loop()
            )

            logger.info(
                f"[{self.full_name}] Started in "
                f"{'websocket' if self.config.mode == ChannelMode.GATEWAY else 'webhook'} mode "
                f"(domain: {self.domain}, render_mode: {self.render_mode})"
            )

        except Exception as e:
            self._set_status(ChannelStatus.ERROR, e)
            raise

    async def stop(self) -> None:
        """Stop the Feishu bot."""
        if not self._running:
            return

        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            # lark.ws.Client has no clean stop() method; the daemon thread
            # will be cleaned up automatically when the process exits.
            self._ws_client = None

            # Fix #15: Clean up pending typing reactions
            self._typing_reactions.clear()

            self._running = False
            self._set_status(ChannelStatus.STOPPED)
            logger.info(f"[{self.full_name}] Stopped")

        except Exception as e:
            logger.error(f"[{self.full_name}] Error during stop: {e}")
            self._set_status(ChannelStatus.ERROR, e)

    # ------------------------------------------------------------------
    # Fix #10: Health check with WS thread observability
    # ------------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """Perform health check with WebSocket thread status."""
        health = await super().health_check()
        if self._ws_thread is not None:
            ws_alive = self._ws_thread.is_alive()
            health["ws_thread_alive"] = ws_alive
            if not ws_alive and self._running:
                health["status"] = ChannelStatus.ERROR.value
                logger.warning(
                    f"[{self.full_name}] WebSocket thread is dead "
                    "but channel is marked running"
                )
        return health

    # ------------------------------------------------------------------
    # Sending (P0-1: card / raw render modes)
    # ------------------------------------------------------------------

    def _determine_msg_type(self, text: str) -> str:
        """Decide whether to send as plain text or interactive card.

        Returns "text" or "interactive" based on render_mode and content.

        Note (Fix #13): Interactive cards support up to ~30KB content, so the
        4000-char text message limit does not constrain card payloads. The split
        at SAFE_MESSAGE_LENGTH (3800) is conservative and safe for both modes.
        """
        if self.render_mode == RENDER_MODE_CARD:
            return MSG_TYPE_INTERACTIVE
        if self.render_mode == RENDER_MODE_RAW:
            return MSG_TYPE_TEXT
        # RENDER_MODE_AUTO: detect markdown
        return MSG_TYPE_INTERACTIVE if should_use_card(text) else MSG_TYPE_TEXT

    async def send(self, message: OutboundMessage) -> None:
        """Send a response message to Feishu."""
        if not self._api_client:
            logger.error(f"[{self.full_name}] API client not initialized")
            return

        # Fix #4: Guard empty content
        if not message.content:
            logger.debug(f"[{self.full_name}] Empty message content, skipping send")
            return

        chat_id: str | None = message.metadata.get("chat_id")
        message_id: str | None = message.metadata.get("message_id")

        if not chat_id:
            logger.error(f"[{self.full_name}] No chat_id in message metadata")
            return

        chunks = self._split_message(message.content, SAFE_MESSAGE_LENGTH)

        for i, chunk in enumerate(chunks):
            # P0-1: Determine format for this chunk
            out_msg_type = self._determine_msg_type(chunk)
            if out_msg_type == MSG_TYPE_INTERACTIVE:
                content_str = build_markdown_card(chunk)
            else:
                content_str = json.dumps({"text": chunk})

            reply_to = message_id if i == 0 else None  # Reply only for the first chunk

            # Fix #7: Capture all loop variables as default params to avoid
            # closure-over-mutable-variable issues
            async def _send(
                text: str = content_str,
                mtype: str = out_msg_type,
                reply_mid: str | None = reply_to,
                cid: str = chat_id,
            ) -> None:
                if reply_mid:
                    request = (
                        ReplyMessageRequest.builder()
                        .message_id(reply_mid)
                        .request_body(
                            ReplyMessageRequestBody.builder()
                            .content(text)
                            .msg_type(mtype)
                            .build()
                        )
                        .build()
                    )
                    response = await asyncio.to_thread(
                        self._api_client.im.v1.message.reply, request
                    )
                else:
                    request = (
                        CreateMessageRequest.builder()
                        .receive_id_type("chat_id")
                        .request_body(
                            CreateMessageRequestBody.builder()
                            .receive_id(cid)
                            .content(text)
                            .msg_type(mtype)
                            .build()
                        )
                        .build()
                    )
                    response = await asyncio.to_thread(
                        self._api_client.im.v1.message.create, request
                    )

                if not response.success():
                    logger.error(
                        f"[{self.full_name}] Send failed: "
                        f"code={response.code}, msg={response.msg}"
                    )
                    raise RuntimeError(f"Feishu API error: {response.msg}")

            await self.send_with_retry(_send)
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)
