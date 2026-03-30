"""Feishu/Lark channel implementation using lark-oapi."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import mimetypes
import re
import threading
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.bus.events import InboundMessage, OutboundMessage
from spoon_bot.channels.base import BaseChannel, ChannelConfig, ChannelMode, ChannelStatus
from spoon_bot.channels.feishu.cards import build_markdown_card, should_use_card
from spoon_bot.channels.feishu.constants import (
    CHAT_TYPE_GROUP,
    CHAT_TYPE_P2P,
    EMOJI_TYPING,
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

_TYPING_INDICATOR_MAX_AGE_MS = 2 * 60_000
_TYPING_BACKOFF_CODES = {429, 99991400, 99991403}
_TYPING_BACKOFF_SECONDS = 60.0
_TYPING_MODE_REACTION = "reaction"
_TYPING_MODE_PLACEHOLDER = "placeholder"
_TYPING_PLACEHOLDER_INTERVAL_SECONDS = 0.8
_TYPING_PLACEHOLDER_FRAMES = ("Typing.", "Typing..", "Typing...")
_DM_POLICY_OPEN = "open"
_DM_POLICY_ALLOWLIST = "allowlist"
_DM_POLICY_DISABLED = "disabled"
_DM_POLICY_PAIRING = "pairing"
_GROUP_POLICY_OPEN = "open"
_GROUP_POLICY_ALLOWLIST = "allowlist"
_GROUP_POLICY_DISABLED = "disabled"
_GROUP_SESSION_SCOPE_GROUP = "group"
_GROUP_SESSION_SCOPE_GROUP_SENDER = "group_sender"
_GROUP_SESSION_SCOPE_GROUP_TOPIC = "group_topic"
_GROUP_SESSION_SCOPE_GROUP_TOPIC_SENDER = "group_topic_sender"
_GROUP_SESSION_SCOPES = {
    _GROUP_SESSION_SCOPE_GROUP,
    _GROUP_SESSION_SCOPE_GROUP_SENDER,
    _GROUP_SESSION_SCOPE_GROUP_TOPIC,
    _GROUP_SESSION_SCOPE_GROUP_TOPIC_SENDER,
}


class _CaseInsensitiveHeaders(dict[str, str]):
    """Case-insensitive header mapping for SDK webhook validation."""

    def __init__(self, headers: Any):
        super().__init__((str(key).lower(), str(value)) for key, value in headers.items())

    def get(self, key: str, default: Any = None) -> Any:
        return super().get(str(key).lower(), default)


@dataclass(slots=True)
class _TypingPlaceholderState:
    """Runtime state for an animated typing placeholder message."""

    placeholder_message_id: str
    stop_event: asyncio.Event
    task: asyncio.Task[None] | None = None


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
        self.allow_from: set[str] = set(
            config.extra.get("allow_from", config.extra.get("allowed_users", []))
        )
        self.dm_policy: str = self._normalize_dm_policy(
            config.extra.get("dm_policy", _DM_POLICY_OPEN)
        )
        self.group_policy: str = self._normalize_group_policy(
            config.extra.get("group_policy", _GROUP_POLICY_ALLOWLIST)
        )
        self.group_allow_from: set[str] = set(
            config.extra.get("group_allow_from", config.extra.get("allowed_chats", []))
        )
        self.group_sender_allow_from: set[str] = set(
            config.extra.get("group_sender_allow_from", [])
        )
        self.group_session_scope: str = self._normalize_group_session_scope(
            config.extra.get("group_session_scope", _GROUP_SESSION_SCOPE_GROUP_SENDER)
        )
        self.require_mention: bool = config.extra.get("require_mention", True)
        self.render_mode: str = config.extra.get("render_mode", RENDER_MODE_AUTO)
        self.typing_indicator: bool = config.extra.get("typing_indicator", True)
        raw_typing_mode = str(
            config.extra.get("typing_mode", _TYPING_MODE_REACTION) or ""
        ).strip().lower()
        self.typing_mode: str = (
            raw_typing_mode
            if raw_typing_mode in {_TYPING_MODE_REACTION, _TYPING_MODE_PLACEHOLDER}
            else _TYPING_MODE_REACTION
        )
        raw_typing_emoji = str(config.extra.get("typing_emoji", EMOJI_TYPING) or "").strip()
        self.typing_emoji: str = raw_typing_emoji or EMOJI_TYPING

        # Runtime state
        self._api_client: Any | None = None  # lark.Client for sending
        self._ws_client: Any | None = None   # lark.ws.Client for receiving
        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_event_handler: Any | None = None
        self._event_handler: Any | None = None
        self._ws_stop_requested = threading.Event()
        self._bot_open_id: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # P0-3: Media helper (initialised in start())
        self._media: FeishuMedia | None = None

        # P0-4: Typing indicator — maps message_id -> reaction_id
        # Empty string "" is used as a sentinel meaning "reaction pending"
        self._typing_reactions: dict[str, str] = {}
        self._typing_placeholders: dict[str, _TypingPlaceholderState] = {}
        self._typing_backoff_until: float = 0.0

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
    def _normalize_dm_policy(value: Any) -> str:
        """Normalize DM policy, mapping unsupported pairing to allowlist."""
        policy = str(value or _DM_POLICY_OPEN).strip().lower()
        if policy == _DM_POLICY_PAIRING:
            return _DM_POLICY_ALLOWLIST
        if policy in {_DM_POLICY_OPEN, _DM_POLICY_ALLOWLIST, _DM_POLICY_DISABLED}:
            return policy
        return _DM_POLICY_OPEN

    @staticmethod
    def _normalize_group_policy(value: Any) -> str:
        """Normalize group access policy."""
        policy = str(value or _GROUP_POLICY_ALLOWLIST).strip().lower()
        if policy in {_GROUP_POLICY_OPEN, _GROUP_POLICY_ALLOWLIST, _GROUP_POLICY_DISABLED}:
            return policy
        return _GROUP_POLICY_ALLOWLIST

    @staticmethod
    def _normalize_group_session_scope(value: Any) -> str:
        """Normalize group session scoping mode."""
        scope = str(value or _GROUP_SESSION_SCOPE_GROUP_SENDER).strip().lower()
        if scope in _GROUP_SESSION_SCOPES:
            return scope
        return _GROUP_SESSION_SCOPE_GROUP_SENDER

    @staticmethod
    def _is_group_chat(chat_type: str) -> bool:
        """Return True when a message belongs to a group context."""
        return chat_type != CHAT_TYPE_P2P

    @staticmethod
    def _extract_thread_id(message: Any) -> str | None:
        """Extract a best-effort topic/thread identifier for scoped sessions."""
        for attr in ("thread_id", "root_id", "parent_id"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _build_session_key(
        self,
        *,
        chat_id: str,
        chat_type: str,
        sender_id: str,
        thread_id: str | None = None,
    ) -> str:
        """Build a session key honoring the configured group session scope."""
        if not self._is_group_chat(chat_type):
            return f"feishu_{self.account_id}_{chat_id}"

        parts = [chat_id]
        if self.group_session_scope in {
            _GROUP_SESSION_SCOPE_GROUP_TOPIC,
            _GROUP_SESSION_SCOPE_GROUP_TOPIC_SENDER,
        } and thread_id:
            parts.extend(["topic", thread_id])

        if self.group_session_scope in {
            _GROUP_SESSION_SCOPE_GROUP_SENDER,
            _GROUP_SESSION_SCOPE_GROUP_TOPIC_SENDER,
        }:
            parts.extend(["sender", sender_id])

        return f"feishu_{self.account_id}_{':'.join(parts)}"

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

    def _get_domain_url(self) -> str:
        """Resolve the configured Feishu/Lark domain to the SDK base URL."""
        return _DOMAIN_MAP.get(self.domain, _DOMAIN_MAP["feishu"])

    @staticmethod
    def _normalize_epoch_ms(value: Any) -> int | None:
        """Normalize a timestamp value to epoch milliseconds."""
        if value in (None, ""):
            return None
        try:
            ts = int(float(value))
        except (TypeError, ValueError):
            return None
        if ts < 10_000_000_000:
            ts *= 1000
        return ts

    @staticmethod
    def _extract_typing_backoff_code(value: Any) -> int | None:
        """Extract a Feishu backoff/rate-limit code from a response or exception."""
        if value is None:
            return None

        code = getattr(value, "code", None)
        if isinstance(code, int) and code in _TYPING_BACKOFF_CODES:
            return code

        response = getattr(value, "response", None)
        status = getattr(response, "status", None)
        if isinstance(status, int) and status in _TYPING_BACKOFF_CODES:
            return status

        data = getattr(response, "data", None)
        data_code = getattr(data, "code", None) if data is not None else None
        if isinstance(data_code, int) and data_code in _TYPING_BACKOFF_CODES:
            return data_code

        return None

    def _activate_typing_backoff(self, code: int) -> None:
        """Suppress typing indicators temporarily after rate-limit/quota failures."""
        self._typing_backoff_until = max(
            self._typing_backoff_until,
            time.monotonic() + _TYPING_BACKOFF_SECONDS,
        )
        logger.warning(
            f"[{self.full_name}] Typing indicator backing off after Feishu API code {code}"
        )

    def _typing_backoff_active(self) -> bool:
        """Return True when typing indicator retries are temporarily suppressed."""
        return time.monotonic() < self._typing_backoff_until

    def _is_stale_typing_message(self, message: InboundMessage) -> bool:
        """Return True when the inbound message is too old for a fresh typing indicator."""
        message_create_time_ms = self._normalize_epoch_ms(
            message.metadata.get("message_create_time")
        )
        if message_create_time_ms is None:
            return False
        return int(time.time() * 1000) - message_create_time_ms > _TYPING_INDICATOR_MAX_AGE_MS

    @staticmethod
    def _extract_response_message_id(response: Any) -> str | None:
        """Best-effort extraction of a sent message id from SDK responses."""
        data = getattr(response, "data", None)
        if data is None:
            return None
        message_id = getattr(data, "message_id", None)
        if isinstance(message_id, str) and message_id:
            return message_id
        body = getattr(data, "body", None)
        body_message_id = getattr(body, "message_id", None) if body is not None else None
        if isinstance(body_message_id, str) and body_message_id:
            return body_message_id
        return None

    def _serialize_outbound_content(self, content: str, msg_type: str) -> str:
        """Serialize text/card content into the Feishu API payload format."""
        if msg_type == MSG_TYPE_INTERACTIVE:
            return build_markdown_card(content)
        return json.dumps({"text": content})

    async def _send_api_message(
        self,
        *,
        chat_id: str | None,
        content: str,
        msg_type: str,
        reply_to: str | None = None,
    ) -> str | None:
        """Send a Feishu message and return the created message id when available."""
        if not self._api_client:
            raise RuntimeError(f"[{self.full_name}] API client not initialized")

        content_str = self._serialize_outbound_content(content, msg_type)

        async def _send() -> Any:
            if reply_to:
                request = (
                    ReplyMessageRequest.builder()
                    .message_id(reply_to)
                    .request_body(
                        ReplyMessageRequestBody.builder()
                        .content(content_str)
                        .msg_type(msg_type)
                        .build()
                    )
                    .build()
                )
                response = await asyncio.to_thread(
                    self._api_client.im.v1.message.reply, request
                )
            else:
                if not chat_id:
                    raise RuntimeError(f"[{self.full_name}] No chat_id in message metadata")
                request = (
                    CreateMessageRequest.builder()
                    .receive_id_type("chat_id")
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(chat_id)
                        .content(content_str)
                        .msg_type(msg_type)
                        .build()
                    )
                    .build()
                )
                response = await asyncio.to_thread(
                    self._api_client.im.v1.message.create, request
                )

            success = getattr(response, "success", None)
            if not callable(success) or not success():
                code = getattr(response, "code", "unknown")
                msg = getattr(response, "msg", "")
                logger.error(f"[{self.full_name}] Send failed: code={code}, msg={msg}")
                raise RuntimeError(f"Feishu API error: {msg}")
            return response

        response = await self.send_with_retry(_send)
        return self._extract_response_message_id(response)

    def _build_api_client(self) -> Any:
        """Build the SDK API client used for outbound requests."""
        return (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .domain(self._get_domain_url())
            .log_level(lark.LogLevel.INFO)
            .build()
        )

    def _build_ws_client(self) -> Any:
        """Build the SDK WebSocket client used for inbound events."""
        return lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=self._ws_event_handler,
            log_level=lark.LogLevel.DEBUG,
            domain=self._get_domain_url(),
        )

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

        if self._is_group_chat(chat_type):
            if self.group_policy == _GROUP_POLICY_DISABLED:
                logger.debug(f"[{self.full_name}] Group access disabled for chat: {chat_id}")
                return False

            if (
                self.group_policy == _GROUP_POLICY_ALLOWLIST
                and chat_id not in self.group_allow_from
            ):
                logger.debug(
                    f"[{self.full_name}] Group chat not in group_allow_from: {chat_id}"
                )
                return False

            if self.group_sender_allow_from and sender_id not in self.group_sender_allow_from:
                logger.debug(
                    f"[{self.full_name}] Group sender not in group_sender_allow_from: {sender_id}"
                )
                return False

            if self.require_mention and not self._is_bot_mentioned(mentions):
                logger.debug(f"[{self.full_name}] Bot not mentioned in group message")
                return False
            return True

        if self.dm_policy == _DM_POLICY_DISABLED:
            logger.debug(f"[{self.full_name}] DM access disabled for sender: {sender_id}")
            return False

        if self.dm_policy == _DM_POLICY_ALLOWLIST and sender_id not in self.allow_from:
            logger.debug(f"[{self.full_name}] DM sender not in allow_from: {sender_id}")
            return False

        if chat_type == CHAT_TYPE_GROUP and self.require_mention:
            if not self._is_bot_mentioned(mentions):
                logger.debug(f"[{self.full_name}] Bot not mentioned in group message")
                return False

        return True

    def _is_duplicate_message(self, message_id: str) -> bool:
        """Check if message was already processed. Records it if new.

        Uses monotonic timestamps so stale entries can expire even if the
        process clock changes, and enforces a hard upper bound on tracked ids.
        """
        now = time.monotonic()
        cutoff = now - MESSAGE_DEDUP_TTL
        with self._dedup_lock:
            existing = self._processed_messages.get(message_id)
            if existing is not None and existing > cutoff:
                return True
            self._processed_messages = {
                key: ts
                for key, ts in self._processed_messages.items()
                if ts > cutoff
            }
            self._processed_messages[message_id] = now
            overflow = len(self._processed_messages) - MESSAGE_DEDUP_MAX
            if overflow > 0:
                for stale_id in list(self._processed_messages)[:overflow]:
                    self._processed_messages.pop(stale_id, None)
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
    def _guess_image_suffix(payload: bytes) -> str:
        """Best-effort file extension detection for downloaded image payloads."""
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if payload.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if payload.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if payload.startswith(b"BM"):
            return ".bmp"
        if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
            return ".webp"
        return ".png"

    def _media_storage_root(self) -> Path:
        """Return the directory used for downloaded inbound media."""
        workspace = getattr(self._agent_loop, "workspace", None)
        if workspace:
            return Path(workspace) / ".channel_media" / "feishu" / self.account_id
        return Path(tempfile.gettempdir()) / "spoon-bot-media" / "feishu" / self.account_id

    def _path_for_agent(self, file_path: Path) -> str:
        """Return a workspace-relative path when possible, else an absolute path."""
        workspace = getattr(self._agent_loop, "workspace", None)
        if workspace:
            try:
                return file_path.relative_to(Path(workspace)).as_posix()
            except ValueError:
                pass
        return str(file_path)

    def _persist_downloaded_media(self, payload: bytes, *, stem: str, suffix: str) -> tuple[Path, str]:
        """Persist downloaded media to disk and return both file path forms."""
        root = self._media_storage_root()
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=root,
            prefix=f"{stem}-",
            suffix=suffix,
        ) as handle:
            handle.write(payload)
            file_path = Path(handle.name)
        return file_path, self._path_for_agent(file_path)

    @staticmethod
    def _attachment_ref(
        agent_path: str,
        *,
        file_name: str,
        file_type: str,
        mime_type: str | None,
        size: int,
    ) -> dict[str, Any]:
        """Build attachment metadata for AgentLoop attachment context."""
        attachment = {
            "uri": agent_path,
            "workspace_path": agent_path,
            "name": file_name,
            "file_name": file_name,
            "file_type": file_type,
            "size": size,
        }
        if mime_type:
            attachment["mime_type"] = mime_type
        return attachment

    def _materialize_image_attachment(
        self,
        image_key: str,
        *,
        stem: str,
        file_name: str | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        """Download a Feishu image and persist it for agent consumption."""
        if not self._media or not image_key:
            return None

        payload = self._media.download_image(image_key)
        suffix = Path(file_name).suffix if file_name else ""
        if not suffix:
            suffix = self._guess_image_suffix(payload)
        if file_name and not Path(file_name).suffix:
            display_name = f"{file_name}{suffix}"
        else:
            display_name = file_name or f"{stem}{suffix}"
        _, agent_path = self._persist_downloaded_media(payload, stem=stem, suffix=suffix)
        mime_type = mimetypes.guess_type(display_name)[0] or mimetypes.guess_type(f"x{suffix}")[0]

        return (
            agent_path,
            self._attachment_ref(
                agent_path,
                file_name=display_name,
                file_type=MSG_TYPE_IMAGE,
                mime_type=mime_type,
                size=len(payload),
            ),
        )

    def _materialize_file_attachment(
        self,
        *,
        message_id: str,
        file_key: str,
        resource_type: str,
        stem: str,
        file_name: str,
        file_type: str,
    ) -> dict[str, Any] | None:
        """Download a Feishu file/audio/video resource and persist it."""
        if not self._media or not file_key:
            return None

        payload = self._media.download_file(message_id, file_key, resource_type)
        suffix = Path(file_name).suffix
        if not suffix:
            suffix = {
                MSG_TYPE_AUDIO: ".opus",
                MSG_TYPE_VIDEO: ".mp4",
                MSG_TYPE_STICKER: ".webp",
            }.get(file_type, ".bin")
            file_name = f"{file_name}{suffix}"
        _, agent_path = self._persist_downloaded_media(payload, stem=stem, suffix=suffix)
        mime_type = mimetypes.guess_type(file_name)[0]

        return self._attachment_ref(
            agent_path,
            file_name=file_name,
            file_type=file_type,
            mime_type=mime_type,
            size=len(payload),
        )

    def _materialize_inbound_media(
        self,
        *,
        message_id: str,
        content: dict[str, Any],
        msg_type: str,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Download inbound Feishu media into workspace-backed files."""
        if not self._media:
            return [], []

        media_paths: list[str] = []
        attachments: list[dict[str, Any]] = []

        try:
            if msg_type == MSG_TYPE_POST:
                for index, image_key in enumerate(self._extract_post_image_keys(content), start=1):
                    item = self._materialize_image_attachment(
                        image_key,
                        stem=f"{message_id}-post-image-{index}",
                    )
                    if item is None:
                        continue
                    media_path, attachment = item
                    media_paths.append(media_path)
                    attachments.append(attachment)

            elif msg_type == MSG_TYPE_IMAGE:
                item = self._materialize_image_attachment(
                    content.get("image_key", ""),
                    stem=f"{message_id}-image",
                )
                if item is not None:
                    media_path, attachment = item
                    media_paths.append(media_path)
                    attachments.append(attachment)

            elif msg_type in (MSG_TYPE_FILE, MSG_TYPE_AUDIO, MSG_TYPE_STICKER):
                attachment = self._materialize_file_attachment(
                    message_id=message_id,
                    file_key=content.get("file_key", ""),
                    resource_type="audio" if msg_type == MSG_TYPE_AUDIO else "file",
                    stem=f"{message_id}-{msg_type}",
                    file_name=content.get("file_name") or f"{msg_type}-{message_id}",
                    file_type=msg_type,
                )
                if attachment is not None:
                    attachments.append(attachment)

            elif msg_type == MSG_TYPE_VIDEO:
                video_attachment = self._materialize_file_attachment(
                    message_id=message_id,
                    file_key=content.get("file_key", ""),
                    resource_type="video",
                    stem=f"{message_id}-video",
                    file_name=content.get("file_name") or f"video-{message_id}",
                    file_type=MSG_TYPE_VIDEO,
                )
                if video_attachment is not None:
                    attachments.append(video_attachment)

                thumbnail = self._materialize_image_attachment(
                    content.get("image_key", ""),
                    stem=f"{message_id}-video-thumbnail",
                    file_name=f"video-thumbnail-{message_id}.png",
                )
                if thumbnail is not None:
                    media_path, attachment = thumbnail
                    media_paths.append(media_path)
                    attachments.append(attachment)

        except Exception as e:
            logger.warning(f"[{self.full_name}] Failed to materialize {msg_type} media: {e}")

        return media_paths, attachments

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

    async def _add_reaction(self, message_id: str, emoji_type: str = EMOJI_TYPING) -> str | None:
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
            backoff_code = self._extract_typing_backoff_code(response)
            if backoff_code is not None:
                self._activate_typing_backoff(backoff_code)
            logger.debug(
                f"[{self.full_name}] Add reaction failed: code={response.code}, msg={response.msg}"
            )
        except Exception as e:
            backoff_code = self._extract_typing_backoff_code(e)
            if backoff_code is not None:
                self._activate_typing_backoff(backoff_code)
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
                response = await asyncio.to_thread(
                    self._api_client.im.v1.message_reaction.delete, request
                )
                success = getattr(response, "success", None)
                if not callable(success) or success():
                    return  # Success
                backoff_code = self._extract_typing_backoff_code(response)
                if backoff_code is not None:
                    self._activate_typing_backoff(backoff_code)
                    return
                logger.debug(
                    f"[{self.full_name}] Remove reaction failed: "
                    f"code={getattr(response, 'code', 'unknown')}, "
                    f"msg={getattr(response, 'msg', '')}"
                )
                return
            except Exception as e:
                backoff_code = self._extract_typing_backoff_code(e)
                if backoff_code is not None:
                    self._activate_typing_backoff(backoff_code)
                    return
                if attempt == 0:
                    logger.debug(f"[{self.full_name}] Remove reaction retry after: {e}")
                    await asyncio.sleep(0.5)
                else:
                    logger.warning(
                        f"[{self.full_name}] Remove reaction failed permanently "
                        f"(msg={message_id}, reaction={reaction_id}): {e}"
                    )

    async def _typing_placeholder_loop(
        self, source_message_id: str, state: _TypingPlaceholderState
    ) -> None:
        """Animate a placeholder message as `Typing.` -> `Typing...` while work runs."""
        frame_index = 1
        try:
            while not state.stop_event.is_set():
                await asyncio.sleep(_TYPING_PLACEHOLDER_INTERVAL_SECONDS)
                if state.stop_event.is_set():
                    break
                frame = _TYPING_PLACEHOLDER_FRAMES[frame_index % len(_TYPING_PLACEHOLDER_FRAMES)]
                updated = await self._edit_message(
                    state.placeholder_message_id,
                    frame,
                    MSG_TYPE_INTERACTIVE,
                )
                if not updated:
                    logger.debug(
                        f"[{self.full_name}] Stopping typing placeholder updates for "
                        f"{source_message_id}: edit failed"
                    )
                    break
                frame_index += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(
                f"[{self.full_name}] Typing placeholder loop failed for "
                f"{source_message_id}: {e}"
            )

    async def _start_typing_placeholder(
        self, message: InboundMessage, msg_id: str
    ) -> None:
        """Create and animate a placeholder reply that simulates typing."""
        if msg_id in self._typing_placeholders:
            return

        reply_to = message.metadata.get("message_id")
        chat_id = message.metadata.get("chat_id")
        if not reply_to and not chat_id:
            return

        try:
            placeholder_message_id = await self._send_api_message(
                chat_id=chat_id,
                content=_TYPING_PLACEHOLDER_FRAMES[0],
                msg_type=MSG_TYPE_INTERACTIVE,
                reply_to=reply_to,
            )
        except Exception as e:
            logger.debug(
                f"[{self.full_name}] Failed to create typing placeholder for {msg_id}: {e}"
            )
            return

        if not placeholder_message_id:
            return

        state = _TypingPlaceholderState(
            placeholder_message_id=placeholder_message_id,
            stop_event=asyncio.Event(),
        )
        state.task = asyncio.create_task(
            self._typing_placeholder_loop(msg_id, state),
            name=f"feishu-typing-placeholder-{msg_id}",
        )
        self._typing_placeholders[msg_id] = state

    async def _stop_typing_placeholder(self, msg_id: str) -> None:
        """Stop the placeholder animation but keep the message for final patching."""
        state = self._typing_placeholders.get(msg_id)
        if not state:
            return
        state.stop_event.set()
        task = state.task
        if task and not task.done():
            try:
                await task
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug(
                    f"[{self.full_name}] Typing placeholder cleanup failed for "
                    f"{msg_id}: {e}"
                )

    async def on_processing_start(self, message: InboundMessage) -> None:
        """Show the configured Feishu typing indicator when agent work begins."""
        if not self.typing_indicator:
            return
        msg_id: str | None = message.metadata.get("message_id")
        if not msg_id:
            return
        if self._is_stale_typing_message(message):
            return
        if self.typing_mode == _TYPING_MODE_PLACEHOLDER:
            await self._start_typing_placeholder(message, msg_id)
            return
        if msg_id in self._typing_reactions:
            return
        if self._typing_backoff_active():
            return
        self._typing_reactions[msg_id] = ""
        reaction_id = await self._add_reaction(msg_id, self.typing_emoji)
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
        """Stop the configured Feishu typing indicator when agent work ends."""
        msg_id: str | None = message.metadata.get("message_id")
        if not msg_id:
            return
        if self.typing_mode == _TYPING_MODE_PLACEHOLDER:
            await self._stop_typing_placeholder(msg_id)
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
            content_str = self._serialize_outbound_content(content, msg_type)
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
        domain_url = self._get_domain_url()
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

        def on_p2p_chat_entered(data: Any) -> None:
            """No-op callback for bot p2p chat access events."""
            event = getattr(data, "event", None)
            chat_id = getattr(event, "chat_id", None)
            logger.debug(
                f"[{channel_ref.full_name}] Ignoring bot p2p chat entered event"
                f" chat_id={chat_id}"
            )

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
                thread_id = channel_ref._extract_thread_id(message)

                if not channel_ref._check_access(sender_id, chat_id, chat_type, mentions):
                    return

                # P0-2: Parse content by type
                if msg_type == MSG_TYPE_TEXT:
                    text = content_json.get("text", "").strip()
                    if channel_ref._is_group_chat(chat_type):
                        text = channel_ref._strip_mention_tokens(text)
                    media_paths: list[str] = []
                    attachments: list[dict[str, Any]] = []

                elif msg_type == MSG_TYPE_POST:
                    # Fix #9: Pass bot_open_id to skip bot @mention residue
                    text = channel_ref._parse_post_content(
                        content_json, channel_ref._bot_open_id
                    )
                    if channel_ref._is_group_chat(chat_type):
                        text = channel_ref._strip_mention_tokens(text)
                    media_paths, attachments = channel_ref._materialize_inbound_media(
                        message_id=message_id,
                        content=content_json,
                        msg_type=msg_type,
                    )

                else:
                    # image / file / audio / video / sticker
                    text = f"[{msg_type}]"
                    media_paths, attachments = channel_ref._materialize_inbound_media(
                        message_id=message_id,
                        content=content_json,
                        msg_type=msg_type,
                    )

                if not text and not media_paths and not attachments:
                    return

                # Fix #11: Resolve sender name directly (we're already in a thread,
                # no need to bridge through the async event loop)
                try:
                    sender_name = channel_ref._resolve_sender_name_sync(sender_id)
                except Exception:
                    sender_name = sender_id

                session_key = channel_ref._build_session_key(
                    chat_id=chat_id,
                    chat_type=chat_type,
                    sender_id=sender_id,
                    thread_id=thread_id,
                )
                message_create_time = (
                    getattr(message, "create_time", None)
                    or getattr(message, "message_create_time", None)
                )

                inbound = InboundMessage(
                    content=text,
                    channel=channel_ref.full_name,
                    session_key=session_key,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    message_id=message_id,
                    media=media_paths,
                    metadata={
                        "chat_id": chat_id,
                        "chat_type": chat_type,
                        "is_dm": chat_type == CHAT_TYPE_P2P,
                        "thread_id": thread_id,
                        "message_id": message_id,
                        "message_create_time": message_create_time,
                        "msg_type": msg_type,
                        "session_scope": (
                            "dm"
                            if chat_type == CHAT_TYPE_P2P
                            else channel_ref.group_session_scope
                        ),
                        "think_level": "off",
                        "verbose": False,
                        "attachments": attachments,
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
            .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(
                on_p2p_chat_entered
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
        self._ws_loop = new_loop
        asyncio.set_event_loop(new_loop)

        # Patch the module-level loop so lark's ws.Client uses this thread's loop
        import lark_oapi.ws.client as ws_module
        ws_module.loop = new_loop

        try:
            self._ws_client = self._build_ws_client()
            self._ws_client.start()
        except RuntimeError as e:
            if self._ws_stop_requested.is_set():
                logger.debug(f"[{self.full_name}] WebSocket loop stopped: {e}")
            else:
                logger.error(f"[{self.full_name}] WebSocket client error: {e}")
        except Exception as e:
            if self._ws_stop_requested.is_set():
                logger.debug(f"[{self.full_name}] WebSocket shutdown complete")
            else:
                logger.error(f"[{self.full_name}] WebSocket client error: {e}")
        finally:
            pending = [task for task in asyncio.all_tasks(new_loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                try:
                    new_loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                except Exception:
                    pass
            self._ws_client = None
            self._ws_loop = None
            asyncio.set_event_loop(None)
            new_loop.close()

    async def _shutdown_ws_runtime(self) -> None:
        """Disconnect the SDK client and wait for the worker thread to exit."""
        ws_client = self._ws_client
        ws_loop = self._ws_loop
        ws_thread = self._ws_thread

        if ws_client is not None:
            setattr(ws_client, "_auto_reconnect", False)

        if ws_loop is not None and ws_loop.is_running():
            if ws_client is not None:
                disconnect = getattr(ws_client, "_disconnect", None)
                if disconnect is not None:
                    future = asyncio.run_coroutine_threadsafe(disconnect(), ws_loop)
                    try:
                        await asyncio.wait_for(asyncio.wrap_future(future), timeout=5)
                    except asyncio.TimeoutError:
                        logger.warning(f"[{self.full_name}] Timed out while disconnecting WebSocket client")
                    except Exception as e:
                        logger.warning(f"[{self.full_name}] WebSocket disconnect error: {e}")
            ws_loop.call_soon_threadsafe(ws_loop.stop)

        if ws_thread and ws_thread.is_alive():
            await asyncio.to_thread(ws_thread.join, 5)
            if ws_thread.is_alive():
                logger.warning(f"[{self.full_name}] WebSocket thread did not exit cleanly")

    async def handle_webhook(self, request: Any) -> dict[str, Any]:
        """Handle incoming Feishu webhook requests via the SDK event dispatcher."""
        if self.config.mode != ChannelMode.WEBHOOK:
            raise RuntimeError(f"{self.full_name} is not configured for webhook mode")
        if self._event_handler is None:
            raise RuntimeError(f"{self.full_name} webhook handler is not initialized")

        body = await request.body()
        path = getattr(getattr(request, "url", None), "path", None)
        raw_request = SimpleNamespace(
            uri=path or self.config.webhook_path or "/feishu/events",
            headers=_CaseInsensitiveHeaders(request.headers),
            body=body,
        )
        response = await asyncio.to_thread(self._event_handler.do, raw_request)
        status_code = getattr(response, "status_code", 200)
        content = getattr(response, "content", b"") or b""
        text = content.decode("utf-8") if isinstance(content, bytes) else str(content)

        if text:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"ok": status_code < 400, "text": text}
        else:
            payload = {"ok": status_code < 400}

        if isinstance(payload, dict) and status_code >= 400:
            payload.setdefault("ok", False)
            payload.setdefault("error", text or "Feishu webhook request failed")
            payload.setdefault("status_code", status_code)

        return payload

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the Feishu bot (WebSocket or Webhook mode)."""
        self._set_status(ChannelStatus.STARTING)
        self._loop = asyncio.get_running_loop()
        self._ws_stop_requested.clear()

        try:
            # API client for sending messages
            self._api_client = self._build_api_client()

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
            self._ws_stop_requested.set()
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            await self._shutdown_ws_runtime()
            self._ws_client = None
            self._ws_thread = None
            self._ws_loop = None
            self._ws_event_handler = None
            self._event_handler = None

            placeholder_tasks: list[asyncio.Task[None]] = []
            for state in self._typing_placeholders.values():
                state.stop_event.set()
                task = state.task
                if task and not task.done():
                    placeholder_tasks.append(task)
                    task.cancel()
            if placeholder_tasks:
                await asyncio.gather(*placeholder_tasks, return_exceptions=True)
            self._typing_placeholders.clear()
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

    @staticmethod
    def _resolve_reply_target(message: OutboundMessage) -> str | None:
        """Return the message id this outbound message should reply to."""
        return message.reply_to or message.metadata.get("message_id")

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
        reply_target = self._resolve_reply_target(message)
        placeholder_state = (
            self._typing_placeholders.pop(reply_target, None)
            if self.typing_mode == _TYPING_MODE_PLACEHOLDER and reply_target
            else None
        )

        if not chat_id and not reply_target and placeholder_state is None:
            logger.error(f"[{self.full_name}] No chat_id in message metadata")
            return

        chunks = self._split_message(message.content, SAFE_MESSAGE_LENGTH)

        for i, chunk in enumerate(chunks):
            out_msg_type = self._determine_msg_type(chunk)
            if i == 0 and placeholder_state is not None:
                updated = await self._edit_message(
                    placeholder_state.placeholder_message_id,
                    chunk,
                    MSG_TYPE_INTERACTIVE,
                )
                if updated:
                    if i < len(chunks) - 1:
                        await asyncio.sleep(0.5)
                    continue
            reply_to = reply_target if i == 0 else None
            await self._send_api_message(
                chat_id=chat_id,
                content=chunk,
                msg_type=out_msg_type,
                reply_to=reply_to,
            )
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)
