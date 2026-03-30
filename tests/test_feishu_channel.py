"""Tests for Feishu channel."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spoon_bot.channels.base import ChannelConfig, ChannelMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_feishu_config(
    *,
    mode: ChannelMode = ChannelMode.GATEWAY,
    webhook_path: str | None = None,
    webhook_secret: str | None = None,
    **extra_kwargs,
) -> ChannelConfig:
    """Return a minimal ChannelConfig for FeishuChannel."""
    defaults = {
        "app_id": "cli_test",
        "app_secret": "secret",
        "verification_token": "",
        "encrypt_key": "",
        "domain": "feishu",
        "dm_policy": "open",
        "allow_from": [],
        "group_policy": "open",
        "group_allow_from": [],
        "group_sender_allow_from": [],
        "group_session_scope": "group_sender",
        "group_agent_config": {},
        "allowed_chats": [],
        "allowed_users": [],
        "require_mention": True,
    }
    defaults.update(extra_kwargs)
    return ChannelConfig(
        name="feishu",
        mode=mode,
        webhook_path=webhook_path,
        webhook_secret=webhook_secret,
        **defaults,
    )


# ---------------------------------------------------------------------------
# FeishuChannel unit tests (lark-oapi dependency is mocked)
# ---------------------------------------------------------------------------

class TestFeishuChannel:
    """Unit tests for FeishuChannel that mock lark-oapi."""

    def _make_channel(self, **extra_kwargs):
        """Create a FeishuChannel with lark-oapi mocked out."""
        with patch("spoon_bot.channels.feishu.channel.FEISHU_AVAILABLE", True), \
             patch("spoon_bot.channels.feishu.channel.lark", create=True), \
             patch("spoon_bot.channels.feishu.channel.CreateMessageRequest", create=True), \
             patch("spoon_bot.channels.feishu.channel.CreateMessageRequestBody", create=True), \
             patch("spoon_bot.channels.feishu.channel.ReplyMessageRequest", create=True), \
             patch("spoon_bot.channels.feishu.channel.ReplyMessageRequestBody", create=True):
            from spoon_bot.channels.feishu.channel import FeishuChannel
            config = _make_feishu_config(**extra_kwargs)
            return FeishuChannel(config, "testbot")

    def test_init_requires_app_id_and_secret(self):
        """Missing app_id or app_secret raises ValueError."""
        with patch("spoon_bot.channels.feishu.channel.FEISHU_AVAILABLE", True), \
             patch("spoon_bot.channels.feishu.channel.lark", create=True), \
             patch("spoon_bot.channels.feishu.channel.CreateMessageRequest", create=True), \
             patch("spoon_bot.channels.feishu.channel.CreateMessageRequestBody", create=True), \
             patch("spoon_bot.channels.feishu.channel.ReplyMessageRequest", create=True), \
             patch("spoon_bot.channels.feishu.channel.ReplyMessageRequestBody", create=True):
            from spoon_bot.channels.feishu.channel import FeishuChannel

            config_no_id = _make_feishu_config(app_id="", app_secret="s")
            with pytest.raises(ValueError, match="app_id"):
                FeishuChannel(config_no_id, "bot")

            config_no_secret = _make_feishu_config(app_id="id", app_secret="")
            with pytest.raises(ValueError, match="app_secret"):
                FeishuChannel(config_no_secret, "bot")

    def test_package_exports_channel_class(self):
        """The feishu package should expose FeishuChannel via __init__.py."""
        from spoon_bot.channels.feishu import FeishuChannel

        assert FeishuChannel.__name__ == "FeishuChannel"

    def test_materialize_image_media_to_workspace_relative_path(self, tmp_path):
        """Inbound image messages should become workspace-backed file paths."""
        ch = self._make_channel()
        ch._agent_loop = SimpleNamespace(workspace=tmp_path)
        ch._media = SimpleNamespace(download_image=lambda image_key: b"\x89PNG\r\n\x1a\npng")

        media, attachments = ch._materialize_inbound_media(
            message_id="msg-1",
            content={"image_key": "img_123"},
            msg_type="image",
        )

        assert len(media) == 1
        stored = tmp_path / Path(media[0])
        assert stored.is_file()
        assert attachments[0]["workspace_path"] == media[0]
        assert attachments[0]["mime_type"] == "image/png"

    def test_materialize_file_attachment_without_multimodal_media(self, tmp_path):
        """Non-image Feishu files should still be exposed as agent attachments."""
        ch = self._make_channel()
        ch._agent_loop = SimpleNamespace(workspace=tmp_path)
        ch._media = SimpleNamespace(
            download_file=lambda message_id, file_key, resource_type: b"hello"
        )

        media, attachments = ch._materialize_inbound_media(
            message_id="msg-2",
            content={"file_key": "file_123", "file_name": "notes.txt"},
            msg_type="file",
        )

        assert media == []
        assert attachments[0]["workspace_path"].endswith(".txt")
        assert attachments[0]["mime_type"] == "text/plain"

    def test_resolve_reply_target_prefers_reply_to(self):
        """Generic reply_to should win over Feishu-specific metadata fallback."""
        from spoon_bot.bus.events import OutboundMessage

        ch = self._make_channel()
        outbound = OutboundMessage(
            content="hello",
            reply_to="reply-mid",
            metadata={"message_id": "meta-mid", "chat_id": "chat-1"},
        )

        assert ch._resolve_reply_target(outbound) == "reply-mid"

    def test_init_extracts_config(self):
        """FeishuChannel reads all fields from config.extra."""
        ch = self._make_channel(
            app_id="myapp",
            app_secret="mysecret",
            domain="lark",
            allowed_chats=["chat1"],
            allowed_users=["ou_user1"],
            require_mention=False,
            typing_indicator=False,
            typing_mode="placeholder",
            typing_emoji="THINK",
        )
        assert ch.app_id == "myapp"
        assert ch.app_secret == "mysecret"
        assert ch.domain == "lark"
        assert "chat1" in ch.allowed_chats
        assert "ou_user1" in ch.allowed_users
        assert ch.require_mention is False
        assert ch.typing_indicator is False
        assert ch.typing_mode == "placeholder"
        assert ch.typing_emoji == "THINK"
        assert ch.group_policy == "open"
        assert ch.group_session_scope == "group_sender"

    def test_init_defaults_typing_emoji_to_typing(self):
        """Feishu typing indicator should default to the Typing emoji."""
        ch = self._make_channel()

        assert ch.typing_indicator is True
        assert ch.typing_mode == "reaction"
        assert ch.typing_emoji == "Typing"

    def test_build_api_client_uses_configured_domain(self):
        """Outbound API client should honor the configured Lark domain."""
        ch = self._make_channel(domain="lark")

        with patch("spoon_bot.channels.feishu.channel.lark", create=True) as mock_lark:
            builder = MagicMock()
            fake_client = object()
            builder.app_id.return_value = builder
            builder.app_secret.return_value = builder
            builder.domain.return_value = builder
            builder.log_level.return_value = builder
            builder.build.return_value = fake_client
            mock_lark.Client.builder.return_value = builder
            mock_lark.LogLevel.INFO = "info"

            assert ch._build_api_client() is fake_client

        builder.app_id.assert_called_once_with("cli_test")
        builder.app_secret.assert_called_once_with("secret")
        builder.domain.assert_called_once_with("https://open.larksuite.com")
        builder.log_level.assert_called_once_with("info")
        builder.build.assert_called_once_with()

    def test_build_ws_client_uses_configured_domain(self):
        """WS client should honor the configured Lark domain."""
        ch = self._make_channel(domain="lark")
        ch._ws_event_handler = object()

        with patch("spoon_bot.channels.feishu.channel.lark", create=True) as mock_lark:
            fake_client = object()
            mock_lark.LogLevel.DEBUG = "debug"
            mock_lark.ws.Client.return_value = fake_client

            assert ch._build_ws_client() is fake_client

        mock_lark.ws.Client.assert_called_once_with(
            "cli_test",
            "secret",
            event_handler=ch._ws_event_handler,
            log_level="debug",
            domain="https://open.larksuite.com",
        )

    def test_build_event_handler_registers_chat_access_event(self):
        """WS event handler should register the p2p chat access event as a no-op."""
        ch = self._make_channel()
        registered: dict[str, object] = {}
        built_handler = object()

        class FakeBuilder:
            def register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(self, callback):
                registered["chat_access"] = callback
                return self

            def register_p2_im_message_receive_v1(self, callback):
                registered["message_receive"] = callback
                return self

            def build(self):
                return built_handler

        with patch("spoon_bot.channels.feishu.channel.lark", create=True) as mock_lark:
            mock_lark.EventDispatcherHandler.builder.return_value = FakeBuilder()
            mock_lark.LogLevel.DEBUG = "debug"

            assert ch._build_event_handler() is built_handler

        assert "chat_access" in registered
        assert "message_receive" in registered

        ch.publish = AsyncMock()
        registered["chat_access"](SimpleNamespace(event=SimpleNamespace(chat_id="oc_chat")))
        ch.publish.assert_not_called()

    def test_split_message_short(self):
        """Short messages are returned as a single chunk."""
        from spoon_bot.channels.feishu.channel import FeishuChannel
        result = FeishuChannel._split_message("hello world", 3800)
        assert result == ["hello world"]

    def test_split_message_splits_at_newline(self):
        """Long messages split preferring newlines."""
        from spoon_bot.channels.feishu.channel import FeishuChannel
        line = "y" * 2000
        content = f"{line}\n{line}"
        chunks = FeishuChannel._split_message(content, 3800)
        assert all(len(c) <= 3800 for c in chunks)
        assert len(chunks) > 1

    def test_split_message_hard_cut(self):
        """Messages with no whitespace are hard-cut at the limit."""
        from spoon_bot.channels.feishu.channel import FeishuChannel
        content = "z" * 8000
        chunks = FeishuChannel._split_message(content, 3800)
        assert all(len(c) <= 3800 for c in chunks)
        assert "".join(chunks) == content

    def test_strip_mention_tokens(self):
        """@_user_N tokens are removed from text."""
        from spoon_bot.channels.feishu.channel import FeishuChannel
        text = "@_user_1 hello @_user_2 world"
        result = FeishuChannel._strip_mention_tokens(text)
        assert result == "hello  world"

    def test_strip_mention_tokens_no_mentions(self):
        """Text without mentions is returned unchanged."""
        from spoon_bot.channels.feishu.channel import FeishuChannel
        text = "plain message"
        assert FeishuChannel._strip_mention_tokens(text) == "plain message"

    def test_check_access_user_allowlist(self):
        """Users not in allowlist are rejected."""
        ch = self._make_channel(allowed_users=["ou_allowed"])
        assert ch._check_access("ou_other", "chat1", "p2p", None) is False

    def test_check_access_user_allowed(self):
        """Users in allowlist are accepted."""
        ch = self._make_channel(allowed_users=["ou_allowed"])
        assert ch._check_access("ou_allowed", "chat1", "p2p", None) is True

    def test_check_access_chat_allowlist(self):
        """Chats not in allowlist are rejected."""
        ch = self._make_channel(allowed_chats=["oc_allowed"])
        assert ch._check_access("ou_anyone", "oc_other", "group", None) is False

    def test_check_access_group_policy_allowlist_blocks_unlisted_chat(self):
        """Group allowlist should block chats outside group_allow_from."""
        ch = self._make_channel(group_policy="allowlist", group_allow_from=["oc_allowed"])
        assert ch._check_access("ou_user", "oc_other", "group", None) is False

    def test_check_access_group_policy_allowlist_allows_listed_chat(self):
        """Group allowlist should allow explicitly listed chats."""
        ch = self._make_channel(
            group_policy="allowlist",
            group_allow_from=["oc_allowed"],
            require_mention=False,
        )
        assert ch._check_access("ou_user", "oc_allowed", "group", None) is True

    def test_check_access_group_sender_allowlist(self):
        """Only approved senders may trigger the bot inside allowed groups."""
        ch = self._make_channel(
            group_policy="allowlist",
            group_allow_from=["oc_allowed"],
            group_sender_allow_from=["ou_allowed"],
            require_mention=False,
        )
        assert ch._check_access("ou_other", "oc_allowed", "group", None) is False
        assert ch._check_access("ou_allowed", "oc_allowed", "group", None) is True

    def test_check_access_group_mention_required(self):
        """Group messages without @mention are rejected when require_mention=True."""
        ch = self._make_channel(require_mention=True)
        ch._bot_open_id = "ou_bot"

        # Empty mentions list — bot not mentioned
        assert ch._check_access("ou_user", "chat1", "group", []) is False

    def test_check_access_dm_no_mention_required(self):
        """DMs are accepted without @mention even when require_mention=True."""
        ch = self._make_channel(require_mention=True)
        ch._bot_open_id = "ou_bot"
        # p2p chat — mention check is skipped
        assert ch._check_access("ou_user", "chat1", "p2p", None) is True

    def test_check_access_dm_allowlist(self):
        """DM allowlist should reject unapproved senders."""
        ch = self._make_channel(dm_policy="allowlist", allow_from=["ou_allowed"])
        assert ch._check_access("ou_other", "chat1", "p2p", None) is False
        assert ch._check_access("ou_allowed", "chat1", "p2p", None) is True

    def test_check_access_dm_disabled(self):
        """DM policy disabled should reject direct messages."""
        ch = self._make_channel(dm_policy="disabled")
        assert ch._check_access("ou_user", "chat1", "p2p", None) is False

    def test_check_access_group_with_mention(self):
        """Group messages with the bot mentioned are accepted."""
        ch = self._make_channel(require_mention=True)
        ch._bot_open_id = "ou_bot"

        mention = MagicMock()
        mention.id.open_id = "ou_bot"

        assert ch._check_access("ou_user", "chat1", "group", [mention]) is True

    def test_is_bot_mentioned_no_open_id_yet(self):
        """When bot_open_id is unknown, _is_bot_mentioned returns True (accept all)."""
        ch = self._make_channel()
        ch._bot_open_id = None
        # Any non-empty mentions should return True
        assert ch._is_bot_mentioned([MagicMock()]) is True

    def test_is_bot_mentioned_empty_list(self):
        """Empty mentions list returns False."""
        ch = self._make_channel()
        ch._bot_open_id = "ou_bot"
        assert ch._is_bot_mentioned([]) is False

    def test_build_session_key_isolates_group_senders_by_default(self):
        """Default group_session_scope should isolate each sender in group chats."""
        ch = self._make_channel(group_session_scope="group_sender")

        alice = ch._build_session_key(
            chat_id="oc_group",
            chat_type="group",
            sender_id="ou_alice",
        )
        bob = ch._build_session_key(
            chat_id="oc_group",
            chat_type="group",
            sender_id="ou_bob",
        )

        assert alice != bob
        assert alice.endswith("oc_group:sender:ou_alice")
        assert bob.endswith("oc_group:sender:ou_bob")

    def test_build_session_key_can_share_group_context(self):
        """group scope should keep one shared session per chat."""
        ch = self._make_channel(group_session_scope="group")

        alice = ch._build_session_key(
            chat_id="oc_group",
            chat_type="group",
            sender_id="ou_alice",
        )
        bob = ch._build_session_key(
            chat_id="oc_group",
            chat_type="group",
            sender_id="ou_bob",
        )

        assert alice == bob == "feishu_testbot_oc_group"

    def test_dedup_cache_enforces_max_size(self):
        """Dedup cache should drop the oldest ids after reaching the hard cap."""
        ch = self._make_channel()

        with patch("spoon_bot.channels.feishu.channel.MESSAGE_DEDUP_MAX", 2), \
             patch("spoon_bot.channels.feishu.channel.MESSAGE_DEDUP_TTL", 300), \
             patch("spoon_bot.channels.feishu.channel.time.monotonic", side_effect=[1.0, 2.0, 3.0]):
            assert ch._is_duplicate_message("m1") is False
            assert ch._is_duplicate_message("m2") is False
            assert ch._is_duplicate_message("m3") is False

        assert set(ch._processed_messages) == {"m2", "m3"}

    def test_dedup_cache_expires_old_entries(self):
        """Expired dedup entries should not block a message forever."""
        ch = self._make_channel()

        with patch("spoon_bot.channels.feishu.channel.MESSAGE_DEDUP_TTL", 10), \
             patch("spoon_bot.channels.feishu.channel.time.monotonic", side_effect=[100.0, 111.0]):
            assert ch._is_duplicate_message("msg-1") is False
            assert ch._is_duplicate_message("msg-1") is False

        assert ch._processed_messages == {"msg-1": 111.0}

    def test_normalize_epoch_ms_accepts_seconds_and_milliseconds(self):
        """Epoch timestamps should normalize to milliseconds."""
        ch = self._make_channel()

        assert ch._normalize_epoch_ms(1711795200) == 1711795200000
        assert ch._normalize_epoch_ms(1711795200000) == 1711795200000
        assert ch._normalize_epoch_ms("1711795200") == 1711795200000

    def test_extract_typing_backoff_code_from_response_and_exception_shapes(self):
        """Backoff helper should detect Feishu quota/rate-limit codes."""
        ch = self._make_channel()

        assert ch._extract_typing_backoff_code(SimpleNamespace(code=429)) == 429
        assert ch._extract_typing_backoff_code(
            SimpleNamespace(response=SimpleNamespace(status=429, data=None))
        ) == 429
        assert ch._extract_typing_backoff_code(
            SimpleNamespace(response=SimpleNamespace(status=500, data=SimpleNamespace(code=99991403)))
        ) == 99991403
        assert ch._extract_typing_backoff_code(SimpleNamespace(code=500)) is None

    @pytest.mark.asyncio
    async def test_on_processing_start_uses_configured_typing_emoji(self):
        """Typing reaction should use the configured emoji and store the reaction id."""
        ch = self._make_channel(typing_emoji="THINK")
        message = SimpleNamespace(metadata={"message_id": "msg-1"})

        with patch.object(ch, "_add_reaction", AsyncMock(return_value="reaction-1")) as mock_add:
            await ch.on_processing_start(message)

        mock_add.assert_awaited_once_with("msg-1", "THINK")
        assert ch._typing_reactions["msg-1"] == "reaction-1"

    @pytest.mark.asyncio
    async def test_on_processing_start_skips_when_typing_indicator_disabled(self):
        """Typing indicator can be disabled per Feishu account config."""
        ch = self._make_channel(typing_indicator=False)
        message = SimpleNamespace(metadata={"message_id": "msg-1"})

        with patch.object(ch, "_add_reaction", AsyncMock()) as mock_add:
            await ch.on_processing_start(message)

        mock_add.assert_not_awaited()
        assert ch._typing_reactions == {}

    @pytest.mark.asyncio
    async def test_on_processing_start_skips_when_typing_reaction_already_active(self):
        """Typing indicator should not be re-added when the message is already marked active."""
        ch = self._make_channel()
        ch._typing_reactions["msg-1"] = "reaction-1"
        message = SimpleNamespace(metadata={"message_id": "msg-1"})

        with patch.object(ch, "_add_reaction", AsyncMock()) as mock_add:
            await ch.on_processing_start(message)

        mock_add.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_processing_start_skips_for_stale_messages(self):
        """Old replayed messages should not get a fresh typing indicator."""
        ch = self._make_channel()
        message = SimpleNamespace(
            metadata={
                "message_id": "msg-1",
                "message_create_time": 1711795200000,
            }
        )

        with patch("spoon_bot.channels.feishu.channel.time.time", return_value=1711795381), \
             patch.object(ch, "_add_reaction", AsyncMock()) as mock_add:
            await ch.on_processing_start(message)

        mock_add.assert_not_awaited()
        assert ch._typing_reactions == {}

    @pytest.mark.asyncio
    async def test_on_processing_start_skips_when_typing_backoff_active(self):
        """Rate-limit backoff should temporarily suppress new typing indicators."""
        ch = self._make_channel()
        message = SimpleNamespace(metadata={"message_id": "msg-1"})

        with patch("spoon_bot.channels.feishu.channel.time.monotonic", return_value=10.0), \
             patch.object(ch, "_add_reaction", AsyncMock()) as mock_add:
            ch._typing_backoff_until = 20.0
            await ch.on_processing_start(message)

        mock_add.assert_not_awaited()
        assert ch._typing_reactions == {}

    @pytest.mark.asyncio
    async def test_on_processing_end_removes_active_typing_reaction(self):
        """Typing reaction should be removed when processing completes."""
        ch = self._make_channel()
        ch._typing_reactions["msg-1"] = "reaction-1"
        message = SimpleNamespace(metadata={"message_id": "msg-1"})

        with patch.object(ch, "_remove_reaction", AsyncMock()) as mock_remove:
            await ch.on_processing_end(message)

        mock_remove.assert_awaited_once_with("msg-1", "reaction-1")
        assert "msg-1" not in ch._typing_reactions

    @pytest.mark.asyncio
    async def test_on_processing_start_placeholder_mode_creates_placeholder_and_task(self):
        """Placeholder typing mode should send an initial reply and start the animation loop."""
        ch = self._make_channel(typing_mode="placeholder")
        message = SimpleNamespace(metadata={"message_id": "msg-1", "chat_id": "chat-1"})

        async def fake_loop(_msg_id, state):
            await state.stop_event.wait()

        mock_loop = AsyncMock(side_effect=fake_loop)
        with patch.object(ch, "_send_api_message", AsyncMock(return_value="typing-mid")) as mock_send, \
             patch.object(ch, "_typing_placeholder_loop", mock_loop):
            await ch.on_processing_start(message)
            await asyncio.sleep(0)
            state = ch._typing_placeholders["msg-1"]
            await ch.on_processing_end(message)

        mock_send.assert_awaited_once_with(
            chat_id="chat-1",
            content="Typing.",
            msg_type="interactive",
            reply_to="msg-1",
        )
        assert state.placeholder_message_id == "typing-mid"
        assert state.stop_event.is_set()
        assert state.task is not None and state.task.done()

    @pytest.mark.asyncio
    async def test_on_processing_start_placeholder_mode_skips_stale_messages(self):
        """Placeholder typing should not start for stale replayed messages."""
        ch = self._make_channel(typing_mode="placeholder")
        message = SimpleNamespace(
            metadata={
                "message_id": "msg-1",
                "chat_id": "chat-1",
                "message_create_time": 1711795200000,
            }
        )

        with patch("spoon_bot.channels.feishu.channel.time.time", return_value=1711795381), \
             patch.object(ch, "_send_api_message", AsyncMock()) as mock_send:
            await ch.on_processing_start(message)

        mock_send.assert_not_awaited()
        assert ch._typing_placeholders == {}

    @pytest.mark.asyncio
    async def test_send_placeholder_mode_edits_first_chunk_and_sends_remainder(self):
        """Placeholder typing should patch the first chunk into the placeholder message."""
        from spoon_bot.bus.events import OutboundMessage

        ch = self._make_channel(typing_mode="placeholder")
        ch._api_client = object()
        ch._typing_placeholders["reply-mid"] = SimpleNamespace(
            placeholder_message_id="typing-mid",
            stop_event=asyncio.Event(),
            task=None,
        )
        outbound = OutboundMessage(
            content="ignored",
            reply_to="reply-mid",
            metadata={"chat_id": "chat-1"},
        )

        with patch.object(ch, "_split_message", return_value=["first", "second"]), \
             patch.object(ch, "_determine_msg_type", side_effect=["text", "text"]), \
             patch.object(ch, "_edit_message", AsyncMock(return_value=True)) as mock_edit, \
             patch.object(ch, "_send_api_message", AsyncMock(return_value="msg-2")) as mock_send:
            await ch.send(outbound)

        mock_edit.assert_awaited_once_with("typing-mid", "first", "interactive")
        mock_send.assert_awaited_once_with(
            chat_id="chat-1",
            content="second",
            msg_type="text",
            reply_to=None,
        )
        assert "reply-mid" not in ch._typing_placeholders

    @pytest.mark.asyncio
    async def test_send_placeholder_mode_falls_back_to_reply_when_edit_fails(self):
        """If patching the placeholder fails, the first chunk should fall back to a normal reply."""
        from spoon_bot.bus.events import OutboundMessage

        ch = self._make_channel(typing_mode="placeholder")
        ch._api_client = object()
        ch._typing_placeholders["reply-mid"] = SimpleNamespace(
            placeholder_message_id="typing-mid",
            stop_event=asyncio.Event(),
            task=None,
        )
        outbound = OutboundMessage(
            content="ignored",
            reply_to="reply-mid",
            metadata={"chat_id": "chat-1"},
        )

        with patch.object(ch, "_split_message", return_value=["first"]), \
             patch.object(ch, "_determine_msg_type", return_value="text"), \
             patch.object(ch, "_edit_message", AsyncMock(return_value=False)) as mock_edit, \
             patch.object(ch, "_send_api_message", AsyncMock(return_value="msg-1")) as mock_send:
            await ch.send(outbound)

        mock_edit.assert_awaited_once_with("typing-mid", "first", "interactive")
        mock_send.assert_awaited_once_with(
            chat_id="chat-1",
            content="first",
            msg_type="text",
            reply_to="reply-mid",
        )
        assert "reply-mid" not in ch._typing_placeholders

    @pytest.mark.asyncio
    async def test_handle_webhook_uses_sdk_event_dispatcher(self):
        """Webhook mode should delegate raw requests to the SDK dispatcher."""
        ch = self._make_channel(mode=ChannelMode.WEBHOOK, webhook_path="/feishu/events")

        def _dispatch(req):
            assert req.uri == "/feishu/events"
            assert req.body == b'{"type":"url_verification"}'
            assert req.headers.get("X-Lark-Request-Timestamp") == "123"
            return SimpleNamespace(status_code=200, content=b'{"challenge":"abc"}')

        class DummyRequest:
            headers = {"x-lark-request-timestamp": "123"}
            url = SimpleNamespace(path="/feishu/events")

            async def body(self):
                return b'{"type":"url_verification"}'

        ch._event_handler = SimpleNamespace(do=_dispatch)

        assert await ch.handle_webhook(DummyRequest()) == {"challenge": "abc"}

    @pytest.mark.asyncio
    async def test_handle_webhook_runs_dispatcher_off_thread(self):
        """Webhook dispatch should use a worker thread to avoid blocking the loop."""
        ch = self._make_channel(mode=ChannelMode.WEBHOOK, webhook_path="/feishu/events")

        class DummyRequest:
            headers = {}
            url = SimpleNamespace(path="/feishu/events")

            async def body(self):
                return b"{}"

        thread_ids: list[int] = []

        def _dispatch(_req):
            import threading as _threading

            thread_ids.append(_threading.get_ident())
            return SimpleNamespace(status_code=200, content=b'{"ok":true}')

        ch._event_handler = SimpleNamespace(do=_dispatch)

        result = await ch.handle_webhook(DummyRequest())

        assert result == {"ok": True}
        assert thread_ids and thread_ids[0] != threading.get_ident()

    @pytest.mark.asyncio
    async def test_stop_disconnects_websocket_thread(self):
        """stop() should disable reconnect, disconnect, and join the WS thread."""
        ch = self._make_channel()
        ready = threading.Event()
        loop = asyncio.new_event_loop()

        def _run_loop():
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            loop.run_forever()
            loop.close()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        assert ready.wait(1), "worker event loop did not start"

        disconnect_called = threading.Event()

        class DummyClient:
            def __init__(self):
                self._auto_reconnect = True

            async def _disconnect(self):
                disconnect_called.set()

        ch._running = True
        ch._ws_client = DummyClient()
        ch._ws_loop = loop
        ch._ws_thread = thread

        await ch.stop()

        assert disconnect_called.is_set()
        assert ch._ws_client is None
        assert not thread.is_alive()


# ---------------------------------------------------------------------------
# Feishu config loading tests
# ---------------------------------------------------------------------------

class TestFeishuConfig:
    """Tests for Feishu config loading in ChannelsConfig."""

    def _make_yaml(self, extra_account: dict, tmp_path, mode: str = "ws") -> str:
        import yaml
        content = {
            "channels": {
                "feishu": {
                    "enabled": True,
                    "accounts": [{
                        "name": "bot",
                        "app_id": "cli_test",
                        "app_secret": "secret",
                        "mode": mode,
                        **extra_account,
                    }],
                }
            }
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(content))
        return str(p)

    def test_ws_mode_maps_to_gateway(self, tmp_path):
        """mode: ws (default) maps to ChannelMode.GATEWAY."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({}, tmp_path, mode="ws")
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.mode == ChannelMode.GATEWAY

    def test_webhook_mode_maps_to_webhook(self, tmp_path):
        """mode: webhook maps to ChannelMode.WEBHOOK."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({}, tmp_path, mode="webhook")
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.mode == ChannelMode.WEBHOOK

    def test_domain_passed_through(self, tmp_path):
        """domain field is passed into ChannelConfig.extra."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({"domain": "lark"}, tmp_path)
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.extra["domain"] == "lark"

    def test_domain_defaults_to_feishu(self, tmp_path):
        """When domain is not specified, it defaults to 'feishu'."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({}, tmp_path)
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.extra["domain"] == "feishu"

    def test_allowed_chats_passed_through(self, tmp_path):
        """allowed_chats list is included in ChannelConfig.extra."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({"allowed_chats": ["oc_abc123"]}, tmp_path)
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert "oc_abc123" in config.extra["allowed_chats"]
        assert "oc_abc123" in config.extra["group_allow_from"]

    def test_dm_and_group_policies_default_to_safe_values(self, tmp_path):
        """Feishu config should default DMs open but groups allowlisted and isolated by sender."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({}, tmp_path)
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.extra["dm_policy"] == "open"
        assert config.extra["allow_from"] == []
        assert config.extra["group_policy"] == "allowlist"
        assert config.extra["group_allow_from"] == []
        assert config.extra["group_sender_allow_from"] == []
        assert config.extra["group_session_scope"] == "group_sender"
        assert config.extra["group_agent_config"] == {}

    def test_group_access_settings_pass_through(self, tmp_path):
        """Feishu-specific DM/group policies should be preserved in ChannelConfig.extra."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml(
            {
                "dm_policy": "allowlist",
                "allow_from": ["ou_me"],
                "group_policy": "allowlist",
                "group_allow_from": ["oc_team"],
                "group_sender_allow_from": ["ou_me"],
                "group_session_scope": "group",
                "group_agent_config": {"tool_profile": "group_safe"},
            },
            tmp_path,
        )
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.extra["dm_policy"] == "allowlist"
        assert config.extra["allow_from"] == ["ou_me"]
        assert config.extra["group_policy"] == "allowlist"
        assert config.extra["group_allow_from"] == ["oc_team"]
        assert config.extra["group_sender_allow_from"] == ["ou_me"]
        assert config.extra["group_session_scope"] == "group"
        assert config.extra["group_agent_config"] == {"tool_profile": "group_safe"}

    def test_require_mention_default_true(self, tmp_path):
        """require_mention defaults to True."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({}, tmp_path)
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.extra["require_mention"] is True

    def test_require_mention_can_be_disabled(self, tmp_path):
        """require_mention can be set to False."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({"require_mention": False}, tmp_path)
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.extra["require_mention"] is False

    def test_typing_indicator_defaults_true(self, tmp_path):
        """typing_indicator defaults to True for Feishu accounts."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({}, tmp_path)
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.extra["typing_indicator"] is True
        assert config.extra["typing_mode"] == "reaction"
        assert config.extra["typing_emoji"] == "Typing"

    def test_typing_settings_passed_through(self, tmp_path):
        """typing indicator settings are passed into ChannelConfig.extra."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml(
            {
                "typing_indicator": False,
                "typing_mode": "placeholder",
                "typing_emoji": "THINK",
            },
            tmp_path,
        )
        cfg = load_channels_config(path)
        config, _ = cfg.get_feishu_configs()[0]
        assert config.extra["typing_indicator"] is False
        assert config.extra["typing_mode"] == "placeholder"
        assert config.extra["typing_emoji"] == "THINK"
