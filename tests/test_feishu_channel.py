"""Tests for Feishu channel."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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

    def test_init_extracts_config(self):
        """FeishuChannel reads all fields from config.extra."""
        ch = self._make_channel(
            app_id="myapp",
            app_secret="mysecret",
            domain="lark",
            allowed_chats=["chat1"],
            allowed_users=["ou_user1"],
            require_mention=False,
        )
        assert ch.app_id == "myapp"
        assert ch.app_secret == "mysecret"
        assert ch.domain == "lark"
        assert "chat1" in ch.allowed_chats
        assert "ou_user1" in ch.allowed_users
        assert ch.require_mention is False

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
