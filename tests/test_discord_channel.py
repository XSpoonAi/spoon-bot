"""Tests for Discord channel."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spoon_bot.channels.base import ChannelConfig, ChannelMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_discord_config(**extra_kwargs) -> ChannelConfig:
    """Return a minimal ChannelConfig for DiscordChannel."""
    defaults = {
        "token": "test-discord-token",
        "allowed_guilds": [],
        "allowed_users": [],
        "proxy_url": None,
        "require_mention": True,
        "allow_dm": True,
        "intents": [],
    }
    defaults.update(extra_kwargs)
    return ChannelConfig(name="discord", mode=ChannelMode.GATEWAY, **defaults)


# ---------------------------------------------------------------------------
# DiscordChannel unit tests (discord.py dependency is mocked)
# ---------------------------------------------------------------------------

class TestDiscordChannel:
    """Unit tests for DiscordChannel that mock discord.py."""

    def _make_channel(self, **extra_kwargs):
        """Create a DiscordChannel with discord.py mocked out."""
        with patch("spoon_bot.channels.discord.channel.DISCORD_AVAILABLE", True), \
             patch("spoon_bot.channels.discord.channel.discord", create=True):
            from spoon_bot.channels.discord.channel import DiscordChannel
            config = _make_discord_config(**extra_kwargs)
            return DiscordChannel(config, "testbot")

    def test_init_requires_token(self):
        """Missing token raises ValueError."""
        with patch("spoon_bot.channels.discord.channel.DISCORD_AVAILABLE", True), \
             patch("spoon_bot.channels.discord.channel.discord", create=True):
            from spoon_bot.channels.discord.channel import DiscordChannel
            config = _make_discord_config(token="")
            with pytest.raises(ValueError, match="token"):
                DiscordChannel(config, "testbot")

    def test_init_extracts_config(self):
        """DiscordChannel reads all fields from config.extra."""
        ch = self._make_channel(
            token="tok123",
            allowed_guilds=[111, 222],
            allowed_users=[999],
            proxy_url="http://127.0.0.1:7897",
            require_mention=False,
            allow_dm=False,
        )
        assert ch.token == "tok123"
        assert 111 in ch.allowed_guilds
        assert 999 in ch.allowed_users
        assert ch.proxy_url == "http://127.0.0.1:7897"
        assert ch.require_mention is False
        assert ch.allow_dm is False

    def test_split_message_short(self):
        """Short messages are returned as a single chunk."""
        from spoon_bot.channels.discord.channel import DiscordChannel
        result = DiscordChannel._split_message("hello", 1950)
        assert result == ["hello"]

    def test_split_message_splits_at_newline(self):
        """Long messages are split preferring newline positions."""
        from spoon_bot.channels.discord.channel import DiscordChannel
        line = "x" * 1000
        content = f"{line}\n{line}\n{line}"
        chunks = DiscordChannel._split_message(content, 1950)
        assert all(len(c) <= 1950 for c in chunks)
        assert "".join(chunks) == content.replace("\n", "")

    def test_split_message_hard_cut(self):
        """Messages with no whitespace are hard-cut at the limit."""
        from spoon_bot.channels.discord.channel import DiscordChannel
        content = "x" * 4000
        chunks = DiscordChannel._split_message(content, 1950)
        assert all(len(c) <= 1950 for c in chunks)
        assert "".join(chunks) == content

    def test_check_access_user_allowlist(self):
        """Users not in allowlist are rejected."""
        ch = self._make_channel(allowed_users=[111])

        msg = MagicMock()
        msg.author.id = 999
        msg.author.bot = False

        import discord as discord_mock
        with patch("spoon_bot.channels.discord.channel.discord") as d:
            d.DMChannel = MagicMock  # make isinstance check work with MagicMock
            msg.channel.__class__ = object  # not a DMChannel
            assert ch._check_access(msg) is False

    def test_check_access_dm_disabled(self):
        """DMs are rejected when allow_dm=False."""
        ch = self._make_channel(allow_dm=False, allowed_users=[])

        with patch("spoon_bot.channels.discord.channel.discord") as d:
            d.DMChannel = MagicMock
            msg = MagicMock()
            msg.author.id = 42
            msg.author.bot = False
            # Make isinstance(message.channel, discord.DMChannel) return True
            msg.channel.__class__ = d.DMChannel
            assert ch._check_access(msg) is False

    def test_check_access_guild_allowlist(self):
        """Messages from guilds not in the allowlist are rejected."""
        ch = self._make_channel(allowed_guilds=[500])

        with patch("spoon_bot.channels.discord.channel.discord") as d:
            d.DMChannel = type("DMChannel", (), {})  # different class than MagicMock
            msg = MagicMock()
            msg.author.id = 42
            msg.author.bot = False
            msg.channel.__class__ = object  # not a DMChannel
            msg.guild.id = 999  # not in allowlist
            ch._client = MagicMock()
            ch._client.user = MagicMock()
            # mention check: bot not mentioned
            ch._client.user.mentioned_in.return_value = False
            assert ch._check_access(msg) is False

    def test_check_access_mention_required(self):
        """Guild messages that don't mention the bot are rejected."""
        ch = self._make_channel(require_mention=True, allowed_guilds=[])

        with patch("spoon_bot.channels.discord.channel.discord") as d:
            d.DMChannel = type("DMChannel", (), {})
            msg = MagicMock()
            msg.author.id = 42
            msg.author.bot = False
            msg.channel.__class__ = object
            msg.guild.id = 100
            ch._client = MagicMock()
            ch._client.user = MagicMock()
            ch._client.user.mentioned_in.return_value = False
            assert ch._check_access(msg) is False

    def test_check_access_mention_present_passes(self):
        """Guild messages that do mention the bot are accepted."""
        ch = self._make_channel(require_mention=True, allowed_guilds=[])

        with patch("spoon_bot.channels.discord.channel.discord") as d:
            d.DMChannel = type("DMChannel", (), {})
            msg = MagicMock()
            msg.author.id = 42
            msg.author.bot = False
            msg.channel.__class__ = object
            msg.guild.id = 100
            ch._client = MagicMock()
            ch._client.user = MagicMock()
            ch._client.user.mentioned_in.return_value = True
            assert ch._check_access(msg) is True


# ---------------------------------------------------------------------------
# Discord config loading tests
# ---------------------------------------------------------------------------

class TestDiscordConfig:
    """Tests for Discord config loading in ChannelsConfig."""

    def _make_yaml(self, extra_account: dict, tmp_path) -> str:
        """Write a minimal config.yaml and return the path."""
        import yaml
        content = {
            "channels": {
                "discord": {
                    "enabled": True,
                    "accounts": [{"name": "bot", "token": "tok", **extra_account}],
                }
            }
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(content))
        return str(p)

    def test_proxy_url_passed_through(self, tmp_path):
        """proxy_url from YAML is included in ChannelConfig.extra."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({"proxy_url": "http://proxy:7897"}, tmp_path)
        cfg = load_channels_config(path)
        configs = cfg.get_discord_configs()
        assert len(configs) == 1
        config, _ = configs[0]
        assert config.extra["proxy_url"] == "http://proxy:7897"

    def test_proxy_url_from_env(self, tmp_path, monkeypatch):
        """DISCORD_PROXY env var is used as fallback."""
        monkeypatch.setenv("DISCORD_PROXY", "http://env-proxy:1234")
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({}, tmp_path)
        cfg = load_channels_config(path)
        configs = cfg.get_discord_configs()
        config, _ = configs[0]
        assert config.extra.get("proxy_url") == "http://env-proxy:1234"

    def test_require_mention_default_true(self, tmp_path):
        """require_mention defaults to True."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({}, tmp_path)
        cfg = load_channels_config(path)
        config, _ = cfg.get_discord_configs()[0]
        assert config.extra["require_mention"] is True

    def test_allow_dm_default_true(self, tmp_path):
        """allow_dm defaults to True."""
        from spoon_bot.channels.config import load_channels_config
        path = self._make_yaml({}, tmp_path)
        cfg = load_channels_config(path)
        config, _ = cfg.get_discord_configs()[0]
        assert config.extra["allow_dm"] is True
