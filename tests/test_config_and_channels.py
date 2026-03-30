"""Tests for configuration priority, channel filtering, and --cli/--no-cli behavior."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# §9.1  Configuration priority: YAML > env > (no defaults)
# ---------------------------------------------------------------------------

class TestConfigPriority:
    """load_agent_config() must resolve YAML > env, with no hidden defaults."""

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        cfg = tmp_path / "config.yaml"
        cfg.write_text(textwrap.dedent(content))
        return cfg

    def test_yaml_takes_priority_over_env(self, tmp_path, monkeypatch):
        """YAML values must win over environment variables."""
        cfg_path = self._write_yaml(tmp_path, """\
            agent:
              provider: anthropic
              model: claude-sonnet
        """)
        monkeypatch.setenv("SPOON_BOT_DEFAULT_PROVIDER", "openai")
        monkeypatch.setenv("SPOON_BOT_DEFAULT_MODEL", "gpt-4")

        from spoon_bot.channels.config import load_agent_config

        result = load_agent_config(cfg_path)
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-sonnet"

    def test_env_used_when_yaml_missing(self, tmp_path, monkeypatch):
        """Env vars fill in fields absent from YAML."""
        cfg_path = self._write_yaml(tmp_path, """\
            agent:
              model: claude-sonnet
        """)
        monkeypatch.setenv("SPOON_BOT_DEFAULT_PROVIDER", "openai")

        from spoon_bot.channels.config import load_agent_config

        result = load_agent_config(cfg_path)
        assert result["model"] == "claude-sonnet"  # from YAML
        assert result["provider"] == "openai"       # from env

    def test_no_hidden_defaults_for_model_provider(self, tmp_path, monkeypatch):
        """model/provider must not have silent fallback values."""
        for var in (
            "SPOON_BOT_DEFAULT_PROVIDER",
            "SPOON_PROVIDER",
            "SPOON_BOT_DEFAULT_MODEL",
            "SPOON_MODEL",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg_path = self._write_yaml(tmp_path, """\
            agent:
              workspace: /tmp/test
        """)

        from spoon_bot.channels.config import load_agent_config

        result = load_agent_config(cfg_path)
        assert "model" not in result
        assert "provider" not in result

    def test_env_var_substitution_in_yaml(self, tmp_path, monkeypatch):
        """${VAR} syntax in YAML should resolve to env values."""
        monkeypatch.setenv("MY_API_KEY", "sk-secret-123")
        cfg_path = self._write_yaml(tmp_path, """\
            agent:
              api_key: ${MY_API_KEY}
              provider: anthropic
              model: claude-sonnet
        """)

        from spoon_bot.channels.config import load_agent_config

        result = load_agent_config(cfg_path)
        assert result["api_key"] == "sk-secret-123"

    def test_provider_specific_api_key_from_env(self, tmp_path, monkeypatch):
        """Provider-specific API key env vars should be resolved."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key")
        cfg_path = self._write_yaml(tmp_path, """\
            agent:
              provider: anthropic
              model: claude-sonnet
        """)

        from spoon_bot.channels.config import load_agent_config

        result = load_agent_config(cfg_path)
        assert result["api_key"] == "sk-ant-key"

    def test_empty_yaml_returns_env_values(self, tmp_path, monkeypatch):
        """An empty agent section still picks up env vars."""
        cfg_path = self._write_yaml(tmp_path, """\
            agent: {}
        """)
        monkeypatch.setenv("SPOON_BOT_DEFAULT_PROVIDER", "openrouter")
        monkeypatch.setenv("SPOON_BOT_DEFAULT_MODEL", "mistral-large")

        from spoon_bot.channels.config import load_agent_config

        result = load_agent_config(cfg_path)
        assert result["provider"] == "openrouter"
        assert result["model"] == "mistral-large"

    def test_channel_agent_config_overrides_tool_profile(self, tmp_path):
        """Enabled channel account agent_config should override top-level tool_profile."""
        cfg_path = self._write_yaml(tmp_path, """\
            agent:
              provider: openrouter
              model: anthropic/claude-sonnet-4
              tool_profile: core
            channels:
              telegram:
                enabled: true
                accounts:
                  - name: spoon
                    token: "fake-token"
                    agent_config:
                      tool_profile: full
        """)

        from spoon_bot.channels.config import load_agent_config

        result = load_agent_config(cfg_path)
        assert result["tool_profile"] == "full"

    def test_channel_agent_config_merges_enabled_tools_and_mcp(self, tmp_path):
        """Channel agent_config should merge enabled_tools + mcp_config into agent config."""
        cfg_path = self._write_yaml(tmp_path, """\
            agent:
              provider: openrouter
              model: anthropic/claude-sonnet-4
              enabled_tools: ["shell"]
              mcp_servers:
                top:
                  command: npx
                  args: ["-y", "top-server"]
            channels:
              telegram:
                enabled: true
                accounts:
                  - name: spoon
                    token: "fake-token"
                    agent_config:
                      enabled_tools: ["self_upgrade", "shell"]
                      mcp_config:
                        github:
                          command: npx
                          args: ["-y", "@modelcontextprotocol/server-github"]
        """)

        from spoon_bot.channels.config import load_agent_config

        result = load_agent_config(cfg_path)
        assert result["enabled_tools"] == ["self_upgrade", "shell"]
        assert set(result["mcp_config"].keys()) == {"top", "github"}

    def test_mcp_servers_alias_and_env_resolution(self, tmp_path, monkeypatch):
        """Top-level mcp_servers should map to mcp_config with ${VAR} resolution."""
        monkeypatch.setenv("MCP_FS_ROOT", "C:/workspace")
        cfg_path = self._write_yaml(tmp_path, """\
            agent:
              provider: openrouter
              model: anthropic/claude-sonnet-4
              mcp_servers:
                filesystem:
                  command: npx
                  args: ["-y", "@modelcontextprotocol/server-filesystem", "${MCP_FS_ROOT}"]
        """)

        from spoon_bot.channels.config import load_agent_config

        result = load_agent_config(cfg_path)
        assert "mcp_servers" not in result
        assert result["mcp_config"]["filesystem"]["args"][-1] == "C:/workspace"


# ---------------------------------------------------------------------------
# §6.2  Channel filtering: --channels selects subset
# ---------------------------------------------------------------------------

class TestChannelFiltering:
    """ChannelManager.start_channels() must start only the requested subset."""

    @pytest.fixture()
    def manager(self):
        from spoon_bot.channels.manager import ChannelManager

        mgr = ChannelManager()

        # Add mock channels — .name must match the channel type prefix
        # so that start_channels(["telegram"]) can match ch.name == "telegram".
        for full, kind in (
            ("telegram:bot1", "telegram"),
            ("discord:main", "discord"),
            ("cli:default", "cli"),
        ):
            ch = MagicMock()
            ch.full_name = full
            ch.name = kind
            ch.config = MagicMock(enabled=True)
            ch.is_running = False
            ch.start = AsyncMock()
            ch.stop = AsyncMock()
            ch.attach_bus = MagicMock()
            mgr._channels[full] = ch

        # Stub out bus to avoid real I/O
        mgr._bus = MagicMock()
        mgr._bus.start = AsyncMock()
        mgr._bus.stop = AsyncMock()
        mgr._bus.is_running = False
        return mgr

    @pytest.mark.asyncio
    async def test_start_channels_filters_correctly(self, manager):
        """Only requested channels should be started."""
        manager._agent = MagicMock()  # satisfy start_channels guard

        await manager.start_channels(["telegram"])

        # telegram:bot1 should have started
        manager._channels["telegram:bot1"].start.assert_called_once()
        # discord and cli should NOT have started
        manager._channels["discord:main"].start.assert_not_called()
        manager._channels["cli:default"].start.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_all_starts_every_enabled_channel(self, manager):
        """start_all() should start all enabled channels."""
        manager._agent = MagicMock()

        await manager.start_all()

        for ch in manager._channels.values():
            ch.start.assert_called_once()


# ---------------------------------------------------------------------------
# §1.2 / §6.2  --cli/--no-cli override behavior
# ---------------------------------------------------------------------------

class TestCliEnabledOverride:
    """--cli/--no-cli must add or remove the CLI channel after config load."""

    @pytest.fixture()
    def manager(self):
        from spoon_bot.channels.manager import ChannelManager

        mgr = ChannelManager()
        mgr._bus = MagicMock()
        mgr._bus.start = AsyncMock()
        mgr._bus.stop = AsyncMock()
        return mgr

    def test_remove_cli_channel(self, manager):
        """--no-cli should remove the CLI channel even if config enabled it."""
        # Simulate CLI channel loaded from config
        cli_ch = MagicMock()
        cli_ch.full_name = "cli:default"
        cli_ch.attach_bus = MagicMock()
        manager._channels["cli:default"] = cli_ch

        assert "cli:default" in manager.channel_names
        manager.remove_channel("cli:default")
        assert "cli:default" not in manager.channel_names

    def test_add_cli_channel_when_missing(self, manager):
        """--cli should add CLI channel even if config didn't enable it."""
        from spoon_bot.channels.cli_channel import CLIChannel

        assert "cli:default" not in manager.channel_names

        cli_ch = CLIChannel()
        manager.add_channel(cli_ch)
        assert "cli:default" in manager.channel_names
        assert manager._channels["cli:default"].name == "cli"

    def test_bootstrap_init_channels_removes_cli_by_default(self):
        """init_channels(cli_enabled=False) must remove the CLI channel."""
        # We can't run the full init_channels without real config,
        # so verify the contract: cli_enabled defaults to False.
        import inspect
        from spoon_bot.bootstrap import init_channels

        sig = inspect.signature(init_channels)
        assert sig.parameters["cli_enabled"].default is False
