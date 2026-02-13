"""Unified configuration loader for channels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from spoon_bot.channels.base import ChannelConfig, ChannelMode


class ChannelsConfig:
    """Configuration container for all channels."""

    def __init__(self, config_dict: dict[str, Any]):
        """
        Initialize from configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary
        """
        self.raw = config_dict
        self.telegram = config_dict.get("telegram", {})
        self.discord = config_dict.get("discord", {})
        self.feishu = config_dict.get("feishu", {})
        self.cli = config_dict.get("cli", {})

    def get_telegram_configs(self) -> list[tuple[ChannelConfig, str]]:
        """Get Telegram channel configurations.

        Returns:
            List of tuples containing (ChannelConfig, account_name)
        """
        if not self.telegram.get("enabled", False):
            return []

        configs = []
        accounts = self.telegram.get("accounts", [])

        for account in accounts:
            name = account.get("name", "default")
            mode_str = account.get("mode", "polling")
            mode = ChannelMode.WEBHOOK if mode_str == "webhook" else ChannelMode.POLLING

            config = ChannelConfig(
                name="telegram",
                mode=mode,
                enabled=True,
                retry_max_attempts=account.get("retry_max_attempts", 3),
                retry_delay=account.get("retry_delay", 1.0),
                health_check_interval=account.get("health_check_interval", 60.0),
                webhook_path=account.get("webhook_url"),
                webhook_secret=account.get("webhook_secret"),
                # Telegram-specific config
                token=self._resolve_env(account.get("token")),
                allowed_users=account.get("allowed_users", []),
                groups=account.get("groups", {}),
                media_max_mb=account.get("media_max_mb", 20),
                agent_config=account.get("agent_config", {}),
            )
            configs.append((config, name))

        return configs

    def get_discord_configs(self) -> list[tuple[ChannelConfig, str]]:
        """Get Discord channel configurations."""
        if not self.discord.get("enabled", False):
            return []

        configs = []
        accounts = self.discord.get("accounts", [])

        for account in accounts:
            name = account.get("name", "default")

            config = ChannelConfig(
                name="discord",
                mode=ChannelMode.GATEWAY,
                enabled=True,
                retry_max_attempts=account.get("retry_max_attempts", 3),
                retry_delay=account.get("retry_delay", 1.0),
                health_check_interval=account.get("health_check_interval", 60.0),
                # Discord-specific config
                token=self._resolve_env(account.get("token")),
                intents=account.get("intents", []),
                allowed_guilds=account.get("allowed_guilds", []),
                allowed_users=account.get("allowed_users", []),
                agent_config=account.get("agent_config", {}),
            )
            configs.append((config, name))

        return configs

    def get_feishu_configs(self) -> list[tuple[ChannelConfig, str]]:
        """Get Feishu channel configurations."""
        if not self.feishu.get("enabled", False):
            return []

        configs = []
        accounts = self.feishu.get("accounts", [])

        for account in accounts:
            name = account.get("name", "default")

            config = ChannelConfig(
                name="feishu",
                mode=ChannelMode.WEBHOOK,
                enabled=True,
                retry_max_attempts=account.get("retry_max_attempts", 3),
                retry_delay=account.get("retry_delay", 1.0),
                health_check_interval=account.get("health_check_interval", 60.0),
                webhook_path=account.get("webhook_url"),
                # Feishu-specific config
                app_id=self._resolve_env(account.get("app_id")),
                app_secret=self._resolve_env(account.get("app_secret")),
                verification_token=self._resolve_env(account.get("verification_token")),
                encrypt_key=self._resolve_env(account.get("encrypt_key")),
                agent_config=account.get("agent_config", {}),
            )
            configs.append((config, name))

        return configs

    def is_cli_enabled(self) -> bool:
        """Check if CLI channel is enabled."""
        return self.cli.get("enabled", True)

    @staticmethod
    def _resolve_env(value: str | None) -> str | None:
        """
        Resolve environment variable references.

        Supports ${VAR_NAME} syntax.

        Args:
            value: String that may contain ${VAR_NAME}

        Returns:
            Resolved string or None
        """
        if not value:
            return None

        if value.startswith("${") and value.endswith("}"):
            var_name = value[2:-1]
            return os.getenv(var_name)

        return value


def load_channels_config(config_path: str | Path | None = None) -> ChannelsConfig:
    """
    Load channels configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for:
                     1. SPOON_BOT_CONFIG environment variable
                     2. ~/.spoon-bot/config.yaml
                     3. ./config.yaml

    Returns:
        ChannelsConfig object

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    # Determine config path
    if config_path is None:
        config_path = os.getenv("SPOON_BOT_CONFIG")

    if config_path is None:
        # Try default locations
        default_paths = [
            Path.home() / ".spoon-bot" / "config.yaml",
            Path("config.yaml"),
        ]

        for path in default_paths:
            if path.exists():
                config_path = path
                break

    if config_path is None:
        # No config found, return empty config
        logger.warning("No channels config file found, using defaults")
        return ChannelsConfig({})

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading channels config from: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f) or {}

    # Validate config is a dictionary
    if not isinstance(full_config, dict):
        raise ValueError(f"Invalid config file format: expected dict, got {type(full_config).__name__}")

    # Extract channels section
    channels_dict = full_config.get("channels", {})

    return ChannelsConfig(channels_dict)


def create_default_config(output_path: str | Path) -> None:
    """
    Create a default configuration file.

    Args:
        output_path: Path to write config file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = {
        "agent": {
            "model": "anthropic/claude-3.5-sonnet",
            "provider": "openrouter",
            "workspace": "~/.spoon-bot/workspace",
            "max_iterations": 20,
            "tool_profile": "core",
        },
        "sessions": {
            "default_timeout": 3600,
            "max_sessions": 100,
            "persistence": True,
        },
        "channels": {
            "telegram": {
                "enabled": False,
                "accounts": [
                    {
                        "name": "main_bot",
                        "token": "${TELEGRAM_BOT_TOKEN}",
                        "mode": "polling",
                        "allowed_users": [],
                        "groups": {
                            "enabled": True,
                            "require_mention": True,
                            "allowed_chats": [],
                        },
                        "agent_config": {
                            "tool_profile": "full",
                        },
                    }
                ],
            },
            "discord": {
                "enabled": False,
                "accounts": [
                    {
                        "name": "dev_bot",
                        "token": "${DISCORD_BOT_TOKEN}",
                        "intents": [
                            "guilds",
                            "guild_messages",
                            "message_content",
                            "dm_messages",
                        ],
                        "allowed_guilds": [],
                        "allowed_users": [],
                        "agent_config": {
                            "tool_profile": "coding",
                        },
                    }
                ],
            },
            "feishu": {
                "enabled": False,
                "accounts": [
                    {
                        "name": "enterprise_bot",
                        "app_id": "${FEISHU_APP_ID}",
                        "app_secret": "${FEISHU_APP_SECRET}",
                        "verification_token": "${FEISHU_VERIFICATION_TOKEN}",
                        "encrypt_key": "${FEISHU_ENCRYPT_KEY}",
                        "webhook_url": "https://your-domain.com/feishu/webhook",
                    }
                ],
            },
            "cli": {
                "enabled": True,
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"Created default config at: {output_path}")
