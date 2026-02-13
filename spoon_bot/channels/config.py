"""Unified configuration loader for channels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from spoon_bot.channels.base import ChannelConfig, ChannelMode
from spoon_bot.defaults import (
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_RETRY_MAX_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_HEALTH_CHECK_INTERVAL,
    DEFAULT_MEDIA_MAX_MB,
)


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""

    def __init__(self, channel: str, field: str, message: str):
        self.channel = channel
        self.field = field
        super().__init__(f"[{channel}] {field}: {message}")


class ChannelsConfig:
    """Configuration container for all channels."""

    # Required fields for each channel type
    _REQUIRED_FIELDS = {
        "telegram": ["token"],
        "discord": ["token"],
        "feishu": ["app_id", "app_secret"],
    }

    def __init__(self, config_dict: dict[str, Any]):
        """
        Initialize from configuration dictionary.

        Args:
            config_dict: Raw configuration dictionary

        Raises:
            ConfigValidationError: If validation fails
        """
        self.raw = config_dict
        self.telegram = config_dict.get("telegram", {})
        self.discord = config_dict.get("discord", {})
        self.feishu = config_dict.get("feishu", {})
        self.cli = config_dict.get("cli", {})

        # Validate all enabled channels
        self._validate_all()

    def _validate_all(self) -> None:
        """Validate all channel configurations."""
        if self.telegram.get("enabled", False):
            self._validate_channel("telegram", self.telegram)
        if self.discord.get("enabled", False):
            self._validate_channel("discord", self.discord)
        if self.feishu.get("enabled", False):
            self._validate_channel("feishu", self.feishu)

    def _validate_channel(self, channel_type: str, config: dict[str, Any]) -> None:
        """
        Validate a channel configuration.

        Args:
            channel_type: Type of channel (telegram, discord, feishu)
            config: Channel configuration dict

        Raises:
            ConfigValidationError: If validation fails
        """
        # Validate accounts field
        accounts = config.get("accounts")
        if accounts is None:
            raise ConfigValidationError(
                channel_type, "accounts", "field is required when channel is enabled"
            )
        if not isinstance(accounts, list):
            raise ConfigValidationError(
                channel_type, "accounts", f"must be a list, got {type(accounts).__name__}"
            )
        if len(accounts) == 0:
            raise ConfigValidationError(
                channel_type, "accounts", "must contain at least one account"
            )

        # Validate each account
        required_fields = self._REQUIRED_FIELDS.get(channel_type, [])
        for i, account in enumerate(accounts):
            if not isinstance(account, dict):
                raise ConfigValidationError(
                    channel_type, f"accounts[{i}]", f"must be a dict, got {type(account).__name__}"
                )

            account_name = account.get("name", f"accounts[{i}]")

            # Check required fields
            for field in required_fields:
                value = account.get(field)
                if value is None:
                    raise ConfigValidationError(
                        channel_type, f"{account_name}.{field}", "is required"
                    )
                # Check if it's an unresolved env var
                resolved = self._resolve_env(value)
                if resolved is None and value.startswith("${"):
                    logger.warning(
                        f"[{channel_type}] {account_name}.{field}: "
                        f"environment variable {value} is not set"
                    )

    def _build_common_config(
        self,
        account: dict[str, Any],
        channel_name: str,
        mode: ChannelMode,
    ) -> dict[str, Any]:
        """
        Build common configuration fields shared across all channel types.

        This reduces duplication when constructing ChannelConfig objects.
        Default values are imported from spoon_bot.defaults for consistency.

        Args:
            account: Raw account configuration dict
            channel_name: Channel type name (e.g., "telegram", "discord")
            mode: Channel operating mode

        Returns:
            Dictionary with common config fields
        """
        return {
            "name": channel_name,
            "mode": mode,
            "enabled": True,
            "retry_max_attempts": account.get("retry_max_attempts", DEFAULT_RETRY_MAX_ATTEMPTS),
            "retry_delay": account.get("retry_delay", DEFAULT_RETRY_DELAY),
            "health_check_interval": account.get("health_check_interval", DEFAULT_HEALTH_CHECK_INTERVAL),
            "agent_config": account.get("agent_config", {}),
        }

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

            # Build config with common fields + Telegram-specific fields
            common = self._build_common_config(account, "telegram", mode)
            config = ChannelConfig(
                **common,
                webhook_path=account.get("webhook_url"),
                webhook_secret=account.get("webhook_secret"),
                # Telegram-specific
                token=self._resolve_env(account.get("token")),
                allowed_users=account.get("allowed_users", []),
                groups=account.get("groups", {}),
                media_max_mb=account.get("media_max_mb", DEFAULT_MEDIA_MAX_MB),
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

            # Build config with common fields + Discord-specific fields
            common = self._build_common_config(account, "discord", ChannelMode.GATEWAY)
            config = ChannelConfig(
                **common,
                # Discord-specific
                token=self._resolve_env(account.get("token")),
                intents=account.get("intents", []),
                allowed_guilds=account.get("allowed_guilds", []),
                allowed_users=account.get("allowed_users", []),
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

            # Build config with common fields + Feishu-specific fields
            common = self._build_common_config(account, "feishu", ChannelMode.WEBHOOK)
            config = ChannelConfig(
                **common,
                webhook_path=account.get("webhook_url"),
                # Feishu-specific
                app_id=self._resolve_env(account.get("app_id")),
                app_secret=self._resolve_env(account.get("app_secret")),
                verification_token=self._resolve_env(account.get("verification_token")),
                encrypt_key=self._resolve_env(account.get("encrypt_key")),
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

    from spoon_bot.defaults import DEFAULT_MAX_ITERATIONS

    default_config = {
        "agent": {
            "model": DEFAULT_MODEL,
            "provider": DEFAULT_PROVIDER,
            "workspace": "~/.spoon-bot/workspace",
            "max_iterations": DEFAULT_MAX_ITERATIONS,
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
