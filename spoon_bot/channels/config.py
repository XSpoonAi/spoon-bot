"""Unified configuration loader for channels."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from spoon_bot.channels.base import ChannelConfig, ChannelMode


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

    # Well-known environment variable names for channel credentials.
    # Used as fallback when a credential is not explicitly set in YAML.
    _ENV_FALLBACKS: dict[str, dict[str, str]] = {
        "telegram": {"token": "TELEGRAM_BOT_TOKEN"},
        "discord": {"token": "DISCORD_BOT_TOKEN"},
        "feishu": {
            "app_id": "FEISHU_APP_ID",
            "app_secret": "FEISHU_APP_SECRET",
            "verification_token": "FEISHU_VERIFICATION_TOKEN",
            "encrypt_key": "FEISHU_ENCRYPT_KEY",
        },
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

        # Auto-discover channels from env vars when not in YAML
        self._auto_discover_channels()

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

            # Check required fields (with env-var fallback)
            for field in required_fields:
                value = account.get(field)
                resolved = self._resolve_with_fallback(value, channel_type, field)
                if resolved is None:
                    # Build a helpful message
                    fallback_var = self._ENV_FALLBACKS.get(channel_type, {}).get(field)
                    hint = f" or set {fallback_var}" if fallback_var else ""
                    raise ConfigValidationError(
                        channel_type,
                        f"{account_name}.{field}",
                        f"is required (set in config.yaml{hint})",
                    )
                elif value and value.startswith("${") and self._resolve_env(value) is None:
                    # Explicit ${VAR} in YAML but that var is unset —
                    # fallback resolved it, log for transparency.
                    logger.debug(
                        f"[{channel_type}] {account_name}.{field}: "
                        f"{value} not set, using env fallback"
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
            "retry_max_attempts": account.get("retry_max_attempts", 3),
            "retry_delay": account.get("retry_delay", 1.0),
            "health_check_interval": account.get("health_check_interval", 60.0),
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

            # allowed_users: YAML list takes priority; fall back to TELEGRAM_USER_ID env var
            allowed_users = account.get("allowed_users", [])
            if not allowed_users:
                env_uid = os.getenv("TELEGRAM_USER_ID")
                if env_uid:
                    try:
                        allowed_users = [int(env_uid)]
                    except ValueError:
                        logger.warning(f"TELEGRAM_USER_ID={env_uid!r} is not a valid integer, ignoring")

            # proxy_url: YAML > TELEGRAM_PROXY > HTTPS_PROXY
            proxy_url = (
                account.get("proxy_url")
                or os.getenv("TELEGRAM_PROXY")
                or os.getenv("HTTPS_PROXY")
            )

            config = ChannelConfig(
                **common,
                webhook_path=account.get("webhook_url"),
                webhook_secret=account.get("webhook_secret"),
                # Telegram-specific (falls back to TELEGRAM_BOT_TOKEN env var)
                token=self._resolve_with_fallback(account.get("token"), "telegram", "token"),
                allowed_users=allowed_users,
                groups=account.get("groups", {}),
                media_max_mb=account.get("media_max_mb", 20),
                proxy_url=proxy_url,
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

            # allowed_guilds / allowed_users: YAML takes priority; fall back to env vars
            allowed_guilds = account.get("allowed_guilds", [])
            if not allowed_guilds:
                env_gid = os.getenv("DISCORD_GUILD_ID")
                if env_gid:
                    try:
                        allowed_guilds = [int(env_gid)]
                    except ValueError:
                        logger.warning(f"DISCORD_GUILD_ID={env_gid!r} is not a valid integer, ignoring")

            allowed_users = account.get("allowed_users", [])
            if not allowed_users:
                env_uid = os.getenv("DISCORD_USER_ID")
                if env_uid:
                    try:
                        allowed_users = [int(env_uid)]
                    except ValueError:
                        logger.warning(f"DISCORD_USER_ID={env_uid!r} is not a valid integer, ignoring")

            config = ChannelConfig(
                **common,
                # Discord-specific (falls back to DISCORD_BOT_TOKEN env var)
                token=self._resolve_with_fallback(account.get("token"), "discord", "token"),
                intents=account.get("intents", []),
                allowed_guilds=allowed_guilds,
                allowed_users=allowed_users,
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

            # mode: "ws" (WebSocket, default) -> GATEWAY, "webhook" -> WEBHOOK
            mode_str = account.get("mode", "ws")
            mode = ChannelMode.WEBHOOK if mode_str == "webhook" else ChannelMode.GATEWAY

            # Build config with common fields + Feishu-specific fields
            common = self._build_common_config(account, "feishu", mode)

            # allowed_chats / allowed_users: YAML takes priority; fall back to env vars
            allowed_chats = account.get("allowed_chats", [])
            allowed_users = account.get("allowed_users", [])
            if not allowed_users:
                env_uid = os.getenv("FEISHU_USER_ID")
                if env_uid:
                    allowed_users = [env_uid]

            config = ChannelConfig(
                **common,
                webhook_path=account.get("webhook_url"),
                # Feishu-specific (falls back to FEISHU_* env vars)
                app_id=self._resolve_with_fallback(account.get("app_id"), "feishu", "app_id"),
                app_secret=self._resolve_with_fallback(account.get("app_secret"), "feishu", "app_secret"),
                verification_token=self._resolve_with_fallback(account.get("verification_token"), "feishu", "verification_token"),
                encrypt_key=self._resolve_with_fallback(account.get("encrypt_key"), "feishu", "encrypt_key"),
                domain=account.get("domain", "feishu"),
                allowed_chats=allowed_chats,
                allowed_users=allowed_users,
                require_mention=account.get("require_mention", True),
                render_mode=account.get("render_mode", "auto"),
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

    @classmethod
    def _resolve_with_fallback(
        cls, value: str | None, channel_type: str, field: str
    ) -> str | None:
        """
        Resolve a credential value with env-var fallback.

        Resolution order:
          1. Explicit string value from YAML (returned as-is)
          2. ``${VAR}`` syntax resolved via :meth:`_resolve_env`
          3. Well-known env var from :attr:`_ENV_FALLBACKS`

        Args:
            value: Raw value from YAML (may be None, a literal, or ``${VAR}``).
            channel_type: Channel kind (``telegram``, ``discord``, ``feishu``).
            field: Field name (``token``, ``app_id``, …).

        Returns:
            Resolved credential string, or None if unavailable.
        """
        resolved = cls._resolve_env(value)
        if resolved is not None:
            return resolved
        # Fall back to well-known env var
        fallback_var = cls._ENV_FALLBACKS.get(channel_type, {}).get(field)
        if fallback_var:
            env_val = os.getenv(fallback_var)
            if env_val:
                logger.debug(
                    f"[{channel_type}] {field}: using env var {fallback_var}"
                )
            return env_val
        return None

    def _auto_discover_channels(self) -> None:
        """Auto-enable channels when env vars are set but YAML has no config.

        Only creates a minimal default account entry so that
        :meth:`_resolve_with_fallback` can pick up the credential later.
        Channels explicitly set to ``enabled: false`` are NOT overridden.
        """
        # Telegram
        if (
            not self.telegram.get("enabled")
            and self.telegram.get("enabled") is not False
            and os.getenv("TELEGRAM_BOT_TOKEN")
        ):
            logger.info(
                "Auto-discovered Telegram channel from TELEGRAM_BOT_TOKEN env var"
            )
            account: dict[str, Any] = {"name": "default", "mode": "polling"}
            # Restrict access if TELEGRAM_USER_ID is set
            user_id = os.getenv("TELEGRAM_USER_ID")
            if user_id:
                try:
                    account["allowed_users"] = [int(user_id)]
                except ValueError:
                    logger.warning(
                        f"TELEGRAM_USER_ID={user_id!r} is not a valid integer, ignoring"
                    )
            self.telegram = {"enabled": True, "accounts": [account]}

        # Discord
        if (
            not self.discord.get("enabled")
            and self.discord.get("enabled") is not False
            and os.getenv("DISCORD_BOT_TOKEN")
        ):
            logger.info(
                "Auto-discovered Discord channel from DISCORD_BOT_TOKEN env var"
            )
            account = {"name": "default"}
            # Restrict access if DISCORD_GUILD_ID / DISCORD_USER_ID is set
            guild_id = os.getenv("DISCORD_GUILD_ID")
            if guild_id:
                try:
                    account["allowed_guilds"] = [int(guild_id)]
                except ValueError:
                    logger.warning(
                        f"DISCORD_GUILD_ID={guild_id!r} is not a valid integer, ignoring"
                    )
            discord_user_id = os.getenv("DISCORD_USER_ID")
            if discord_user_id:
                try:
                    account["allowed_users"] = [int(discord_user_id)]
                except ValueError:
                    logger.warning(
                        f"DISCORD_USER_ID={discord_user_id!r} is not a valid integer, ignoring"
                    )
            self.discord = {"enabled": True, "accounts": [account]}

        # Feishu — requires both app_id and app_secret
        if (
            not self.feishu.get("enabled")
            and self.feishu.get("enabled") is not False
            and os.getenv("FEISHU_APP_ID")
            and os.getenv("FEISHU_APP_SECRET")
        ):
            logger.info(
                "Auto-discovered Feishu channel from FEISHU_APP_ID/FEISHU_APP_SECRET env vars"
            )
            self.feishu = {
                "enabled": True,
                "accounts": [{"name": "default", "mode": "ws"}],
            }


def _find_and_load_yaml(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Locate and parse the YAML config file.

    Resolution order:
      1. Explicit ``config_path`` parameter
      2. ``SPOON_BOT_CONFIG`` environment variable
      3. ``~/.spoon-bot/config.yaml``
      4. ``./config.yaml``

    Args:
        config_path: Explicit path (optional).

    Returns:
        Parsed YAML dictionary, or empty dict when no file is found.

    Raises:
        FileNotFoundError: If an explicit path is given but does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    if config_path is None:
        config_path = os.getenv("SPOON_BOT_CONFIG")

    if config_path is None:
        for candidate in [
            Path.home() / ".spoon-bot" / "config.yaml",
            Path("config.yaml"),
        ]:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None:
        return {}

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid config file format: expected dict, got {type(data).__name__}"
        )

    return data


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
        FileNotFoundError: If an explicit config file is not found.
        yaml.YAMLError: If YAML parsing fails.
    """
    full_config = _find_and_load_yaml(config_path)

    if not full_config:
        logger.warning("No channels config file found, using defaults")
        return ChannelsConfig({})

    logger.info("Loading channels config from YAML")
    return ChannelsConfig(full_config.get("channels", {}))


def load_agent_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load agent configuration from YAML and environment variables.

    This is the **single entry point** for all agent configuration resolution.
    Callers (cli.py, server.py, core.py) should use this function instead of
    reading environment variables themselves.

    Resolution priority:  **YAML > env vars**  (no built-in defaults for
    user-facing settings like model/provider).  Callers may overlay CLI args
    on top of the returned dict.

    Supported fields::

        model, provider, api_key, base_url, workspace,
        max_iterations, tool_profile, enabled_tools, enable_skills,
        shell_timeout, max_output, context_window,
        session_store_backend, session_store_dsn, session_store_db_path,
        mcp_config (alias: mcp_servers)

    Args:
        config_path: Explicit path to YAML file (optional).

    Returns:
        Dictionary with resolved agent config values.
        May be empty if nothing is configured.
    """
    # ------------------------------------------------------------------
    # 1. Read YAML
    # ------------------------------------------------------------------
    try:
        full_config = _find_and_load_yaml(config_path)
    except FileNotFoundError:
        raise
    except Exception as exc:
        logger.warning(f"Could not load agent config from YAML: {exc}")
        full_config = {}

    agent_raw: dict[str, Any] = full_config.get("agent", {}) if full_config else {}

    def _resolve_env_deep(value: Any) -> Any:
        """Resolve ${VAR} in nested dict/list values."""
        if isinstance(value, str):
            return ChannelsConfig._resolve_env(value)
        if isinstance(value, list):
            return [_resolve_env_deep(v) for v in value]
        if isinstance(value, dict):
            return {k: _resolve_env_deep(v) for k, v in value.items()}
        return value

    resolved: dict[str, Any] = _resolve_env_deep(agent_raw) if isinstance(agent_raw, dict) else {}

    # Backward-compatible alias: mcp_servers -> mcp_config
    if "mcp_config" not in resolved and isinstance(resolved.get("mcp_servers"), dict):
        resolved["mcp_config"] = resolved.pop("mcp_servers")

    # Merge channel account-level agent overrides into effective global config.
    # Current runtime uses one shared agent for all channels, so we merge these
    # settings rather than applying them per-account.
    channels_cfg = full_config.get("channels", {}) if isinstance(full_config, dict) else {}
    if isinstance(channels_cfg, dict):
        profile_rank = {"core": 0, "research": 1, "web3": 1, "coding": 2, "full": 3}
        selected_profile: str | None = None
        selected_profile_source: str | None = None
        merged_enabled_tools: set[str] = set()
        merged_mcp_config: dict[str, Any] = {}
        scalar_fields = {
            "max_iterations",
            "enable_skills",
            "shell_timeout",
            "max_output",
            "context_window",
            "session_store_backend",
            "session_store_dsn",
            "session_store_db_path",
        }

        for channel_name, channel_cfg in channels_cfg.items():
            if not isinstance(channel_cfg, dict):
                continue
            if not channel_cfg.get("enabled", False):
                continue
            accounts = channel_cfg.get("accounts", [])
            if not isinstance(accounts, list):
                continue

            for account in accounts:
                if not isinstance(account, dict):
                    continue
                account_name = account.get("name", "default")
                agent_override_raw = account.get("agent_config")
                if not isinstance(agent_override_raw, dict) or not agent_override_raw:
                    continue

                agent_override = _resolve_env_deep(agent_override_raw)
                source = f"{channel_name}:{account_name}"

                profile = agent_override.get("tool_profile")
                if isinstance(profile, str):
                    rank = profile_rank.get(profile, -1)
                    current_rank = profile_rank.get(selected_profile, -1) if selected_profile else -1
                    if rank >= current_rank:
                        selected_profile = profile
                        selected_profile_source = source

                enabled_tools = agent_override.get("enabled_tools")
                if isinstance(enabled_tools, list):
                    for name in enabled_tools:
                        if isinstance(name, str) and name.strip():
                            merged_enabled_tools.add(name.strip())

                override_mcp = agent_override.get("mcp_config")
                if override_mcp is None:
                    override_mcp = agent_override.get("mcp_servers")
                if isinstance(override_mcp, dict):
                    for server_name, server_cfg in override_mcp.items():
                        if isinstance(server_cfg, dict):
                            merged_mcp_config[server_name] = server_cfg

                for field in scalar_fields:
                    if field not in agent_override:
                        continue
                    value = agent_override.get(field)
                    if field in resolved and resolved[field] != value:
                        logger.info(
                            f"Agent config field '{field}' overridden by channel account "
                            f"'{source}'"
                        )
                    resolved[field] = value

        if selected_profile:
            if "tool_profile" in resolved and resolved["tool_profile"] != selected_profile:
                logger.info(
                    f"Agent tool_profile overridden by channel account "
                    f"'{selected_profile_source}' -> '{selected_profile}'"
                )
            resolved["tool_profile"] = selected_profile

        if merged_enabled_tools:
            existing = resolved.get("enabled_tools", [])
            if isinstance(existing, list):
                for name in existing:
                    if isinstance(name, str) and name.strip():
                        merged_enabled_tools.add(name.strip())
            resolved["enabled_tools"] = sorted(merged_enabled_tools)

        if merged_mcp_config:
            base_mcp = {}
            if isinstance(resolved.get("mcp_config"), dict):
                base_mcp = dict(resolved["mcp_config"])
            base_mcp.update(merged_mcp_config)
            resolved["mcp_config"] = base_mcp

    # ------------------------------------------------------------------
    # 2. Overlay env vars for fields NOT already set by YAML
    # ------------------------------------------------------------------
    agent_env_map: dict[str, list[str]] = {
        "provider":       ["SPOON_BOT_DEFAULT_PROVIDER", "SPOON_PROVIDER"],
        "model":          ["SPOON_BOT_DEFAULT_MODEL", "SPOON_MODEL"],
        "workspace":      ["SPOON_BOT_WORKSPACE_PATH"],
        "max_iterations": ["SPOON_BOT_MAX_ITERATIONS", "SPOON_MAX_STEPS"],
        "enable_skills":  ["SPOON_BOT_ENABLE_SKILLS"],
        "shell_timeout":  ["SPOON_BOT_SHELL_TIMEOUT"],
        "max_output":     ["SPOON_BOT_MAX_OUTPUT"],
        "context_window": ["CONTEXT_WINDOW"],
    }
    for field, env_vars in agent_env_map.items():
        if not resolved.get(field):
            for var in env_vars:
                val = os.environ.get(var)
                if val:
                    if field in {"max_iterations", "shell_timeout", "max_output", "context_window"}:
                        resolved[field] = int(val)
                    elif field == "enable_skills":
                        resolved[field] = val.lower() == "true"
                    else:
                        resolved[field] = val
                    logger.debug(f"Agent config: {field} from env var {var}")
                    break

    # ------------------------------------------------------------------
    # 3. Resolve base_url from provider-specific env vars
    # ------------------------------------------------------------------
    if not resolved.get("base_url"):
        provider = resolved.get("provider", "")
        if provider == "openai":
            resolved["base_url"] = os.environ.get("OPENAI_BASE_URL") or os.environ.get("BASE_URL")
        elif provider == "anthropic":
            resolved["base_url"] = os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("BASE_URL")
        else:
            resolved["base_url"] = os.environ.get("BASE_URL")

    # ------------------------------------------------------------------
    # 4. Resolve api_key from provider-specific env vars
    # ------------------------------------------------------------------
    if not resolved.get("api_key"):
        provider = resolved.get("provider", "")
        api_key_map: dict[str, str] = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        env_var = api_key_map.get(provider)
        if env_var:
            resolved["api_key"] = os.environ.get(env_var)
        # Generic fallback: try common API key env vars
        if not resolved.get("api_key"):
            resolved["api_key"] = (
                os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )

    # ------------------------------------------------------------------
    # 5. Expand workspace tilde
    # ------------------------------------------------------------------
    if resolved.get("workspace"):
        resolved["workspace"] = str(Path(str(resolved["workspace"])).expanduser())

    # Clean up None values
    resolved = {k: v for k, v in resolved.items() if v is not None}

    if resolved:
        logger.debug(f"Resolved agent config keys: {list(resolved.keys())}")
    return resolved


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
            "model": "your-model-here",
            "provider": "anthropic",
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
