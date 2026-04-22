"""Enhanced channel manager for coordinating multiple channels."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.bus.events import InboundMessage, OutboundMessage
from spoon_bot.bus.queue import MessageBus
from spoon_bot.channels.base import BaseChannel, ChannelStatus
from spoon_bot.channels.config import (
    ChannelsConfig,
    build_group_safe_agent_override,
    load_channels_config,
    merge_agent_config,
    uses_risky_local_tools,
)

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop


class _CircuitBreaker:
    """Simple circuit breaker for LLM calls.

    States:
    - CLOSED (normal): requests pass through.
    - OPEN (tripped): requests are rejected immediately.
    - HALF_OPEN: one probe request is allowed to test recovery.

    Transitions:
    - CLOSED → OPEN when ``failure_threshold`` consecutive failures occur.
    - OPEN → HALF_OPEN after ``recovery_timeout`` seconds.
    - HALF_OPEN → CLOSED on success, → OPEN on failure.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = self.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            import time
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = self.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._state = self.CLOSED

    def record_failure(self) -> None:
        import time
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()
        if self._consecutive_failures >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                f"Circuit breaker OPEN after {self._consecutive_failures} "
                f"consecutive failures (recovery in {self.recovery_timeout}s)"
            )

    def allow_request(self) -> bool:
        return self.state != self.OPEN


class ChannelManager:
    """
    Enhanced manager for multiple communication channels.

    Features:
    - Load channels from configuration
    - Hot reload configuration
    - Health monitoring
    - Selective start/stop
    - Status reporting
    """

    def __init__(self, config: ChannelsConfig | None = None, bus: MessageBus | None = None):
        """
        Initialize channel manager.

        Args:
            config: Channels configuration (loaded from file if None)
            bus: Message bus (creates new if None)
        """
        self._config = config
        self._bus = bus or MessageBus()
        self._channels: dict[str, BaseChannel] = {}
        self._agent: AgentLoop | None = None
        self._default_agent: AgentLoop | None = None
        self._default_agent_config: dict[str, Any] = {}
        self._config_path: Path | None = None
        self._channel_agents: dict[str, AgentLoop] = {}
        self._group_agents: dict[str, AgentLoop] = {}
        self._scoped_agents: dict[str, AgentLoop] = {}
        self._agent_init_lock = asyncio.Lock()
        self._running = False
        self._health_check_task: asyncio.Task | None = None
        self._circuit_breaker = _CircuitBreaker()

    def set_agent(
        self,
        agent: AgentLoop,
        *,
        agent_config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """
        Set the agent for handling messages.

        Args:
            agent: AgentLoop instance.
        """
        self._agent = agent
        self._default_agent = agent
        self._default_agent_config = copy.deepcopy(agent_config or {})
        self._default_agent_config.setdefault("provider", getattr(agent, "provider", None))
        self._default_agent_config.setdefault("model", getattr(agent, "model", None))
        workspace = getattr(agent, "workspace", None)
        if workspace is not None:
            self._default_agent_config.setdefault("workspace", str(workspace))
        base_url = getattr(agent, "base_url", None)
        if base_url is not None:
            self._default_agent_config.setdefault("base_url", base_url)
        self._default_agent_config = {
            key: value
            for key, value in self._default_agent_config.items()
            if value is not None
        }
        self._config_path = (
            Path(config_path).expanduser()
            if config_path is not None
            else self._config_path
        )
        self._bus.set_handler(self._handle_message)

        # Align bus concurrency with agent pool size so neither is wasted.
        pool_size = getattr(agent, "_pool_size", None)
        if pool_size is not None and pool_size != self._bus.max_concurrency:
            self._bus.set_max_concurrency(pool_size)
            logger.info(
                f"Bus max_concurrency aligned to agent pool size: {pool_size}"
            )

        # Propagate agent reference to all existing channels
        for channel in self._channels.values():
            channel_agent = self._channel_agents.get(channel.full_name, agent)
            channel.set_agent(channel_agent)
        logger.info("Agent attached to ChannelManager")

    def add_channel(self, channel: BaseChannel) -> None:
        """
        Add a channel to the manager.

        Args:
            channel: Channel to add.
        """
        channel.attach_bus(self._bus)
        if self._agent:
            channel_agent = self._channel_agents.get(channel.full_name, self._agent)
            channel.set_agent(channel_agent)
        self._channels[channel.full_name] = channel
        logger.info(f"Added channel: {channel.full_name}")

    def remove_channel(self, name: str) -> bool:
        """
        Remove a channel.

        Args:
            name: Channel name to remove (full_name format: "type:account")

        Returns:
            True if channel was removed.
        """
        if name in self._channels:
            del self._channels[name]
            self._group_agents.pop(name, None)
            self._channel_agents.pop(name, None)
            scoped_keys = [key for key in self._scoped_agents if key.startswith(f"{name}:")]
            for key in scoped_keys:
                self._scoped_agents.pop(key, None)
            logger.info(f"Removed channel: {name}")
            return True
        return False

    @staticmethod
    def _reasoning_effort_from_think_level(level: object) -> str | None:
        normalized = str(level or "").strip().lower()
        aliases = {
            "basic": "low",
            "extended": "high",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in {"low", "medium", "high", "xhigh"}:
            return normalized
        return None

    @staticmethod
    def _sanitize_workspace_segment(name: str) -> str:
        """Return a filesystem-safe workspace segment for a channel/account."""
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
        return safe.strip("_") or "channel"

    def _default_workspace_root(self) -> Path:
        """Resolve the base workspace root inherited by managed channel agents."""
        if self._default_agent is not None:
            workspace = getattr(self._default_agent, "workspace", None)
            if workspace:
                return Path(str(workspace)).expanduser()
        workspace = self._default_agent_config.get("workspace")
        if workspace:
            return Path(str(workspace)).expanduser()
        return Path.home() / ".spoon-bot" / "workspace"

    def _derive_channel_workspace(
        self,
        channel_name: str,
        channel_override: dict[str, Any] | None,
    ) -> str:
        """Pick the workspace for a dedicated channel agent."""
        workspace = (
            channel_override.get("workspace")
            if isinstance(channel_override, dict)
            else None
        )
        if workspace:
            return str(Path(str(workspace)).expanduser())
        base_workspace = self._default_workspace_root()
        channel_workspace = (
            base_workspace
            / "channels"
            / self._sanitize_workspace_segment(channel_name)
        )
        return str(channel_workspace)

    def _channel_agent_config(self, channel: BaseChannel) -> dict[str, Any]:
        """Merge top-level agent config with a channel/account-specific override."""
        return merge_agent_config(
            self._default_agent_config,
            channel.config.extra.get("agent_config"),
        )

    def _group_agent_config(
        self,
        channel: BaseChannel,
        channel_agent_config: dict[str, Any],
        channel_workspace: str,
    ) -> dict[str, Any]:
        """Build the safe group-chat agent config for an external channel."""
        group_override = channel.config.extra.get("group_agent_config")
        group_config = merge_agent_config(
            channel_agent_config,
            build_group_safe_agent_override(group_override),
        )
        # Group chats share the per-channel workspace so inbound media/attachments
        # remain visible, but they must not escalate tool access.
        group_config["workspace"] = channel_workspace
        return group_config

    @staticmethod
    def _group_policy_enabled(channel: BaseChannel) -> bool:
        """Return True when this channel is configured to handle group chats."""
        if channel.name != "feishu":
            return False
        policy = str(channel.config.extra.get("group_policy", "allowlist")).strip().lower()
        return policy != "disabled"

    def _validate_group_agent_config(
        self,
        channel: BaseChannel,
        group_agent_config: dict[str, Any],
    ) -> None:
        """Reject unsafe group-chat agent configs at startup."""
        if bool(group_agent_config.get("yolo_mode")):
            raise ValueError(
                f"[{channel.full_name}] group_agent_config may not enable yolo_mode for group chats"
            )
        if uses_risky_local_tools(group_agent_config):
            raise ValueError(
                f"[{channel.full_name}] group_agent_config may not enable local shell/filesystem tools"
            )

    async def _create_managed_agent(
        self,
        agent_config: dict[str, Any],
        *,
        session_key: str,
    ) -> AgentLoop:
        """Create an AgentLoop from a resolved agent config."""
        from spoon_bot.agent.loop import create_agent

        create_kwargs: dict[str, Any] = {
            "model": agent_config.get("model"),
            "provider": agent_config.get("provider"),
            "api_key": agent_config.get("api_key"),
            "base_url": agent_config.get("base_url"),
            "workspace": agent_config.get("workspace"),
            "enable_skills": agent_config.get("enable_skills", True),
            "session_key": session_key,
            "config_path": self._config_path,
            "yolo_mode": bool(agent_config.get("yolo_mode", False)),
        }
        if agent_config.get("mcp_config") is not None:
            create_kwargs["mcp_config"] = agent_config["mcp_config"]
        if agent_config.get("shell_timeout") is not None:
            create_kwargs["shell_timeout"] = int(agent_config["shell_timeout"])
        if agent_config.get("max_output") is not None:
            create_kwargs["max_output"] = int(agent_config["max_output"])
        if agent_config.get("context_window") is not None:
            create_kwargs["context_window"] = int(agent_config["context_window"])
        if agent_config.get("enabled_tools") is not None:
            create_kwargs["enabled_tools"] = set(agent_config["enabled_tools"])
        if agent_config.get("tool_profile") is not None:
            create_kwargs["tool_profile"] = agent_config["tool_profile"]
        if agent_config.get("session_store_backend") is not None:
            create_kwargs["session_store_backend"] = agent_config["session_store_backend"]
        if agent_config.get("session_store_dsn") is not None:
            create_kwargs["session_store_dsn"] = agent_config["session_store_dsn"]
        if agent_config.get("session_store_db_path") is not None:
            create_kwargs["session_store_db_path"] = agent_config["session_store_db_path"]
        if agent_config.get("max_iterations") is not None:
            create_kwargs["max_iterations"] = int(agent_config["max_iterations"])
        if agent_config.get("auto_reload"):
            create_kwargs["auto_reload"] = True
            if agent_config.get("auto_reload_interval") is not None:
                create_kwargs["auto_reload_interval"] = float(
                    agent_config["auto_reload_interval"]
                )
        if agent_config.get("auto_commit") is not None:
            create_kwargs["auto_commit"] = bool(agent_config["auto_commit"])

        return await create_agent(**create_kwargs)

    async def _ensure_channel_agent(self, channel: BaseChannel) -> AgentLoop | None:
        """Create and attach a dedicated agent for an external channel."""
        if channel.name == "cli":
            return self._default_agent
        if not self._default_agent:
            return self._agent
        existing = self._channel_agents.get(channel.full_name)
        if existing is not None:
            channel.set_agent(existing)
            return existing

        async with self._agent_init_lock:
            existing = self._channel_agents.get(channel.full_name)
            if existing is not None:
                channel.set_agent(existing)
                return existing

            channel_agent_config = self._channel_agent_config(channel)
            channel_workspace = self._derive_channel_workspace(
                channel.full_name,
                channel.config.extra.get("agent_config"),
            )
            channel_agent_config["workspace"] = channel_workspace
            group_agent_config: dict[str, Any] | None = None
            if self._group_policy_enabled(channel):
                group_agent_config = self._group_agent_config(
                    channel,
                    channel_agent_config,
                    channel_workspace,
                )
                self._validate_group_agent_config(channel, group_agent_config)

            logger.info(
                f"Creating dedicated agent for {channel.full_name} "
                f"(workspace={channel_workspace})"
            )
            channel_agent = await self._create_managed_agent(
                channel_agent_config,
                session_key=channel.full_name,
            )
            self._channel_agents[channel.full_name] = channel_agent
            channel.set_agent(channel_agent)

            if group_agent_config is not None:
                logger.info(
                    f"Creating safe group agent for {channel.full_name} "
                    f"(workspace={channel_workspace})"
                )
                self._group_agents[channel.full_name] = await self._create_managed_agent(
                    group_agent_config,
                    session_key=f"{channel.full_name}:group",
                )

            return channel_agent

    async def _ensure_channel_agents(
        self,
        channels: list[BaseChannel] | None = None,
    ) -> None:
        """Ensure all requested channels have dedicated runtime agents."""
        if not self._default_agent:
            return
        targets = channels if channels is not None else list(self._channels.values())
        for channel in targets:
            await self._ensure_channel_agent(channel)

    async def _cleanup_isolated_agents(self) -> None:
        """Cleanup dedicated per-channel agents owned by the manager."""
        managed_agents: dict[int, AgentLoop] = {}
        for agent in (
            list(self._group_agents.values())
            + list(self._channel_agents.values())
            + list(self._scoped_agents.values())
        ):
            managed_agents[id(agent)] = agent

        self._group_agents.clear()
        self._channel_agents.clear()
        self._scoped_agents.clear()

        for agent in managed_agents.values():
            try:
                await agent.cleanup()
            except Exception as e:
                logger.debug(f"Managed agent cleanup failed: {e}")

    def _select_agent_for_message(self, message: InboundMessage) -> AgentLoop | None:
        """Choose the runtime agent for an inbound message."""
        is_dm = bool(message.metadata.get("is_dm", False))
        if not is_dm:
            group_agent = self._group_agents.get(message.channel)
            if group_agent is not None:
                return group_agent
        return self._channel_agents.get(message.channel) or self._agent

    @staticmethod
    def _scoped_agent_cache_key(channel_name: str, scope_key: str) -> str:
        """Build a stable cache key for scoped runtime agents."""
        return f"{channel_name}:{scope_key}"

    async def _ensure_scoped_agent_for_message(
        self,
        channel: BaseChannel | None,
        message: InboundMessage,
    ) -> AgentLoop | None:
        """Create a dedicated scoped agent for per-group or per-DM overrides."""
        if channel is None or not self._default_agent:
            return None

        scope_key = message.metadata.get("agent_scope_key")
        override = message.metadata.get("runtime_agent_override")
        if not isinstance(scope_key, str) or not scope_key.strip():
            return None
        if not isinstance(override, dict) or not override:
            return None

        cache_key = self._scoped_agent_cache_key(channel.full_name, scope_key.strip())
        existing = self._scoped_agents.get(cache_key)
        if existing is not None:
            return existing

        async with self._agent_init_lock:
            existing = self._scoped_agents.get(cache_key)
            if existing is not None:
                return existing

            channel_agent_config = self._channel_agent_config(channel)
            channel_workspace = self._derive_channel_workspace(
                channel.full_name,
                channel.config.extra.get("agent_config"),
            )
            channel_agent_config["workspace"] = channel_workspace

            is_dm = bool(message.metadata.get("is_dm", False))
            if is_dm:
                scoped_agent_config = merge_agent_config(channel_agent_config, override)
            else:
                base_group_config = self._group_agent_config(
                    channel,
                    channel_agent_config,
                    channel_workspace,
                )
                scoped_agent_config = merge_agent_config(base_group_config, override)
                scoped_agent_config["workspace"] = channel_workspace
                self._validate_group_agent_config(channel, scoped_agent_config)

            logger.info(f"Creating scoped agent for {cache_key}")
            scoped_agent = await self._create_managed_agent(
                scoped_agent_config,
                session_key=cache_key,
            )
            self._scoped_agents[cache_key] = scoped_agent
            return scoped_agent

    async def _resolve_agent_for_message(
        self,
        channel: BaseChannel | None,
        message: InboundMessage,
    ) -> AgentLoop | None:
        """Resolve the runtime agent for one message, including scoped overrides."""
        scoped_agent = await self._ensure_scoped_agent_for_message(channel, message)
        if scoped_agent is not None:
            return scoped_agent
        return self._select_agent_for_message(message)

    def get_channel(self, name: str) -> BaseChannel | None:
        """
        Get a channel by name.

        Args:
            name: Channel full name

        Returns:
            Channel instance or None
        """
        return self._channels.get(name)

    # Channel type registry - single source of truth for channel metadata
    # Each entry: kind -> {import_path, class_name, packages, install_extra, pip_package}
    _CHANNEL_REGISTRY = {
        "telegram": {
            "import_path": "spoon_bot.channels.telegram.channel",
            "class_name": "TelegramChannel",
            "packages": {"telegram", "python-telegram-bot"},
            "install_extra": "telegram",
            "pip_package": "python-telegram-bot[all]>=21.0",
        },
        "discord": {
            "import_path": "spoon_bot.channels.discord.channel",
            "class_name": "DiscordChannel",
            "packages": {"discord", "aiohttp"},
            "install_extra": "discord",
            "pip_package": "discord.py>=2.3.0",
        },
        "feishu": {
            "import_path": "spoon_bot.channels.feishu.channel",
            "class_name": "FeishuChannel",
            "packages": {"lark", "lark_oapi"},
            "install_extra": "feishu",
            "pip_package": "lark-oapi>=1.2.0",
        },
    }

    def _is_missing_dependency(self, kind: str, error: ImportError) -> bool:
        """
        Check if an ImportError is due to a missing external dependency.

        Handles two cases:
        1. Real import failures (error.name is set by Python import machinery)
        2. Manually raised ImportError/ModuleNotFoundError in __init__ when
           AVAILABLE flag is False (error.name may be None; fall back to
           matching the error message against known package names)

        Args:
            kind: Channel type name
            error: The ImportError to check

        Returns:
            True if this is a missing dependency, False if it's an internal error
        """
        registry_entry = self._CHANNEL_REGISTRY.get(kind, {})
        known_packages = registry_entry.get("packages", set())

        # Case 1: Real import failure — error.name is set by Python
        missing_name = getattr(error, "name", None)
        if missing_name:
            if missing_name in known_packages:
                return True
            for pkg in known_packages:
                if missing_name.startswith(f"{pkg}."):
                    return True

        # Case 2: Manually raised ImportError from __init__ (e.g.
        # "python-telegram-bot is required ...", "discord.py is required ...")
        # when the module-level import already failed silently.
        if missing_name is None:
            msg = str(error).lower()
            for pkg in known_packages:
                if pkg.lower() in msg:
                    return True
            # Also match the pip package name (e.g. "python-telegram-bot")
            pip_pkg = registry_entry.get("pip_package", "")
            if pip_pkg and pip_pkg.split(">=")[0].split("[")[0].lower() in msg:
                return True

        return False

    def _load_channel_type(
        self,
        kind: str,
        import_path: str,
        class_name: str,
        configs: list[tuple[Any, str]],
        missing_deps: list[str],
    ) -> None:
        """
        Load channels of a specific type.

        This unified loader handles ImportError collection for all channel types,
        reducing code duplication. It distinguishes between missing dependencies
        (which get installation hints) and internal import errors (which are
        logged with full traceback for debugging).

        Args:
            kind: Channel type name (e.g., "telegram", "discord", "feishu")
            import_path: Full module path to import from
            class_name: Channel class name to instantiate
            configs: List of (config, account_id) tuples
            missing_deps: List to append missing dependency names to
        """
        import importlib
        import traceback

        for config, account_id in configs:
            try:
                module = importlib.import_module(import_path)
                channel_class = getattr(module, class_name)
                channel = channel_class(config, account_id)
                self.add_channel(channel)
            except ImportError as e:
                if self._is_missing_dependency(kind, e):
                    # Missing external dependency - suggest installation
                    logger.warning(f"Missing {kind} dependency: {e.name or e}")
                    if kind not in missing_deps:
                        missing_deps.append(kind)
                else:
                    # Internal import error - log with full traceback but
                    # do NOT crash; skip this channel and continue loading others.
                    logger.error(
                        f"Internal import error in {kind} channel module:\n"
                        f"{traceback.format_exc()}"
                    )

    def _build_install_hint(self, missing_deps: list[str]) -> str:
        """
        Build installation hint message for missing dependencies.

        Args:
            missing_deps: List of missing channel types

        Returns:
            Formatted installation hint string
        """
        extras = []
        packages = []
        for kind in missing_deps:
            entry = self._CHANNEL_REGISTRY.get(kind, {})
            extras.append(entry.get("install_extra", kind))
            packages.append(entry.get("pip_package", kind))

        extras_str = ",".join(extras)
        return (
            f"Missing dependencies for channels: {', '.join(missing_deps)}. "
            f"Install with: uv pip install -e \".[{extras_str}]\""
        )

    async def load_from_config(
        self,
        config_path: str | Path | None = None,
        include_cli: bool = True,
    ) -> None:
        """
        Load channels from configuration file.

        Args:
            config_path: Path to config file (uses default locations if None)
            include_cli: Whether to create the CLI channel when enabled

        Raises:
            ImportError: If required channel dependencies are missing
        """
        if config_path is not None:
            self._config_path = Path(config_path).expanduser()
        self._config = load_channels_config(config_path)
        logger.info("Configuration loaded, creating channels...")

        missing_deps: list[str] = []

        # Config getters for each channel type
        config_getters = {
            "telegram": self._config.get_telegram_configs,
            "discord": self._config.get_discord_configs,
            "feishu": self._config.get_feishu_configs,
        }

        # Load all channel types using registry-driven loader
        for kind, entry in self._CHANNEL_REGISTRY.items():
            config_getter = config_getters.get(kind)
            if config_getter:
                configs = config_getter()
                if configs:
                    self._load_channel_type(
                        kind,
                        entry["import_path"],
                        entry["class_name"],
                        configs,
                        missing_deps,
                    )

        # CLI channel (if enabled) - special case, no external deps
        if include_cli and self._config.is_cli_enabled():
            try:
                from spoon_bot.channels.cli_channel import CLIChannel

                cli_channel = CLIChannel()
                self.add_channel(cli_channel)
            except ImportError as e:
                logger.warning(f"Failed to load CLI channel: {e}")

        # Warn (but don't crash) if some configured channels are missing dependencies
        if missing_deps:
            logger.warning(self._build_install_hint(missing_deps))

        logger.info(f"Loaded {len(self._channels)} channels from configuration")

    async def reload_config(self, config_path: str | Path | None = None) -> None:
        """
        Hot reload configuration.

        Stops existing channels, reloads config, and starts new channels.

        Args:
            config_path: Path to config file
        """
        logger.info("Reloading configuration...")

        # Save running state before stopping (stop_all sets _running to False)
        was_running = self._running

        # Stop all channels
        await self.stop_all()

        # Clear channels
        self._channels.clear()

        # Reload
        await self.load_from_config(config_path)

        # Restart if was running before reload
        if was_running:
            await self.start_all()

        logger.info("Configuration reloaded")

    async def start_all(self) -> None:
        """Start all channels and the message bus."""
        if not self._agent:
            raise RuntimeError("No agent set. Call set_agent() first.")

        if self._running:
            logger.warning("ChannelManager already running")
            return

        await self._ensure_channel_agents()

        # Start message bus
        await self._bus.start()

        # Start all channels
        started = 0
        for channel in self._channels.values():
            if not channel.config.enabled:
                logger.debug(f"Skipping disabled channel: {channel.full_name}")
                continue

            try:
                await channel.start()
                started += 1
            except Exception as e:
                logger.error(f"Failed to start channel {channel.full_name}: {e}")

        self._running = True

        # Start health check loop
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            f"ChannelManager started: {started}/{len(self._channels)} channels running"
        )

    async def start_channels(self, channel_names: list[str]) -> None:
        """
        Start specific channels.

        Args:
            channel_names: List of channel names to start (e.g., ["telegram", "discord"])
        """
        if not self._agent:
            raise RuntimeError("No agent set. Call set_agent() first.")

        matching_channels = [
            ch
            for ch in self._channels.values()
            if ch.name in channel_names or ch.full_name in channel_names
        ]
        await self._ensure_channel_agents(matching_channels)

        # Start message bus if not running
        if not self._bus.is_running:
            await self._bus.start()

        for name in channel_names:
            # Find channels matching this name (may be multiple accounts)
            matching = [
                ch for ch in self._channels.values() if ch.name == name or ch.full_name == name
            ]

            for channel in matching:
                if not channel.config.enabled:
                    logger.warning(f"Channel {channel.full_name} is disabled in config")
                    continue

                try:
                    await channel.start()
                    logger.info(f"Started channel: {channel.full_name}")
                except Exception as e:
                    logger.error(f"Failed to start channel {channel.full_name}: {e}")

        self._running = True

        # Start health check if not already running
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def stop_all(self) -> None:
        """Stop all channels and the message bus."""
        if not self._running and not self._channel_agents and not self._group_agents:
            return

        # Stop health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._running:
            # Stop all channels
            for channel in self._channels.values():
                try:
                    await channel.stop()
                except Exception as e:
                    logger.error(f"Failed to stop channel {channel.full_name}: {e}")

            # Stop message bus
            await self._bus.stop()

        await self._cleanup_isolated_agents()

        self._running = False
        logger.info("ChannelManager stopped")

    async def stop(self) -> None:
        """Stop all channels and the message bus (alias for stop_all)."""
        await self.stop_all()

    async def stop_channels(self, channel_names: list[str]) -> None:
        """
        Stop specific channels.

        Args:
            channel_names: List of channel names to stop
        """
        for name in channel_names:
            matching = [
                ch for ch in self._channels.values() if ch.name == name or ch.full_name == name
            ]

            for channel in matching:
                try:
                    await channel.stop()
                    logger.info(f"Stopped channel: {channel.full_name}")
                except Exception as e:
                    logger.error(f"Failed to stop channel {channel.full_name}: {e}")

    async def health_check_all(self) -> dict[str, Any]:
        """
        Get health status of all channels.

        Returns:
            Dictionary with health information for each channel
        """
        health = {
            "manager_running": self._running,
            "bus_running": self._bus.is_running,
            "total_channels": len(self._channels),
            "channels": {},
        }

        for name, channel in self._channels.items():
            try:
                channel_health = await channel.health_check()
                health["channels"][name] = channel_health
            except Exception as e:
                health["channels"][name] = {
                    "status": "error",
                    "error": str(e),
                }

        # Count by status
        running = sum(
            1
            for ch in health["channels"].values()
            if ch.get("status") == ChannelStatus.RUNNING.value
        )
        health["running_channels"] = running

        return health

    async def _health_check_loop(self) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                health = await self.health_check_all()
                logger.debug(
                    f"Health: {health['running_channels']}/{health['total_channels']} channels running"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    @staticmethod
    def _try_builtin_command(message: InboundMessage) -> OutboundMessage | None:
        """Handle built-in slash commands that don't need the agent.

        Returns an ``OutboundMessage`` if the message was a recognised
        built-in command, or ``None`` to fall through to agent processing.
        """
        text = (message.content or "").strip()
        if text.lower() in ("/myid", "/myid@bot"):
            try:
                from spoon_bot.gateway.api.v1.identity import format_identity_text

                reply = format_identity_text()
            except Exception as exc:
                reply = f"Identity lookup failed: {exc}"
            return OutboundMessage(
                content=reply,
                channel=message.channel,
                reply_to=message.message_id,
                metadata=message.metadata.copy(),
            )
        return None

    async def _handle_message(self, message: InboundMessage) -> OutboundMessage | None:
        """
        Handle incoming message by routing to agent.

        Args:
            message: Inbound message to process.

        Returns:
            Agent's response as OutboundMessage.
        """
        builtin_response = self._try_builtin_command(message)
        if builtin_response is not None:
            return builtin_response

        channel = self._channels.get(message.channel)
        agent = await self._resolve_agent_for_message(channel, message)
        if not agent:
            logger.error("No agent set")
            return None

        # Circuit breaker: reject immediately if LLM is known to be down
        if not self._circuit_breaker.allow_request():
            logger.warning(
                f"[{message.channel}] Circuit breaker OPEN, rejecting message"
            )
            return OutboundMessage(
                content=(
                    "The AI service is temporarily unavailable. "
                    "Please try again in a moment."
                ),
                channel=message.channel,
                reply_to=message.message_id,
                metadata=message.metadata.copy(),
            )

        # Notify channel that processing is starting (typing indicators, etc.)
        if channel:
            try:
                await channel.on_processing_start(message)
            except Exception as e:
                logger.debug(f"on_processing_start failed: {e}")

        try:
            logger.info(
                f"[{message.channel}] Agent processing message from {message.sender_id}: "
                f"{message.content[:50]}..."
            )

            # Inject sender name prefix for group chats so the LLM can
            # distinguish who is speaking in multi-user conversations.
            content = message.content
            chat_type = message.metadata.get("chat_type", "")
            is_dm = message.metadata.get("is_dm", False)
            if message.sender_name and not is_dm and chat_type != "dm":
                content = f"[{message.sender_name}]: {content}"

            # Route based on think/verbose metadata from channel
            think_level = message.metadata.get("think_level", "off")
            raw_think_level = str(think_level or "").strip().lower()
            reasoning_effort = self._reasoning_effort_from_think_level(think_level)
            thinking_enabled = raw_think_level in {
                "basic",
                "extended",
                "low",
                "medium",
                "high",
                "xhigh",
            }
            media = message.media if message.has_media else None
            raw_attachments = message.metadata.get("attachments")
            attachments = raw_attachments if isinstance(raw_attachments, list) else None
            session_key = message.session_key or None

            if thinking_enabled:
                process_kwargs = {
                    "message": content,
                    "media": media,
                    "session_key": session_key,
                    "attachments": attachments,
                }
                if reasoning_effort:
                    process_kwargs["reasoning_effort"] = reasoning_effort
                response_text, thinking_content = await agent.process_with_thinking(**process_kwargs)
                # Include thinking content in verbose mode
                if message.metadata.get("verbose") and thinking_content:
                    response_text = (
                        f"\ud83d\udcad *Thinking:*\n{thinking_content}\n\n"
                        f"---\n\n{response_text}"
                    )
            else:
                process_kwargs = {
                    "message": content,
                    "media": media,
                    "session_key": session_key,
                    "attachments": attachments,
                }
                if reasoning_effort:
                    process_kwargs["reasoning_effort"] = reasoning_effort
                response_text = await agent.process(**process_kwargs)

            self._circuit_breaker.record_success()

            if not response_text:
                logger.warning(
                    f"[{message.channel}] Agent returned empty response for {message.sender_id}"
                )
                response_text = "I processed your request but have no output to share. Please try again."

            logger.info(
                f"[{message.channel}] Agent response ready "
                f"({len(response_text)} chars) for {message.sender_id}"
            )

            # Create outbound message
            return OutboundMessage(
                content=response_text,
                channel=message.channel,
                reply_to=message.message_id,
                metadata=message.metadata.copy(),
            )

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error handling message: {e}", exc_info=True)
            # Use a generic user-facing message to avoid leaking internal
            # details (file paths, API keys, tracebacks).
            try:
                from spoon_bot.exceptions import user_friendly_error
                friendly = user_friendly_error(e)
            except Exception:
                friendly = "Sorry, an unexpected error occurred. Please try again."
            return OutboundMessage(
                content=friendly,
                channel=message.channel,
                reply_to=message.message_id,
                metadata=message.metadata.copy(),
            )
        finally:
            # Always stop typing indicators, even on error
            if channel:
                try:
                    await channel.on_processing_end(message)
                except Exception as e:
                    logger.debug(f"on_processing_end failed: {e}")

    @property
    def channel_names(self) -> list[str]:
        """Get list of channel full names."""
        return list(self._channels.keys())

    @property
    def channels_by_type(self) -> dict[str, list[str]]:
        """Get channels grouped by type."""
        by_type: dict[str, list[str]] = {}
        for channel in self._channels.values():
            if channel.name not in by_type:
                by_type[channel.name] = []
            by_type[channel.name].append(channel.full_name)
        return by_type

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    @property
    def running_channels_count(self) -> int:
        """Get count of running channels."""
        return sum(1 for ch in self._channels.values() if ch.is_running)
