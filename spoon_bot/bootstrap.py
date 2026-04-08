"""Shared startup helpers for all entry points (CLI agent, CLI gateway, Docker).

Provides the common ``init_channels`` function so that every entry point can
load and start communication channels without duplicating the setup logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop
    from spoon_bot.channels.manager import ChannelManager


async def init_channels(
    agent: AgentLoop,
    config_path: str | Path | None = None,
    channel_names: list[str] | None = None,
    cli_enabled: bool = False,
) -> ChannelManager:
    """Create a ChannelManager, load channels from config, and start them.

    Args:
        agent: Initialized AgentLoop instance.
        config_path: Path to config YAML (uses default locations if None).
        channel_names: If provided, only start these channels; otherwise start all.
        cli_enabled: Whether to keep the CLI channel.  Defaults to ``False``
            because the CLI agent REPL and Docker mode both have their own
            input handling and would conflict with CLIChannel on stdin.

    Returns:
        Running ChannelManager instance.

    Raises:
        FileNotFoundError: If config file not found.
        ImportError: If channel dependencies are missing.
    """
    from spoon_bot.channels.manager import ChannelManager
    from spoon_bot.channels.config import load_agent_config

    manager = ChannelManager()
    manager.set_agent(
        agent,
        agent_config=load_agent_config(config_path),
        config_path=config_path,
    )
    await manager.load_from_config(config_path, include_cli=cli_enabled)

    # Remove CLI channel unless explicitly requested — avoids stdin conflicts
    # with the Rich REPL (agent mode) or Docker (no tty).
    if not cli_enabled:
        manager.remove_channel("cli:default")

    if channel_names:
        await manager.start_channels(channel_names)
    else:
        await manager.start_all()

    logger.info(
        f"ChannelManager ready: {manager.running_channels_count} channel(s) running"
    )
    return manager
