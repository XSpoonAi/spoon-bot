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
) -> ChannelManager:
    """Create a ChannelManager, load channels from config, and start them.

    Args:
        agent: Initialized AgentLoop instance.
        config_path: Path to config YAML (uses default locations if None).
        channel_names: If provided, only start these channels; otherwise start all.

    Returns:
        Running ChannelManager instance.

    Raises:
        FileNotFoundError: If config file not found.
        ImportError: If channel dependencies are missing.
    """
    from spoon_bot.channels.manager import ChannelManager

    manager = ChannelManager()
    manager.set_agent(agent)
    await manager.load_from_config(config_path)

    if channel_names:
        await manager.start_channels(channel_names)
    else:
        await manager.start_all()

    logger.info(
        f"ChannelManager ready: {manager.running_channels_count} channel(s) running"
    )
    return manager
