"""Background file-watcher that auto-reloads skills and MCP on changes.

Uses pure ``asyncio`` polling (no external dependencies).  Tracks
modification times of skill files and config.yaml, then calls the
corresponding ``AgentLoop.reload_*()`` methods when changes are detected.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from loguru import logger

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop


class HotReloadService:
    """Poll skill paths and config file for changes; trigger agent reload."""

    def __init__(
        self,
        agent: AgentLoop,
        poll_interval: float = 5.0,
        watch_skills: bool = True,
        watch_config: bool = True,
        config_path: Path | str | None = None,
        debounce: float = 2.0,
    ) -> None:
        self._agent = agent
        self._poll_interval = max(poll_interval, 1.0)
        self._watch_skills = watch_skills
        self._watch_config = watch_config
        self._config_path = Path(config_path) if config_path else None
        self._debounce = debounce

        self._task: asyncio.Task[None] | None = None
        self._prev_skill_snapshot: dict[str, float] = {}
        self._prev_mcp_hash: int = 0
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background polling loop."""
        if self._running:
            return
        self._running = True

        # Take an initial snapshot so we only react to *future* changes.
        if self._watch_skills:
            self._prev_skill_snapshot = self._snapshot_skill_mtimes()
        if self._watch_config and self._config_path:
            self._prev_mcp_hash = self._hash_mcp_section()

        self._task = asyncio.create_task(self._poll_loop())
        logger.info(
            f"HotReloadService started (interval={self._poll_interval}s, "
            f"skills={self._watch_skills}, config={self._watch_config})"
        )

    async def stop(self) -> None:
        """Cancel the background polling loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("HotReloadService stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        """Main polling loop — runs until cancelled."""
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break

            try:
                await self._check_for_changes()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning(f"HotReloadService poll error: {exc}")

    async def _check_for_changes(self) -> None:
        skills_changed = False
        mcp_changed = False

        if self._watch_skills:
            new_snapshot = self._snapshot_skill_mtimes()
            if new_snapshot != self._prev_skill_snapshot:
                skills_changed = True
                self._prev_skill_snapshot = new_snapshot

        if self._watch_config and self._config_path:
            new_hash = self._hash_mcp_section()
            if new_hash != self._prev_mcp_hash:
                mcp_changed = True
                self._prev_mcp_hash = new_hash

        if not skills_changed and not mcp_changed:
            return

        # Debounce: wait a bit then re-check to avoid rapid-fire reloads
        await asyncio.sleep(self._debounce)

        if skills_changed:
            logger.info("HotReloadService: skill files changed, reloading skills")
            try:
                result = await self._agent.reload_skills()
                logger.info(f"Auto-reload skills result: {result}")
            except Exception as exc:
                logger.error(f"Auto-reload skills failed: {exc}")

        if mcp_changed:
            logger.info("HotReloadService: MCP config changed, reloading MCP")
            try:
                new_config = self._read_mcp_config()
                result = await self._agent.reload_mcp(new_config)
                logger.info(f"Auto-reload MCP result: {result}")
            except Exception as exc:
                logger.error(f"Auto-reload MCP failed: {exc}")

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------

    def _snapshot_skill_mtimes(self) -> dict[str, float]:
        """Collect mtime of every skill-relevant file under skill paths."""
        snapshot: dict[str, float] = {}
        suffixes = {".py", ".md", ".yaml", ".yml", ".json"}
        for skill_dir in self._agent._skill_paths:
            if not skill_dir.exists():
                continue
            for f in skill_dir.rglob("*"):
                if f.is_file() and f.suffix in suffixes:
                    try:
                        snapshot[str(f)] = f.stat().st_mtime
                    except OSError:
                        pass
        return snapshot

    def _hash_mcp_section(self) -> int:
        """Hash the MCP config section of the config file."""
        if not self._config_path or not self._config_path.exists():
            return 0
        try:
            with open(self._config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            agent_cfg = cfg.get("agent", {})
            mcp = agent_cfg.get("mcp_config") or agent_cfg.get("mcp_servers") or {}
            return hash(str(sorted(str(mcp).split())))
        except Exception:
            return 0

    def _read_mcp_config(self) -> dict[str, dict[str, Any]] | None:
        """Read and return the MCP config section from the config file."""
        if not self._config_path or not self._config_path.exists():
            return None
        try:
            with open(self._config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            agent_cfg = cfg.get("agent", {})
            return agent_cfg.get("mcp_config") or agent_cfg.get("mcp_servers")
        except Exception:
            return None
