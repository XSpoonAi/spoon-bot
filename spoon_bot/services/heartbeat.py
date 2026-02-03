"""Heartbeat service for periodic task execution."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable

from loguru import logger


class HeartbeatService:
    """
    Periodic heartbeat service.

    Reads tasks from HEARTBEAT.md and executes them at intervals.
    """

    def __init__(
        self,
        workspace: Path | str,
        interval: int = 1800,  # 30 minutes default
    ):
        """
        Initialize heartbeat service.

        Args:
            workspace: Workspace directory.
            interval: Interval in seconds between heartbeats.
        """
        self._workspace = Path(workspace)
        self._interval = interval
        self._running = False
        self._task: asyncio.Task | None = None
        self._handler: Callable[[str], Awaitable[str]] | None = None
        self._heartbeat_file = self._workspace / "HEARTBEAT.md"

    def set_handler(self, handler: Callable[[str], Awaitable[str]]) -> None:
        """
        Set the handler for executing heartbeat tasks.

        Args:
            handler: Async function that processes task strings.
        """
        self._handler = handler

    def _ensure_heartbeat_file(self) -> None:
        """Ensure HEARTBEAT.md exists with default content."""
        if not self._heartbeat_file.exists():
            self._heartbeat_file.write_text("""# Heartbeat Tasks

Tasks in this file are executed periodically by the agent.
Use checkbox format: `- [ ] Task description`

## Periodic Tasks

- [ ] Check for new messages
- [ ] Review today's notes

## Status

Last heartbeat: Never
""")

    def _parse_tasks(self) -> list[str]:
        """Parse uncompleted tasks from HEARTBEAT.md."""
        if not self._heartbeat_file.exists():
            return []

        content = self._heartbeat_file.read_text(encoding="utf-8")
        tasks = []

        for line in content.split("\n"):
            line = line.strip()
            # Match uncompleted task format: - [ ] Task
            if line.startswith("- [ ] "):
                task = line[6:].strip()
                if task:
                    tasks.append(task)

        return tasks

    def _update_last_heartbeat(self) -> None:
        """Update the last heartbeat timestamp in the file."""
        if not self._heartbeat_file.exists():
            return

        content = self._heartbeat_file.read_text(encoding="utf-8")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Replace last heartbeat line
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("Last heartbeat:"):
                lines[i] = f"Last heartbeat: {timestamp}"
                break
        else:
            # Add if not found
            lines.append(f"\nLast heartbeat: {timestamp}")

        self._heartbeat_file.write_text("\n".join(lines), encoding="utf-8")

    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop."""
        self._ensure_heartbeat_file()

        while self._running:
            try:
                # Wait for interval
                await asyncio.sleep(self._interval)

                if not self._running:
                    break

                logger.info("Heartbeat triggered")

                # Parse tasks
                tasks = self._parse_tasks()
                if not tasks:
                    logger.debug("No heartbeat tasks to execute")
                    continue

                # Execute tasks
                if self._handler:
                    for task in tasks:
                        logger.info(f"Executing heartbeat task: {task}")
                        try:
                            await self._handler(task)
                        except Exception as e:
                            logger.error(f"Heartbeat task failed: {e}")

                # Update timestamp
                self._update_last_heartbeat()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def start(self) -> None:
        """Start the heartbeat service."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"Heartbeat service started (interval: {self._interval}s)")

    async def stop(self) -> None:
        """Stop the heartbeat service."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Heartbeat service stopped")

    async def trigger_now(self) -> list[str]:
        """Trigger heartbeat immediately and return executed tasks."""
        if not self._handler:
            logger.warning("No handler set for heartbeat")
            return []

        tasks = self._parse_tasks()
        results = []

        for task in tasks:
            try:
                result = await self._handler(task)
                results.append(f"{task}: {result[:100]}...")
            except Exception as e:
                results.append(f"{task}: Error - {e}")

        self._update_last_heartbeat()
        return results

    @property
    def is_running(self) -> bool:
        """Check if service is running."""
        return self._running
