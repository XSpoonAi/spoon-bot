"""Spawn tool for background task execution."""

from __future__ import annotations

import asyncio
from typing import Any, TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from spoon_bot.agent.tools.base import Tool

if TYPE_CHECKING:
    pass


class BackgroundTask:
    """Represents a background task."""

    def __init__(self, task_id: str, label: str, task: asyncio.Task):
        self.task_id = task_id
        self.label = label
        self.task = task
        self.result: str | None = None
        self.error: str | None = None


class SpawnTool(Tool):
    """
    Tool for spawning background tasks.

    Allows the agent to run tasks in the background and continue
    with other work.
    """

    def __init__(self):
        """Initialize spawn tool."""
        self._tasks: dict[str, BackgroundTask] = {}
        self._on_complete_callback: Any = None

    def set_completion_callback(self, callback: Any) -> None:
        """Set callback for task completion."""
        self._on_complete_callback = callback

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return """Spawn background tasks. Actions:
- run <task>: Run a task in the background
- status: List all background tasks
- result <task_id>: Get result of a completed task
- cancel <task_id>: Cancel a running task"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run", "status", "result", "cancel"],
                    "description": "Action to perform",
                },
                "task": {
                    "type": "string",
                    "description": "Task description to run",
                },
                "label": {
                    "type": "string",
                    "description": "Label for the task",
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID for result/cancel",
                },
            },
            "required": ["action"],
        }

    async def _run_background_task(
        self,
        task_id: str,
        task_description: str,
    ) -> None:
        """Run a task in the background."""
        bg_task = self._tasks[task_id]

        try:
            # In a full implementation, this would create a subagent
            # For now, simulate background work
            await asyncio.sleep(0.1)

            # Mark as complete
            bg_task.result = f"Task '{task_description}' completed successfully"
            logger.info(f"Background task {task_id} completed")

            # Notify completion
            if self._on_complete_callback:
                await self._on_complete_callback(task_id, bg_task.result)

        except asyncio.CancelledError:
            bg_task.error = "Task was cancelled"
            logger.info(f"Background task {task_id} cancelled")
        except Exception as e:
            bg_task.error = str(e)
            logger.error(f"Background task {task_id} failed: {e}")

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "status")
        task_description = kwargs.get("task")
        label = kwargs.get("label", "Background task")
        task_id = kwargs.get("task_id")

        if action == "run":
            if not task_description:
                return "Error: 'task' is required for 'run' action"

            new_task_id = str(uuid4())[:8]

            # Create and start background task
            task = asyncio.create_task(
                self._run_background_task(new_task_id, task_description)
            )
            bg_task = BackgroundTask(new_task_id, label, task)
            self._tasks[new_task_id] = bg_task

            logger.info(f"Spawned background task: {new_task_id}")
            return f"Spawned task {new_task_id}: {label}"

        elif action == "status":
            if not self._tasks:
                return "No background tasks"

            lines = ["Background tasks:"]
            for tid, task in self._tasks.items():
                status = "completed" if task.result else (
                    "failed" if task.error else "running"
                )
                lines.append(f"  [{tid}] {task.label}: {status}")

            return "\n".join(lines)

        elif action == "result":
            if not task_id:
                return "Error: 'task_id' is required for 'result' action"

            task = self._tasks.get(task_id)
            if not task:
                return f"Task {task_id} not found"

            if task.error:
                return f"Task {task_id} failed: {task.error}"
            elif task.result:
                return f"Task {task_id} result: {task.result}"
            else:
                return f"Task {task_id} is still running"

        elif action == "cancel":
            if not task_id:
                return "Error: 'task_id' is required for 'cancel' action"

            task = self._tasks.get(task_id)
            if not task:
                return f"Task {task_id} not found"

            task.task.cancel()
            return f"Cancelled task {task_id}"

        return f"Unknown action: {action}"
