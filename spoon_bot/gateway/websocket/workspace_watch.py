"""Workspace watch service for websocket file change notifications."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

from loguru import logger


WorkspaceWatchCallback = Callable[[dict[str, Any]], Awaitable[None]]
SANDBOX_WORKSPACE_ROOT = "/workspace"


@dataclass(slots=True)
class WorkspaceWatch:
    """Connection-scoped workspace watch."""

    watch_id: str
    target: Path
    sandbox_path: str
    recursive: bool
    snapshot: dict[str, dict[str, Any]]
    task: asyncio.Task[None] | None = None


class WorkspaceWatchService:
    """Poll workspace paths and emit file change events."""

    def __init__(
        self,
        workspace_root: Path,
        emit_change: WorkspaceWatchCallback,
        *,
        poll_interval: float = 0.75,
    ) -> None:
        self._workspace_root = workspace_root.resolve()
        self._emit_change = emit_change
        self._poll_interval = max(poll_interval, 0.1)
        self._watches: dict[str, WorkspaceWatch] = {}

    async def add_watch(self, path: str, recursive: bool = False) -> str:
        target, sandbox_path = self._resolve_watch_target(path)
        watch_id = f"watch_{uuid4().hex}"
        watch = WorkspaceWatch(
            watch_id=watch_id,
            target=target,
            sandbox_path=sandbox_path,
            recursive=recursive and target.is_dir(),
            snapshot=self._snapshot_target(target, recursive=recursive and target.is_dir()),
        )
        watch.task = asyncio.create_task(self._watch_loop(watch))
        self._watches[watch_id] = watch
        return watch_id

    async def remove_watch(self, watch_id: str) -> bool:
        watch = self._watches.pop(str(watch_id), None)
        if watch is None:
            return False
        await self._stop_watch(watch)
        return True

    async def close(self) -> None:
        watches = list(self._watches.values())
        self._watches.clear()
        for watch in watches:
            await self._stop_watch(watch)

    def _resolve_watch_target(self, requested_path: str) -> tuple[Path, str]:
        raw_path = str(requested_path or "").strip()
        if raw_path == "":
            target = self._workspace_root
        elif raw_path.startswith("/"):
            normalized = Path(raw_path).as_posix()
            sandbox_root = SANDBOX_WORKSPACE_ROOT.rstrip("/")
            if normalized != sandbox_root and not normalized.startswith(sandbox_root + "/"):
                workspace_root_str = self._workspace_root.as_posix().rstrip("/")
                if normalized != workspace_root_str and not normalized.startswith(workspace_root_str + "/"):
                    raise ValueError("Path is outside workspace boundary")
                relative = normalized[len(workspace_root_str):].lstrip("/")
            else:
                relative = normalized[len(sandbox_root):].lstrip("/")
            target = (self._workspace_root / relative).resolve()
        else:
            target = (self._workspace_root / raw_path).resolve()

        try:
            relative = target.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ValueError("Path is outside workspace boundary") from exc

        if not target.exists():
            raise ValueError(f"Path not found: {requested_path}")

        sandbox_path = self._sandbox_path_for(relative)
        return target, sandbox_path

    def _sandbox_path_for(self, relative: Path) -> str:
        relative_str = relative.as_posix()
        if relative_str in ("", "."):
            return SANDBOX_WORKSPACE_ROOT
        return f"{SANDBOX_WORKSPACE_ROOT.rstrip('/')}/{relative_str}"

    async def _watch_loop(self, watch: WorkspaceWatch) -> None:
        previous = watch.snapshot
        try:
            while True:
                await asyncio.sleep(self._poll_interval)
                current = self._snapshot_target(watch.target, recursive=watch.recursive)
                await self._emit_diff(watch, previous, current)
                previous = current
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Workspace watch loop failed: {}", exc)

    async def _emit_diff(
        self,
        watch: WorkspaceWatch,
        previous: dict[str, dict[str, Any]],
        current: dict[str, dict[str, Any]],
    ) -> None:
        previous_paths = set(previous)
        current_paths = set(current)

        for path in sorted(previous_paths - current_paths):
            await self._emit_change({
                "watch_id": watch.watch_id,
                "path": path,
                "change_type": "deleted",
            })

        for path in sorted(current_paths - previous_paths):
            payload = {
                "watch_id": watch.watch_id,
                "path": path,
                "change_type": "created",
            }
            payload.update(_snapshot_metadata(current[path]))
            await self._emit_change(payload)

        for path in sorted(previous_paths & current_paths):
            if previous[path] == current[path]:
                continue
            if current[path]["type"] == "dir":
                continue
            payload = {
                "watch_id": watch.watch_id,
                "path": path,
                "change_type": "modified",
            }
            payload.update(_snapshot_metadata(current[path]))
            await self._emit_change(payload)

    async def _stop_watch(self, watch: WorkspaceWatch) -> None:
        task = watch.task
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _snapshot(self, watch: WorkspaceWatch) -> dict[str, dict[str, Any]]:
        return self._snapshot_target(watch.target, recursive=watch.recursive)

    def _snapshot_target(self, target: Path, *, recursive: bool) -> dict[str, dict[str, Any]]:
        if not target.exists():
            return {}

        if target.is_file():
            try:
                relative = target.relative_to(self._workspace_root)
            except ValueError:
                return {}
            return {self._sandbox_path_for(relative): self._entry_snapshot(target)}

        entries: dict[str, dict[str, Any]] = {}
        iterator = target.rglob("*") if recursive else target.iterdir()
        for entry in iterator:
            try:
                relative = entry.relative_to(self._workspace_root)
            except ValueError:
                continue
            entries[self._sandbox_path_for(relative)] = self._entry_snapshot(entry)
        return entries

    @staticmethod
    def _entry_snapshot(entry: Path) -> dict[str, Any]:
        stat = entry.stat()
        is_dir = entry.is_dir()
        return {
            "type": "dir" if is_dir else "file",
            "mtime_ns": stat.st_mtime_ns,
            "mtime": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat().replace("+00:00", "Z"),
            "size": None if is_dir else stat.st_size,
        }


def _snapshot_metadata(snapshot: dict[str, Any]) -> dict[str, Any]:
    mtime_ns = snapshot.get("mtime_ns")
    size = snapshot.get("size")
    payload: dict[str, Any] = {}
    if isinstance(size, int):
        payload["size"] = size
    mtime = snapshot.get("mtime")
    if isinstance(mtime, str) and mtime:
        payload["mtime"] = mtime
    if isinstance(mtime_ns, int):
        payload["mtime_ns"] = mtime_ns
    return payload
