from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from spoon_bot.gateway.websocket.workspace_watch import WorkspaceWatchService


@pytest.mark.asyncio
async def test_workspace_watch_service_emits_created_event(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    events: list[dict[str, object]] = []

    async def emit_change(payload: dict[str, object]) -> None:
        events.append(payload)

    service = WorkspaceWatchService(workspace, emit_change, poll_interval=0.05)
    watch_id = await service.add_watch("/workspace", recursive=True)

    created_file = workspace / "hello.txt"
    created_file.write_text("hello", encoding="utf-8")

    for _ in range(20):
        if any(event.get("path") == "/workspace/hello.txt" for event in events):
            break
        await asyncio.sleep(0.05)

    await service.close()

    assert any(
        event.get("watch_id") == watch_id
        and event.get("path") == "/workspace/hello.txt"
        and event.get("change_type") == "created"
        for event in events
    )


@pytest.mark.asyncio
async def test_workspace_watch_service_remove_watch_stops_events(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    events: list[dict[str, object]] = []

    async def emit_change(payload: dict[str, object]) -> None:
        events.append(payload)

    service = WorkspaceWatchService(workspace, emit_change, poll_interval=0.05)
    watch_id = await service.add_watch("/workspace", recursive=True)
    removed = await service.remove_watch(watch_id)
    assert removed is True

    (workspace / "after-close.txt").write_text("later", encoding="utf-8")
    await asyncio.sleep(0.15)
    await service.close()

    assert events == []


@pytest.mark.asyncio
async def test_workspace_watch_service_survives_transient_missing_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    events: list[dict[str, object]] = []

    async def emit_change(payload: dict[str, object]) -> None:
        events.append(payload)

    service = WorkspaceWatchService(workspace, emit_change, poll_interval=0.05)
    await service.add_watch("/workspace", recursive=True)

    volatile_file = workspace / "volatile.txt"
    volatile_file.write_text("volatile", encoding="utf-8")

    original_entry_snapshot = WorkspaceWatchService._entry_snapshot
    state = {"failed": False}

    def flaky_entry_snapshot(entry: Path) -> dict[str, object]:
        if entry.name == "volatile.txt" and not state["failed"]:
            state["failed"] = True
            if entry.exists():
                entry.unlink()
            raise FileNotFoundError(str(entry))
        return original_entry_snapshot(entry)

    monkeypatch.setattr(
        WorkspaceWatchService,
        "_entry_snapshot",
        staticmethod(flaky_entry_snapshot),
    )

    for _ in range(10):
        if state["failed"]:
            break
        await asyncio.sleep(0.05)

    stable_file = workspace / "stable.txt"
    stable_file.write_text("stable", encoding="utf-8")

    for _ in range(20):
        if any(event.get("path") == "/workspace/stable.txt" for event in events):
            break
        await asyncio.sleep(0.05)

    await service.close()

    assert state["failed"] is True
    assert any(
        event.get("path") == "/workspace/stable.txt"
        and event.get("change_type") == "created"
        for event in events
    )
