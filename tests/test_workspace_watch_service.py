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
