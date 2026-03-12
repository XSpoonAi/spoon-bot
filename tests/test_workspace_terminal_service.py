from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

from spoon_bot.gateway.websocket.workspace_terminal import WorkspaceTerminalService


def _test_shell() -> tuple[str, str]:
    if os.name == "nt":
        return os.environ.get("COMSPEC", "cmd.exe"), "echo hello\r\nexit\r\n"
    return os.environ.get("SHELL", "/bin/sh"), "echo hello\nexit\n"


@pytest.mark.asyncio
async def test_workspace_terminal_service_open_input_and_close() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir) / "workspace"
        workspace.mkdir()

        stdout_events: list[dict[str, object]] = []
        closed_events: list[dict[str, object]] = []

        async def emit_stdout(payload: dict[str, object]) -> None:
            stdout_events.append(payload)

        async def emit_closed(payload: dict[str, object]) -> None:
            closed_events.append(payload)

        service = WorkspaceTerminalService(
            workspace,
            emit_stdout=emit_stdout,
            emit_closed=emit_closed,
            sandbox_id="sbx_test",
        )

        shell, command = _test_shell()
        opened = await service.open(cwd="/workspace", shell=shell, cols=80, rows=24)
        term_id = opened["term_id"]

        await service.input(term_id, command)

        for _ in range(60):
            if closed_events:
                break
            await asyncio.sleep(0.1)

        await service.shutdown()

        assert any("hello" in str(event.get("chunk", "")).lower() for event in stdout_events)
        assert any(event.get("term_id") == term_id for event in closed_events)
        assert all(event.get("sandbox_id") == "sbx_test" for event in stdout_events)


@pytest.mark.asyncio
async def test_workspace_terminal_service_resize_and_manual_close() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir) / "workspace"
        workspace.mkdir()

        closed_events: list[dict[str, object]] = []

        async def emit_stdout(payload: dict[str, object]) -> None:
            return None

        async def emit_closed(payload: dict[str, object]) -> None:
            closed_events.append(payload)

        service = WorkspaceTerminalService(
            workspace,
            emit_stdout=emit_stdout,
            emit_closed=emit_closed,
        )

        shell, _ = _test_shell()
        opened = await service.open(shell=shell)
        term_id = opened["term_id"]

        resized = await service.resize(term_id, cols=100, rows=40)
        assert resized == {"resized": True}

        closed = await service.close(term_id)
        assert closed == {"closed": True}

        for _ in range(40):
            if closed_events:
                break
            await asyncio.sleep(0.1)

        assert any(event.get("term_id") == term_id for event in closed_events)
        assert any(event.get("reason") == "user_closed" for event in closed_events)


@pytest.mark.asyncio
async def test_workspace_terminal_service_rejects_cwd_outside_workspace() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir) / "workspace"
        workspace.mkdir()

        async def emit_stdout(payload: dict[str, object]) -> None:
            return None

        async def emit_closed(payload: dict[str, object]) -> None:
            return None

        service = WorkspaceTerminalService(
            workspace,
            emit_stdout=emit_stdout,
            emit_closed=emit_closed,
        )

        shell, _ = _test_shell()
        with pytest.raises(ValueError, match="outside workspace boundary"):
            await service.open(cwd="/etc", shell=shell)
