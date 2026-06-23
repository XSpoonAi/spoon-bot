from __future__ import annotations

from pathlib import Path

import pytest

from spoon_bot.agent.tools.shell import ShellTool
from spoon_bot.gateway.api.v1.workspace import _build_tree
from spoon_bot.gateway.websocket.workspace_fs import WorkspaceFSService
from spoon_bot.gateway.websocket.workspace_watch import WorkspaceWatchService


def _make_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "state.env").write_text("APP_DIR=/old\n", encoding="utf-8")


def test_workspace_tree_hides_wallet_runtime_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / ".env").write_text("VISIBLE=true\n", encoding="utf-8")
    _make_dir(workspace / ".agent-wallet")
    _make_dir(workspace / ".home-.agent-wallet" / ".agent-wallet")
    _make_dir(workspace / "runtime-wallet")
    _make_dir(workspace / ".home-runtime-wallet" / ".agent-wallet")
    monkeypatch.setenv("SPOON_BOT_WALLET_PATH", str(workspace / "runtime-wallet"))

    nodes = _build_tree(
        workspace,
        include_hidden=True,
        workspace_root=workspace,
    )

    names = {node.name for node in nodes}
    assert {"src", ".env"} <= names
    assert ".agent-wallet" not in names
    assert ".home-.agent-wallet" not in names
    assert "runtime-wallet" not in names
    assert ".home-runtime-wallet" not in names


@pytest.mark.asyncio
async def test_workspace_fs_list_hides_wallet_runtime_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("ok", encoding="utf-8")
    (workspace / ".env").write_text("VISIBLE=true\n", encoding="utf-8")
    _make_dir(workspace / ".agent-wallet")
    _make_dir(workspace / ".home-.agent-wallet" / ".agent-wallet")
    _make_dir(workspace / "runtime-wallet")
    monkeypatch.setenv("SPOON_BOT_WALLET_PATH", str(workspace / "runtime-wallet"))

    service = WorkspaceFSService(workspace)
    listing = await service.list("/workspace")

    names = {entry["name"] for entry in listing["entries"]}
    assert {"notes.txt", ".env"} <= names
    assert ".agent-wallet" not in names
    assert ".home-.agent-wallet" not in names
    assert "runtime-wallet" not in names

    hidden_listing = await service.list("/workspace/.home-.agent-wallet")
    assert hidden_listing == {"entries": [], "has_more": False}


def test_workspace_watch_snapshot_hides_wallet_runtime_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "visible.txt").write_text("ok", encoding="utf-8")
    _make_dir(workspace / ".agent-wallet")
    _make_dir(workspace / ".home-.agent-wallet" / ".agent-wallet")
    _make_dir(workspace / "runtime-wallet")
    monkeypatch.setenv("SPOON_BOT_WALLET_PATH", str(workspace / "runtime-wallet"))

    async def emit_change(_: dict[str, object]) -> None:
        return None

    service = WorkspaceWatchService(workspace, emit_change)
    snapshot = service._snapshot_target(workspace, recursive=True)

    assert "/workspace/visible.txt" in snapshot
    assert not any("agent-wallet" in path for path in snapshot)
    assert not any("runtime-wallet" in path for path in snapshot)


def test_shell_wallet_compat_home_stays_outside_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    wallet = workspace / ".agent-wallet"
    _make_dir(wallet)
    runtime_home_root = tmp_path / "runtime-home"
    monkeypatch.setenv("SPOON_BOT_WALLET_COMPAT_HOME_ROOT", str(runtime_home_root))

    env: dict[str, str] = {}
    tool = ShellTool(working_dir=str(workspace))
    tool._align_wallet_home_for_command(
        env,
        "cat ~/.agent-wallet/state.env",
        str(workspace),
    )

    compat_home = Path(env["HOME"])
    assert runtime_home_root in compat_home.parents
    assert workspace not in compat_home.parents
    assert env["SPOON_BOT_WALLET_PATH"] == str(compat_home / ".agent-wallet")
    assert env["AGENT_WALLET_DIR"] == str(compat_home / ".agent-wallet")
    assert not (workspace / ".home-.agent-wallet").exists()
