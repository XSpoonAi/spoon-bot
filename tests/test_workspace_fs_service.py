from __future__ import annotations

from pathlib import Path

import pytest

from spoon_bot.gateway.websocket.workspace_fs import WorkspaceFSService


@pytest.mark.asyncio
async def test_workspace_fs_service_list_stat_and_read(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nested = workspace / "docs"
    nested.mkdir()
    file_path = nested / "hello.txt"
    file_path.write_text("hello world", encoding="utf-8")

    service = WorkspaceFSService(workspace)

    listing = await service.list("/workspace/docs")
    assert listing["has_more"] is False
    assert listing["entries"] == [
        {
            "name": "hello.txt",
            "path": "/workspace/docs/hello.txt",
            "type": "file",
            "size": 11,
            "mtime": listing["entries"][0]["mtime"],
        }
    ]

    stat = await service.stat("/workspace/docs/hello.txt")
    assert stat["path"] == "/workspace/docs/hello.txt"
    assert stat["type"] == "file"
    assert stat["size"] == 11
    assert isinstance(stat["mode"], str)

    read = await service.read("/workspace/docs/hello.txt", offset=6, limit=5)
    assert read == {
        "path": "/workspace/docs/hello.txt",
        "content": "world",
        "encoding": "utf-8",
        "size": 11,
        "mtime": read["mtime"],
    }


@pytest.mark.asyncio
async def test_workspace_fs_service_write_mkdir_rename_and_remove(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = WorkspaceFSService(workspace)

    mkdir_result = await service.mkdir("/workspace/src/components", recursive=True)
    assert mkdir_result == {"path": "/workspace/src/components", "created": True}

    write_result = await service.write(
        "/workspace/src/components/app.ts",
        content="export const value = 1;\n",
    )
    assert write_result["path"] == "/workspace/src/components/app.ts"
    assert write_result["size"] == len("export const value = 1;\n".encode("utf-8"))

    rename_result = await service.rename(
        "/workspace/src/components/app.ts",
        "/workspace/src/components/page.ts",
    )
    assert rename_result == {
        "from_path": "/workspace/src/components/app.ts",
        "to_path": "/workspace/src/components/page.ts",
        "moved": True,
    }

    remove_result = await service.remove("/workspace/src", recursive=True)
    assert remove_result == {"path": "/workspace/src", "removed": True}
    assert not (workspace / "src").exists()


@pytest.mark.asyncio
async def test_workspace_fs_service_rejects_paths_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = WorkspaceFSService(workspace)

    with pytest.raises(ValueError, match="outside workspace boundary"):
        await service.stat("/etc/passwd")
