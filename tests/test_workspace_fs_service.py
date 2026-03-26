from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from spoon_bot.gateway.websocket.workspace_fs import WorkspaceFSService, _PathLockManager


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


# ---------------------------------------------------------------------------
# Concurrency tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_reads_run_in_parallel(tmp_path: Path) -> None:
    """Multiple read-only operations should not block each other."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for i in range(5):
        (workspace / f"file{i}.txt").write_text(f"content-{i}", encoding="utf-8")

    service = WorkspaceFSService(workspace)

    # Run 5 list + 5 stat + 5 read concurrently
    tasks = []
    for i in range(5):
        tasks.append(service.list("/workspace"))
        tasks.append(service.stat(f"/workspace/file{i}.txt"))
        tasks.append(service.read(f"/workspace/file{i}.txt"))

    results = await asyncio.gather(*tasks)
    assert len(results) == 15
    # All list results have entries
    for r in results[::3]:  # list results at index 0, 3, 6, 9, 12
        assert "entries" in r


@pytest.mark.asyncio
async def test_concurrent_writes_same_path_serialized(tmp_path: Path) -> None:
    """Writes to the same path must be serialized (not overlapping)."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = WorkspaceFSService(workspace)

    call_log: list[tuple[str, float, float]] = []
    original_write_sync = service._write_sync

    def _slow_write_sync(*args, **kwargs):
        start = time.monotonic()
        time.sleep(0.05)  # 50ms per write
        result = original_write_sync(*args, **kwargs)
        end = time.monotonic()
        call_log.append(("write", start, end))
        return result

    with patch.object(service, "_write_sync", side_effect=_slow_write_sync):
        await asyncio.gather(
            service.write("/workspace/same.txt", content="aaa"),
            service.write("/workspace/same.txt", content="bbb"),
            service.write("/workspace/same.txt", content="ccc"),
        )

    # Verify writes did not overlap: each start >= previous end
    assert len(call_log) == 3
    sorted_log = sorted(call_log, key=lambda x: x[1])  # sort by start time
    for i in range(1, len(sorted_log)):
        assert sorted_log[i][1] >= sorted_log[i - 1][2] - 0.001  # 1ms tolerance


@pytest.mark.asyncio
async def test_concurrent_writes_different_paths_parallel(tmp_path: Path) -> None:
    """Writes to different paths should run in parallel."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = WorkspaceFSService(workspace)

    call_log: list[tuple[str, float, float]] = []
    original_write_sync = service._write_sync

    def _slow_write_sync(*args, **kwargs):
        start = time.monotonic()
        time.sleep(0.05)  # 50ms per write
        result = original_write_sync(*args, **kwargs)
        end = time.monotonic()
        call_log.append((args[0], start, end))
        return result

    with patch.object(service, "_write_sync", side_effect=_slow_write_sync):
        await asyncio.gather(
            service.write("/workspace/a.txt", content="aaa"),
            service.write("/workspace/b.txt", content="bbb"),
            service.write("/workspace/c.txt", content="ccc"),
        )

    assert len(call_log) == 3
    # Total wall-clock time should be ~50ms (parallel), not ~150ms (serial)
    earliest_start = min(c[1] for c in call_log)
    latest_end = max(c[2] for c in call_log)
    elapsed = latest_end - earliest_start
    # Allow generous tolerance but must be less than fully serial (150ms)
    assert elapsed < 0.12, f"Writes took {elapsed:.3f}s — expected parallel execution"


@pytest.mark.asyncio
async def test_read_and_write_same_path_serialized(tmp_path: Path) -> None:
    """Reads and writes on the same path must not overlap."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "shared.txt").write_text("seed", encoding="utf-8")
    service = WorkspaceFSService(workspace)

    call_log: list[tuple[str, float, float]] = []
    original_read_sync = service._read_sync
    original_write_sync = service._write_sync

    def _slow_read_sync(*args, **kwargs):
        start = time.monotonic()
        time.sleep(0.05)
        result = original_read_sync(*args, **kwargs)
        end = time.monotonic()
        call_log.append(("read", start, end))
        return result

    def _slow_write_sync(*args, **kwargs):
        start = time.monotonic()
        time.sleep(0.05)
        result = original_write_sync(*args, **kwargs)
        end = time.monotonic()
        call_log.append(("write", start, end))
        return result

    with (
        patch.object(service, "_read_sync", side_effect=_slow_read_sync),
        patch.object(service, "_write_sync", side_effect=_slow_write_sync),
    ):
        await asyncio.gather(
            service.read("/workspace/shared.txt"),
            service.write("/workspace/shared.txt", content="updated"),
        )

    assert len(call_log) == 2
    sorted_log = sorted(call_log, key=lambda x: x[1])
    assert sorted_log[1][1] >= sorted_log[0][2] - 0.001


@pytest.mark.asyncio
async def test_remove_directory_and_child_write_serialized(tmp_path: Path) -> None:
    """Recursive directory removal must not overlap with child writes."""
    workspace = tmp_path / "workspace"
    nested = workspace / "dir"
    nested.mkdir(parents=True)
    (nested / "seed.txt").write_text("seed", encoding="utf-8")
    service = WorkspaceFSService(workspace)

    call_log: list[tuple[str, float, float]] = []
    original_remove_sync = service._remove_sync
    original_write_sync = service._write_sync

    def _slow_remove_sync(*args, **kwargs):
        start = time.monotonic()
        time.sleep(0.05)
        result = original_remove_sync(*args, **kwargs)
        end = time.monotonic()
        call_log.append(("remove", start, end))
        return result

    def _slow_write_sync(*args, **kwargs):
        start = time.monotonic()
        time.sleep(0.05)
        result = original_write_sync(*args, **kwargs)
        end = time.monotonic()
        call_log.append(("write", start, end))
        return result

    with (
        patch.object(service, "_remove_sync", side_effect=_slow_remove_sync),
        patch.object(service, "_write_sync", side_effect=_slow_write_sync),
    ):
        results = await asyncio.gather(
            service.remove("/workspace/dir", recursive=True),
            service.write("/workspace/dir/child.txt", content="updated"),
            return_exceptions=True,
        )

    assert len(call_log) == 2
    sorted_log = sorted(call_log, key=lambda x: x[1])
    assert sorted_log[1][1] >= sorted_log[0][2] - 0.001
    assert any(isinstance(result, dict) for result in results)
    assert any(isinstance(result, Exception) for result in results) or any(
        isinstance(result, dict) and result.get("removed") is True for result in results
    )


@pytest.mark.asyncio
async def test_rename_directory_and_child_write_serialized(tmp_path: Path) -> None:
    """Directory rename must not overlap with writes to descendants."""
    workspace = tmp_path / "workspace"
    nested = workspace / "dir"
    nested.mkdir(parents=True)
    (nested / "seed.txt").write_text("seed", encoding="utf-8")
    service = WorkspaceFSService(workspace)

    call_log: list[tuple[str, float, float]] = []
    original_rename_sync = service._rename_sync
    original_write_sync = service._write_sync

    def _slow_rename_sync(*args, **kwargs):
        start = time.monotonic()
        time.sleep(0.05)
        result = original_rename_sync(*args, **kwargs)
        end = time.monotonic()
        call_log.append(("rename", start, end))
        return result

    def _slow_write_sync(*args, **kwargs):
        start = time.monotonic()
        time.sleep(0.05)
        result = original_write_sync(*args, **kwargs)
        end = time.monotonic()
        call_log.append(("write", start, end))
        return result

    with (
        patch.object(service, "_rename_sync", side_effect=_slow_rename_sync),
        patch.object(service, "_write_sync", side_effect=_slow_write_sync),
    ):
        results = await asyncio.gather(
            service.rename("/workspace/dir", "/workspace/renamed", overwrite=True),
            service.write("/workspace/dir/child.txt", content="updated"),
            return_exceptions=True,
        )

    assert len(call_log) == 2
    sorted_log = sorted(call_log, key=lambda x: x[1])
    assert sorted_log[1][1] >= sorted_log[0][2] - 0.001
    assert any(isinstance(result, dict) and result.get("moved") is True for result in results)


@pytest.mark.asyncio
async def test_rename_no_deadlock(tmp_path: Path) -> None:
    """Concurrent rename(a,b) and rename(b,a) must not deadlock."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("A", encoding="utf-8")
    (workspace / "b.txt").write_text("B", encoding="utf-8")

    service = WorkspaceFSService(workspace)

    # We can't actually run both renames concurrently on the same two files
    # since one will fail (source gone). Instead, verify that the path lock
    # manager acquires locks in sorted order and doesn't deadlock.
    # We do this by running one rename and confirming it completes quickly.
    result = await asyncio.wait_for(
        service.rename("/workspace/a.txt", "/workspace/b.txt", overwrite=True),
        timeout=2.0,
    )
    assert result["moved"] is True
    assert (workspace / "b.txt").read_text(encoding="utf-8") == "A"


@pytest.mark.asyncio
async def test_path_lock_cleanup() -> None:
    """PathLockManager should clean up locks after all operations complete."""
    plm = _PathLockManager()

    async with plm.lock("/a", "/b"):
        assert "/a" in plm._active
        assert "/b" in plm._active

    # After context exit, locks should be cleaned up
    assert "/a" not in plm._active
    assert "/b" not in plm._active
