"""Workspace filesystem service for websocket RPC methods."""

from __future__ import annotations

import asyncio
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SANDBOX_WORKSPACE_ROOT = "/workspace"


class WorkspaceFSService:
    """Implement sandbox filesystem RPCs against the runtime workspace."""

    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()

    async def list(
        self,
        path: str,
        *,
        cursor: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(self._list_sync, path, cursor=cursor, limit=limit)

    async def stat(self, path: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._stat_sync, path)

    async def read(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        offset: int = 0,
        limit: int = 262144,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._read_sync,
            path,
            encoding=encoding,
            offset=offset,
            limit=limit,
        )

    async def write(
        self,
        path: str,
        *,
        content: str,
        encoding: str = "utf-8",
        create: bool = True,
        truncate: bool = True,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._write_sync,
            path,
            content=content,
            encoding=encoding,
            create=create,
            truncate=truncate,
        )

    async def mkdir(self, path: str, *, recursive: bool = False) -> dict[str, Any]:
        return await asyncio.to_thread(self._mkdir_sync, path, recursive=recursive)

    async def rename(
        self,
        from_path: str,
        to_path: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._rename_sync,
            from_path,
            to_path,
            overwrite=overwrite,
        )

    async def remove(self, path: str, *, recursive: bool = False) -> dict[str, Any]:
        return await asyncio.to_thread(self._remove_sync, path, recursive=recursive)

    def _list_sync(self, path: str, *, cursor: str | None, limit: int) -> dict[str, Any]:
        directory, _ = self._resolve_existing_path(path, expect_directory=True)
        effective_limit = max(1, min(int(limit), 1000))
        entries = [self._build_entry(entry) for entry in directory.iterdir()]
        entries.sort(key=lambda item: (item["type"] != "dir", item["name"].lower()))

        start = 0
        if cursor:
            start = next(
                (index + 1 for index, entry in enumerate(entries) if entry["path"] == cursor),
                0,
            )

        page = entries[start:start + effective_limit]
        has_more = start + effective_limit < len(entries)
        next_cursor = page[-1]["path"] if has_more and page else None
        return {
            "entries": page,
            "has_more": has_more,
            **({"next_cursor": next_cursor} if next_cursor else {}),
        }

    def _stat_sync(self, path: str) -> dict[str, Any]:
        target, sandbox_path = self._resolve_existing_path(path)
        return self._build_stat_payload(target, sandbox_path=sandbox_path)

    def _read_sync(
        self,
        path: str,
        *,
        encoding: str,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        target, sandbox_path = self._resolve_existing_path(path)
        if not target.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if offset < 0:
            raise ValueError("offset must be >= 0")
        if limit <= 0:
            raise ValueError("limit must be > 0")

        raw = target.read_bytes()
        sliced = raw[offset:offset + limit]
        try:
            content = sliced.decode(encoding)
        except LookupError as exc:
            raise ValueError(f"Unknown encoding: {encoding}") from exc
        except UnicodeDecodeError as exc:
            raise ValueError(f"Failed to decode file with encoding {encoding}") from exc

        stat = target.stat()
        return {
            "path": sandbox_path,
            "content": content,
            "encoding": encoding,
            "size": stat.st_size,
            "mtime": _format_mtime(stat.st_mtime),
        }

    def _write_sync(
        self,
        path: str,
        *,
        content: str,
        encoding: str,
        create: bool,
        truncate: bool,
    ) -> dict[str, Any]:
        target, sandbox_path = self._resolve_write_path(path)
        exists = target.exists()

        if exists and target.is_dir():
            raise ValueError(f"Path is a directory: {path}")
        if not exists and not create:
            raise ValueError(f"Path does not exist: {path}")

        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if truncate or not exists else "a"
        try:
            with target.open(mode, encoding=encoding) as handle:
                handle.write(content)
        except LookupError as exc:
            raise ValueError(f"Unknown encoding: {encoding}") from exc

        stat = target.stat()
        return {
            "path": sandbox_path,
            "size": stat.st_size,
            "mtime": _format_mtime(stat.st_mtime),
        }

    def _mkdir_sync(self, path: str, *, recursive: bool) -> dict[str, Any]:
        target, sandbox_path = self._resolve_write_path(path)
        existed = target.exists()
        if existed and not target.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        if recursive:
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(exist_ok=existed)

        return {
            "path": sandbox_path,
            "created": not existed,
        }

    def _rename_sync(self, from_path: str, to_path: str, *, overwrite: bool) -> dict[str, Any]:
        source, source_sandbox = self._resolve_existing_path(from_path)
        target, target_sandbox = self._resolve_write_path(to_path)

        if target.exists():
            if not overwrite:
                raise ValueError(f"Target already exists: {to_path}")
            if target.is_dir() and any(target.iterdir()):
                raise ValueError(f"Target directory is not empty: {to_path}")
            if target.is_dir():
                target.rmdir()
            else:
                target.unlink()

        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        return {
            "from_path": source_sandbox,
            "to_path": target_sandbox,
            "moved": True,
        }

    def _remove_sync(self, path: str, *, recursive: bool) -> dict[str, Any]:
        target, sandbox_path = self._resolve_existing_path(path)
        if target.is_dir():
            if recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()

        return {
            "path": sandbox_path,
            "removed": True,
        }

    def _build_entry(self, path: Path) -> dict[str, Any]:
        sandbox_path = self._sandbox_path(path)
        stat = path.lstat()
        entry_type = _entry_type(path)
        size = stat.st_size if entry_type != "dir" else 0
        return {
            "name": path.name,
            "path": sandbox_path,
            "type": entry_type,
            "size": size,
            "mtime": _format_mtime(stat.st_mtime),
        }

    def _build_stat_payload(self, path: Path, *, sandbox_path: str) -> dict[str, Any]:
        stat = path.lstat()
        entry_type = _entry_type(path)
        return {
            "path": sandbox_path,
            "type": entry_type,
            "size": stat.st_size if entry_type != "dir" else 0,
            "mtime": _format_mtime(stat.st_mtime),
            "mode": oct(stat.st_mode & 0o777),
        }

    def _resolve_existing_path(
        self,
        requested_path: str,
        *,
        expect_directory: bool | None = None,
    ) -> tuple[Path, str]:
        target, sandbox_path = self._resolve_path(requested_path)
        if not target.exists():
            raise ValueError(f"Path not found: {requested_path}")
        if expect_directory is True and not target.is_dir():
            raise ValueError(f"Path is not a directory: {requested_path}")
        if expect_directory is False and not target.is_file():
            raise ValueError(f"Path is not a file: {requested_path}")
        return target, sandbox_path

    def _resolve_write_path(self, requested_path: str) -> tuple[Path, str]:
        target, sandbox_path = self._resolve_path(requested_path)
        parent = target.parent
        if not self._is_within_workspace(parent):
            raise ValueError("Path is outside workspace boundary")
        return target, sandbox_path

    def _resolve_path(self, requested_path: str) -> tuple[Path, str]:
        raw_path = str(requested_path or "").strip()
        if raw_path == "" or raw_path == SANDBOX_WORKSPACE_ROOT:
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
            target = (self._workspace_root / relative).resolve(strict=False)
        else:
            target = (self._workspace_root / raw_path).resolve(strict=False)

        if not self._is_within_workspace(target):
            raise ValueError("Path is outside workspace boundary")
        return target, self._sandbox_path(target)

    def _sandbox_path(self, target: Path) -> str:
        relative = target.relative_to(self._workspace_root)
        relative_str = relative.as_posix()
        if relative_str in ("", "."):
            return SANDBOX_WORKSPACE_ROOT
        return f"{SANDBOX_WORKSPACE_ROOT}/{relative_str}"

    def _is_within_workspace(self, target: Path) -> bool:
        try:
            target.relative_to(self._workspace_root)
            return True
        except ValueError:
            return False


def _entry_type(path: Path) -> str:
    if path.is_symlink():
        return "link"
    if path.is_dir():
        return "dir"
    return "file"


def _format_mtime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")
