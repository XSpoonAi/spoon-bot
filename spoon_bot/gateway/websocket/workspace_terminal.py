"""Workspace terminal service for websocket RPC methods."""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal
from uuid import uuid4


SandboxStdoutCallback = Callable[[dict[str, Any]], Awaitable[None]]
TermClosedCallback = Callable[[dict[str, Any]], Awaitable[None]]
TransportKind = Literal["pty", "pipe"]
SANDBOX_WORKSPACE_ROOT = "/workspace"


@dataclass(slots=True)
class TerminalSession:
    """Connection-scoped terminal session."""

    term_id: str
    process: subprocess.Popen[bytes]
    cwd: Path
    shell: str
    cols: int
    rows: int
    env: dict[str, str]
    transport: TransportKind
    master_fd: int | None = None
    close_reason: str = "process_exit"
    closed: bool = False


class WorkspaceTerminalService:
    """Implement terminal RPCs against the runtime workspace."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        emit_stdout: SandboxStdoutCallback,
        emit_closed: TermClosedCallback,
        sandbox_id: str = "runtime",
    ) -> None:
        self._workspace_root = workspace_root.resolve()
        self._emit_stdout = emit_stdout
        self._emit_closed = emit_closed
        self._sandbox_id = sandbox_id.strip() or "runtime"
        self._sessions: dict[str, TerminalSession] = {}
        self._lock = asyncio.Lock()

    async def open(
        self,
        *,
        cwd: str | None = None,
        shell: str | None = None,
        cols: int = 120,
        rows: int = 32,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        resolved_cwd = self._resolve_cwd(cwd)
        resolved_shell = self._resolve_shell(shell)
        merged_env = self._build_env(env)
        term_id = f"term_{uuid4().hex}"
        session = await asyncio.to_thread(
            self._spawn_session,
            term_id,
            resolved_cwd,
            resolved_shell,
            cols,
            rows,
            merged_env,
        )

        loop = asyncio.get_running_loop()
        async with self._lock:
            self._sessions[term_id] = session

        threading.Thread(
            target=self._pump_output,
            args=(loop, session),
            daemon=True,
            name=f"term-output-{term_id}",
        ).start()
        threading.Thread(
            target=self._wait_for_exit,
            args=(loop, session),
            daemon=True,
            name=f"term-exit-{term_id}",
        ).start()
        return {"term_id": term_id}

    async def input(self, term_id: str, text: str) -> dict[str, Any]:
        session = await self._get_session(term_id)
        payload = text.encode()
        if session.transport == "pty":
            assert session.master_fd is not None
            await asyncio.to_thread(os.write, session.master_fd, payload)
        else:
            stdin = session.process.stdin
            if stdin is None:
                raise ValueError(f"Terminal stdin is not available: {term_id}")
            await asyncio.to_thread(self._write_pipe, stdin, payload)
        return {"accepted": True}

    async def resize(self, term_id: str, *, cols: int, rows: int) -> dict[str, Any]:
        session = await self._get_session(term_id)
        session.cols = self._validate_dimension(cols, "cols")
        session.rows = self._validate_dimension(rows, "rows")

        if session.transport == "pty" and session.master_fd is not None:
            await asyncio.to_thread(_set_pty_winsize, session.master_fd, session.rows, session.cols)
        return {"resized": True}

    async def close(self, term_id: str, *, reason: str = "user_closed") -> dict[str, Any]:
        session = await self._get_session(term_id)
        session.close_reason = reason
        await asyncio.to_thread(self._terminate_process, session.process)
        return {"closed": True}

    async def shutdown(self, *, reason: str = "connection_closed") -> None:
        async with self._lock:
            sessions = list(self._sessions.values())
        if not sessions:
            return
        await asyncio.gather(
            *(self.close(session.term_id, reason=reason) for session in sessions),
            return_exceptions=True,
        )

    async def _get_session(self, term_id: str) -> TerminalSession:
        normalized = str(term_id).strip()
        if not normalized:
            raise ValueError("term_id is required")
        async with self._lock:
            session = self._sessions.get(normalized)
        if session is None or session.closed:
            raise ValueError(f"Unknown terminal session: {term_id}")
        return session

    async def _finalize_session(self, term_id: str, return_code: int | None) -> None:
        async with self._lock:
            session = self._sessions.get(term_id)
            if session is None or session.closed:
                return
            session.closed = True
            self._sessions.pop(term_id, None)

        if session.master_fd is not None:
            try:
                os.close(session.master_fd)
            except OSError:
                pass

        payload = {
            "term_id": session.term_id,
            "reason": session.close_reason or "process_exit",
            "ts": _now_iso(),
        }
        if return_code is not None:
            payload["exit_code"] = return_code
        await self._emit_closed(payload)

    def _spawn_session(
        self,
        term_id: str,
        cwd: Path,
        shell: str,
        cols: int,
        rows: int,
        env: dict[str, str],
    ) -> TerminalSession:
        validated_cols = self._validate_dimension(cols, "cols")
        validated_rows = self._validate_dimension(rows, "rows")

        if os.name == "posix":
            master_fd, slave_fd = os.openpty()
            _set_pty_winsize(master_fd, validated_rows, validated_cols)
            try:
                process = subprocess.Popen(
                    _shell_argv(shell),
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    cwd=str(cwd),
                    env=env,
                    start_new_session=True,
                )
            finally:
                os.close(slave_fd)
            return TerminalSession(
                term_id=term_id,
                process=process,
                cwd=cwd,
                shell=shell,
                cols=validated_cols,
                rows=validated_rows,
                env=env,
                transport="pty",
                master_fd=master_fd,
            )

        process = subprocess.Popen(
            _shell_argv(shell),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            env=env,
        )
        return TerminalSession(
            term_id=term_id,
            process=process,
            cwd=cwd,
            shell=shell,
            cols=validated_cols,
            rows=validated_rows,
            env=env,
            transport="pipe",
        )

    def _pump_output(self, loop: asyncio.AbstractEventLoop, session: TerminalSession) -> None:
        try:
            if session.transport == "pty":
                self._pump_pty_output(loop, session)
            else:
                self._pump_pipe_output(loop, session)
        except Exception:
            return

    def _pump_pty_output(self, loop: asyncio.AbstractEventLoop, session: TerminalSession) -> None:
        assert session.master_fd is not None
        while True:
            try:
                chunk = os.read(session.master_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            self._schedule_stdout(loop, session, chunk)

    def _pump_pipe_output(self, loop: asyncio.AbstractEventLoop, session: TerminalSession) -> None:
        stdout = session.process.stdout
        if stdout is None:
            return
        while True:
            chunk = stdout.read(4096)
            if not chunk:
                break
            self._schedule_stdout(loop, session, chunk)

    def _wait_for_exit(self, loop: asyncio.AbstractEventLoop, session: TerminalSession) -> None:
        return_code = session.process.wait()
        asyncio.run_coroutine_threadsafe(
            self._finalize_session(session.term_id, return_code),
            loop,
        )

    def _schedule_stdout(
        self,
        loop: asyncio.AbstractEventLoop,
        session: TerminalSession,
        chunk: bytes,
    ) -> None:
        text = chunk.decode("utf-8", errors="replace")
        payload = {
            "sandbox_id": self._sandbox_id,
            "term_id": session.term_id,
            "stream": "stdout",
            "chunk": text,
            "ts": _now_iso(),
        }
        asyncio.run_coroutine_threadsafe(self._emit_stdout(payload), loop)

    def _resolve_cwd(self, cwd: str | None) -> Path:
        raw = str(cwd or SANDBOX_WORKSPACE_ROOT).strip()
        target = self._resolve_workspace_path(raw)
        if not target.exists():
            raise ValueError(f"cwd not found: {raw}")
        if not target.is_dir():
            raise ValueError(f"cwd is not a directory: {raw}")
        return target

    def _resolve_shell(self, shell: str | None) -> str:
        raw = str(shell or "").strip()
        if raw:
            return raw
        if os.name == "posix":
            return os.environ.get("SHELL", "/bin/bash")
        return os.environ.get("COMSPEC", "cmd.exe")

    def _build_env(self, env: dict[str, str] | None) -> dict[str, str]:
        merged = dict(os.environ)
        if env is None:
            return merged
        if not isinstance(env, dict):
            raise ValueError("env must be an object")
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("env must be a string-to-string map")
            merged[key] = value
        return merged

    def _resolve_workspace_path(self, requested_path: str) -> Path:
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

        try:
            target.relative_to(self._workspace_root)
        except ValueError as exc:
            raise ValueError("Path is outside workspace boundary") from exc
        return target

    @staticmethod
    def _validate_dimension(value: int, name: str) -> int:
        dimension = int(value)
        if dimension <= 0:
            raise ValueError(f"{name} must be > 0")
        return dimension

    @staticmethod
    def _write_pipe(stdin: Any, payload: bytes) -> None:
        stdin.write(payload)
        stdin.flush()

    @staticmethod
    def _terminate_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _shell_argv(shell: str) -> list[str]:
    return [shell.strip()]


def _set_pty_winsize(fd: int, rows: int, cols: int) -> None:
    if os.name != "posix":
        return
    import fcntl
    import struct
    import termios

    winsize = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
