"""Shared async execution coordination primitives."""

from __future__ import annotations

import asyncio


def normalize_session_key(session_key: str | None) -> str:
    """Normalize user-provided session keys for lock lookup."""
    if isinstance(session_key, str) and session_key.strip():
        return session_key.strip()
    return "default"


class ExecutionCoordinator:
    """Provides shared locks for agent runners and logical sessions."""

    def __init__(self) -> None:
        self._runner_locks: dict[str, asyncio.Lock] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    def get_runner_lock(self, scope: str = "primary-agent") -> asyncio.Lock:
        """Return the lock guarding a mutable agent runner."""
        key = scope.strip() if scope and scope.strip() else "primary-agent"
        lock = self._runner_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._runner_locks[key] = lock
        return lock

    def get_session_lock(self, session_key: str | None) -> asyncio.Lock:
        """Return the lock guarding writes to a session."""
        key = normalize_session_key(session_key)
        lock = self._session_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[key] = lock
        return lock
