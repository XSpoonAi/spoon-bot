"""Task-local execution context for tool ownership scoping."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_TOOL_OWNER: ContextVar[str] = ContextVar("tool_owner", default="default")


def normalize_tool_owner_user(user_id: str | None) -> str:
    """Normalize user identity component for tool ownership."""
    if isinstance(user_id, str) and user_id.strip():
        return user_id.strip()
    return "anonymous"


def normalize_tool_owner_session(session_key: str | None) -> str:
    """Normalize session component for tool ownership."""
    if isinstance(session_key, str) and session_key.strip():
        return session_key.strip()
    return "default"


def build_tool_owner_key(user_id: str | None, session_key: str | None) -> str:
    """Build a stable user+session ownership key."""
    return (
        f"user:{normalize_tool_owner_user(user_id)}"
        f"|session:{normalize_tool_owner_session(session_key)}"
    )


def get_tool_owner() -> str:
    """Return current task-local tool owner key."""
    owner = _TOOL_OWNER.get()
    if isinstance(owner, str) and owner.strip():
        return owner.strip()
    return "default"


@contextmanager
def bind_tool_owner(owner: str | None) -> Iterator[str]:
    """Bind tool owner for the current task context."""
    normalized = owner.strip() if isinstance(owner, str) and owner.strip() else "default"
    token = _TOOL_OWNER.set(normalized)
    try:
        yield normalized
    finally:
        _TOOL_OWNER.reset(token)
