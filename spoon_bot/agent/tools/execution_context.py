"""Task-local execution context for tool ownership scoping."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_TOOL_OWNER: ContextVar[str] = ContextVar("tool_owner", default="default")


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
