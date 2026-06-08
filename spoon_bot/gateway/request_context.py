"""Helpers for binding request metadata to a session-scoped agent runtime."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


def _clean_optional(value: Any) -> str:
    text = str(value or "").strip()
    return text


@contextmanager
def bind_agent_request_context(
    agent: Any,
    *,
    user_id: Any = None,
    session_key: Any = None,
    request_id: Any = None,
    trace_id: Any = None,
    task_id: Any = None,
    transport: str = "",
    connection_id: Any = None,
) -> Iterator[dict[str, str]]:
    """Attach request metadata to an agent only for the current call.

    Session runtimes are reused across requests, so the previous values are
    restored when the current request exits.
    """
    if agent is None:
        yield {}
        return

    previous_context = getattr(agent, "_current_request_context", None)
    previous_user_id = getattr(agent, "user_id", None)

    context = {
        "user_id": _clean_optional(user_id) or _clean_optional(previous_user_id),
        "session_key": _clean_optional(session_key),
        "request_id": _clean_optional(request_id),
        "trace_id": _clean_optional(trace_id),
        "task_id": _clean_optional(task_id),
        "transport": _clean_optional(transport),
        "connection_id": _clean_optional(connection_id),
    }
    context = {key: value for key, value in context.items() if value}

    try:
        if context.get("user_id"):
            setattr(agent, "user_id", context["user_id"])
        setattr(agent, "_current_request_context", context)
        yield context
    finally:
        if previous_user_id is not None:
            setattr(agent, "user_id", previous_user_id)
        elif hasattr(agent, "user_id"):
            try:
                delattr(agent, "user_id")
            except Exception:
                pass
        if previous_context is not None:
            setattr(agent, "_current_request_context", previous_context)
        elif hasattr(agent, "_current_request_context"):
            try:
                delattr(agent, "_current_request_context")
            except Exception:
                pass
