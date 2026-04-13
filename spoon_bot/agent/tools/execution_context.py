"""Task-local execution context for tool ownership scoping and stream capture."""

from __future__ import annotations

import json
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4


_TOOL_OWNER: ContextVar[str] = ContextVar("tool_owner", default="default")
_TOOL_OUTPUT_CAPTURE_SCOPE: ContextVar[str | None] = ContextVar(
    "tool_output_capture_scope",
    default=None,
)
_CURRENT_TOOL_INVOCATION: ContextVar[str | None] = ContextVar(
    "current_tool_invocation",
    default=None,
)


@dataclass
class CapturedToolOutput:
    """Full tool output captured for websocket streaming."""

    scope_id: str
    owner: str
    tool_name: str
    arguments_key: str
    summary_output: str = ""
    full_output: str = ""


_TOOL_OUTPUT_LOCK = Lock()
_ACTIVE_TOOL_INVOCATIONS: dict[str, CapturedToolOutput] = {}
_COMPLETED_TOOL_OUTPUTS: dict[str, list[CapturedToolOutput]] = defaultdict(list)
_MAX_CAPTURED_TOOL_OUTPUTS_PER_SCOPE = 256


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


def stringify_tool_output(payload: Any) -> str:
    """Serialize tool output to a plain string for websocket metadata."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (dict, list)):
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(payload)
    return str(payload)


def normalize_tool_arguments(arguments: Any) -> str:
    """Normalize tool arguments so captured output can be matched later."""
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
        except Exception:
            return text
        return normalize_tool_arguments(parsed)
    if isinstance(arguments, (dict, list)):
        try:
            return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(arguments)
    return str(arguments)


@contextmanager
def capture_tool_outputs() -> Iterator[str]:
    """Enable per-request tool output capture for websocket streaming."""
    scope_id = uuid4().hex
    token = _TOOL_OUTPUT_CAPTURE_SCOPE.set(scope_id)
    try:
        yield scope_id
    finally:
        _TOOL_OUTPUT_CAPTURE_SCOPE.reset(token)
        clear_captured_tool_outputs(scope_id)


@contextmanager
def bind_tool_invocation(tool_name: str, arguments: Any) -> Iterator[str | None]:
    """Bind the current tool invocation so tools can publish full output."""
    scope_id = _TOOL_OUTPUT_CAPTURE_SCOPE.get()
    if not scope_id:
        yield None
        return

    invocation_id = uuid4().hex
    capture = CapturedToolOutput(
        scope_id=scope_id,
        owner=get_tool_owner(),
        tool_name=str(tool_name or ""),
        arguments_key=normalize_tool_arguments(arguments),
    )
    with _TOOL_OUTPUT_LOCK:
        _ACTIVE_TOOL_INVOCATIONS[invocation_id] = capture
    token = _CURRENT_TOOL_INVOCATION.set(invocation_id)
    try:
        yield invocation_id
    finally:
        _CURRENT_TOOL_INVOCATION.reset(token)


def capture_tool_output(summary_output: Any, full_output: Any | None = None) -> None:
    """Record summary/full tool output for the current invocation."""
    invocation_id = _CURRENT_TOOL_INVOCATION.get()
    if not invocation_id:
        return

    summary_text = stringify_tool_output(summary_output)
    full_text = stringify_tool_output(full_output if full_output is not None else summary_output)

    with _TOOL_OUTPUT_LOCK:
        capture = _ACTIVE_TOOL_INVOCATIONS.get(invocation_id)
        if capture is None:
            return
        capture.summary_output = summary_text
        capture.full_output = full_text


def finalize_tool_invocation(default_output: Any | None = None) -> None:
    """Move the current invocation capture into the completed queue."""
    invocation_id = _CURRENT_TOOL_INVOCATION.get()
    if not invocation_id:
        return

    with _TOOL_OUTPUT_LOCK:
        capture = _ACTIVE_TOOL_INVOCATIONS.pop(invocation_id, None)
        if capture is None:
            return
        if not capture.summary_output:
            capture.summary_output = stringify_tool_output(default_output)
        if not capture.full_output:
            capture.full_output = capture.summary_output

        bucket = _COMPLETED_TOOL_OUTPUTS[capture.scope_id]
        bucket.append(capture)
        if len(bucket) > _MAX_CAPTURED_TOOL_OUTPUTS_PER_SCOPE:
            del bucket[:-_MAX_CAPTURED_TOOL_OUTPUTS_PER_SCOPE]


def consume_captured_tool_output(
    scope_id: str | None,
    *,
    tool_name: str | None = None,
    arguments: Any = None,
) -> CapturedToolOutput | None:
    """Consume the next captured tool output for a streamed tool_result event."""
    if not scope_id:
        return None

    arguments_key = normalize_tool_arguments(arguments)
    with _TOOL_OUTPUT_LOCK:
        bucket = _COMPLETED_TOOL_OUTPUTS.get(scope_id)
        if not bucket:
            return None

        match_index: int | None = None
        if tool_name and arguments_key:
            for index, item in enumerate(bucket):
                if item.tool_name == tool_name and item.arguments_key == arguments_key:
                    match_index = index
                    break
        if match_index is None and tool_name:
            for index, item in enumerate(bucket):
                if item.tool_name == tool_name:
                    match_index = index
                    break
        if match_index is None:
            match_index = 0

        capture = bucket.pop(match_index)
        if not bucket:
            _COMPLETED_TOOL_OUTPUTS.pop(scope_id, None)
        return capture


def clear_captured_tool_outputs(scope_id: str | None) -> None:
    """Drop any captured tool outputs for a completed/aborted request."""
    if not scope_id:
        return
    with _TOOL_OUTPUT_LOCK:
        _COMPLETED_TOOL_OUTPUTS.pop(scope_id, None)
        stale_invocations = [
            invocation_id
            for invocation_id, capture in _ACTIVE_TOOL_INVOCATIONS.items()
            if capture.scope_id == scope_id
        ]
        for invocation_id in stale_invocations:
            _ACTIVE_TOOL_INVOCATIONS.pop(invocation_id, None)
