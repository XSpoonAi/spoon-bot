"""Task-local execution context for tool ownership scoping and stream capture."""

from __future__ import annotations

import json
import re
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
_TOOL_INVOCATION_DEDUP_STATE: ContextVar[dict[str, Any] | None] = ContextVar(
    "tool_invocation_dedup_state",
    default=None,
)
_REQUEST_EXECUTION_HINTS: ContextVar[dict[str, Any] | None] = ContextVar(
    "request_execution_hints",
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
_DUPLICATE_TOOL_INVOCATION_MESSAGE = (
    "STOP_TOOL_LOOP: Error: duplicate tool invocation suppressed. The same tool and arguments "
    "already executed in this request; use the previous result or choose "
    "different arguments. Do not call more tools for this same action; report "
    "the previous result or blocker now."
)
_REPEATED_TOOL_SERIES_MESSAGE = (
    "STOP_TOOL_LOOP: Error: repeated side-effecting tool series suppressed. This request has "
    "already executed the same kind of external side effect; report the latest "
    "result or ask for an explicit count before continuing. Do not call more "
    "tools for this same action."
)
_REDUNDANT_FILE_READ_MESSAGE = (
    "READ_FILE_CACHE_HIT: requested file range already available in this "
    "request. Treat this read as complete and continue the user's remaining "
    "instructions without calling read_file again for the same path and range."
)
_CONSECUTIVE_TOOL_FAILURE_MESSAGE = (
    "STOP_TOOL_LOOP: Error: consecutive tool failures suppressed. The same tool "
    "has failed repeatedly in this request without producing progress; report "
    "the blocker now instead of trying more alternate commands."
)


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


def _tool_failure_signal(payload: Any) -> str | None:
    """Return a generic failure signal for loop guards, independent of paths/routes."""
    if isinstance(payload, dict):
        for key in ("success", "ok"):
            value = payload.get(key)
            if value is False:
                return f"{key}=false"
        for key in ("returncode", "return_code", "exit_code"):
            value = payload.get(key)
            if isinstance(value, int) and value != 0:
                return f"{key}=nonzero"
        status = payload.get("status")
        if isinstance(status, str) and status.strip().lower() in {"failed", "error"}:
            return "status=failed"

    text = stringify_tool_output(payload).lower()
    if not text.strip():
        return None
    if re.search(r"exit code:\s*[1-9]\d*", text):
        return "nonzero_exit"
    if text.lstrip().startswith("error:"):
        return "error_prefix"
    if "traceback" in text or "exception" in text:
        return "runtime_error"
    return None


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
def track_tool_invocations(
    *,
    max_repeats: int = 2,
    max_series_repeats: int = 2,
    max_consecutive_failures: int = 3,
) -> Iterator[dict[str, Any]]:
    """Track exact tool invocations for the current request.

    This is a data-plane guard: it suppresses identical tool+argument executions
    inside one request without routing based on user prompt wording.
    """
    state: dict[str, Any] = {
        "counts": defaultdict(int),
        "last_exact_key": None,
        "last_exact_repeats": 0,
        "series_counts": defaultdict(int),
        "consecutive_failures": [],
        "file_read_coverage": defaultdict(list),
        "max_repeats": max(1, int(max_repeats or 1)),
        "max_series_repeats": max(1, int(max_series_repeats or 1)),
        "max_consecutive_failures": max(1, int(max_consecutive_failures or 1)),
    }
    token = _TOOL_INVOCATION_DEDUP_STATE.set(state)
    try:
        yield state
    finally:
        _TOOL_INVOCATION_DEDUP_STATE.reset(token)


def suppress_repeated_tool_invocation(tool_name: str, arguments: Any) -> str | None:
    """Return a compact duplicate result when the same call already ran."""
    state = _TOOL_INVOCATION_DEDUP_STATE.get()
    if not isinstance(state, dict):
        return None

    counts = state.get("counts")
    if counts is None:
        return None

    key = f"{str(tool_name or '')}\x1f{normalize_tool_arguments(arguments)}"
    counts[key] += 1
    if state.get("last_exact_key") == key:
        repeat_count = int(state.get("last_exact_repeats") or 0) + 1
    else:
        repeat_count = 1
    state["last_exact_key"] = key
    state["last_exact_repeats"] = repeat_count

    max_repeats = int(state.get("max_repeats") or 1)
    if repeat_count <= max_repeats:
        return None
    return _DUPLICATE_TOOL_INVOCATION_MESSAGE


def suppress_repeated_tool_series(tool_name: str, series_key: str | None) -> str | None:
    """Return a compact result when one request repeats the same side-effect series."""
    state = _TOOL_INVOCATION_DEDUP_STATE.get()
    if not isinstance(state, dict):
        return None
    if not isinstance(series_key, str) or not series_key.strip():
        return None

    counts = state.get("series_counts")
    if counts is None:
        return None

    key = f"{str(tool_name or '')}\x1f{series_key.strip()}"
    counts[key] += 1
    max_repeats = int(state.get("max_series_repeats") or 1)
    if counts[key] <= max_repeats:
        return None
    return _REPEATED_TOOL_SERIES_MESSAGE


def suppress_redundant_file_read(
    path_key: str,
    *,
    offset: int | None,
    limit: int | None,
    total_lines: int,
    content_fingerprint: str | None = None,
) -> str | None:
    """Return a compact result when a request re-reads covered file content."""
    state = _TOOL_INVOCATION_DEDUP_STATE.get()
    if not isinstance(state, dict):
        return None

    coverage = state.get("file_read_coverage")
    if not isinstance(coverage, dict):
        return None

    normalized_path = str(path_key or "").strip()
    if not normalized_path:
        return None

    total = max(1, int(total_lines or 1))
    start = max(1, int(offset or 1))
    if limit is None:
        end = total
    else:
        end = min(total, start + max(1, int(limit or 1)) - 1)
    if end < start:
        end = start

    fingerprint = str(content_fingerprint or "")
    raw_ranges = coverage.setdefault(normalized_path, [])
    ranges: list[tuple[int, int, str]] = []
    for item in raw_ranges:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        existing_fingerprint = str(item[2]) if len(item) >= 3 else ""
        if fingerprint and existing_fingerprint and existing_fingerprint != fingerprint:
            continue
        ranges.append((int(item[0]), int(item[1]), existing_fingerprint))

    for existing_start, existing_end, _existing_fingerprint in ranges:
        if int(existing_start) <= start and end <= int(existing_end):
            return _REDUNDANT_FILE_READ_MESSAGE

    ranges.append((start, end, fingerprint))
    ranges.sort(key=lambda item: (int(item[0]), int(item[1])))
    merged: list[tuple[int, int, str]] = []
    for range_start, range_end, range_fingerprint in ranges:
        range_start = int(range_start)
        range_end = int(range_end)
        if (
            not merged
            or range_fingerprint != merged[-1][2]
            or range_start > merged[-1][1] + 1
        ):
            merged.append((range_start, range_end, range_fingerprint))
            continue
        merged[-1] = (
            merged[-1][0],
            max(merged[-1][1], range_end),
            merged[-1][2],
        )
    coverage[normalized_path] = merged
    return None


def _normalize_file_tracking_path(path_key: Any) -> str:
    """Normalize file tracking paths for request-local cache invalidation."""
    return str(path_key or "").strip().replace("\\", "/").lower()


def invalidate_file_read_tracking(*path_keys: Any) -> None:
    """Invalidate read coverage and exact read counts after a file changes."""
    state = _TOOL_INVOCATION_DEDUP_STATE.get()
    if not isinstance(state, dict):
        return

    candidates = {
        normalized
        for normalized in (_normalize_file_tracking_path(path_key) for path_key in path_keys)
        if normalized
    }
    if not candidates:
        return

    coverage = state.get("file_read_coverage")
    if isinstance(coverage, dict):
        for key in list(coverage.keys()):
            if _normalize_file_tracking_path(key) in candidates:
                coverage.pop(key, None)

    counts = state.get("counts")
    if counts is None:
        return
    for key in list(counts.keys()):
        if not isinstance(key, str):
            continue
        tool_name, separator, argument_key = key.partition("\x1f")
        if separator != "\x1f" or tool_name != "read_file":
            continue
        try:
            arguments = json.loads(argument_key)
        except Exception:
            continue
        if not isinstance(arguments, dict):
            continue
        if _normalize_file_tracking_path(arguments.get("path")) in candidates:
            counts.pop(key, None)


def suppress_after_consecutive_tool_failures(tool_name: str) -> str | None:
    """Return a compact result when one tool keeps failing without progress."""
    state = _TOOL_INVOCATION_DEDUP_STATE.get()
    if not isinstance(state, dict):
        return None

    failures = state.get("consecutive_failures")
    if not isinstance(failures, list):
        return None
    max_failures = int(state.get("max_consecutive_failures") or 3)
    if len(failures) < max_failures:
        return None

    recent = failures[-max_failures:]
    current_name = str(tool_name or "")
    if all(item.get("tool") == current_name for item in recent if isinstance(item, dict)):
        return _CONSECUTIVE_TOOL_FAILURE_MESSAGE
    return None


def record_tool_invocation_result(tool_name: str, result: Any) -> None:
    """Record generic success/failure outcome for per-request failure-loop guards."""
    state = _TOOL_INVOCATION_DEDUP_STATE.get()
    if not isinstance(state, dict):
        return

    failures = state.get("consecutive_failures")
    if not isinstance(failures, list):
        return

    signal = _tool_failure_signal(result)
    if signal is None:
        failures.clear()
        return

    failures.append({"tool": str(tool_name or ""), "signal": signal})
    max_failures = int(state.get("max_consecutive_failures") or 3)
    del failures[: -max(1, max_failures)]


def get_tracked_tool_invocation_counts() -> dict[str, int]:
    """Return per-tool execution counts for the current request."""
    state = _TOOL_INVOCATION_DEDUP_STATE.get()
    if not isinstance(state, dict):
        return {}

    counts = state.get("counts")
    if counts is None:
        return {}

    per_tool: dict[str, int] = defaultdict(int)
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        tool_name = key.split("\x1f", 1)[0]
        if tool_name:
            per_tool[tool_name] += int(value or 0)
    return dict(per_tool)


@contextmanager
def bind_request_execution_hints(hints: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
    """Bind request-scoped execution hints for tool-side guardrails."""
    normalized = hints if isinstance(hints, dict) else {}
    token = _REQUEST_EXECUTION_HINTS.set(normalized)
    try:
        yield normalized
    finally:
        _REQUEST_EXECUTION_HINTS.reset(token)


def get_request_execution_hints() -> dict[str, Any]:
    """Return request-scoped execution hints for the current task."""
    hints = _REQUEST_EXECUTION_HINTS.get()
    if isinstance(hints, dict):
        return hints
    return {}


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
