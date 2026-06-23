"""Same-session transcript compaction helpers.

This module mirrors Claude Code's post-compact shape at a smaller scale:
preserve a compact, ordered view of the active session transcript without
putting prompt/domain routing rules in the agent loop.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from spoon_bot.agent.tools.execution_context import (
    classify_tool_invocation_category,
    sanitize_tool_arguments_for_history,
)

DEFAULT_MAX_SESSION_MESSAGES = 80
DEFAULT_MAX_SESSION_TURNS = 6
DEFAULT_CONTEXT_CHAR_BUDGET = 24_000
EVENT_CLIP_LEVELS = (360, 220, 140)

_OBSERVED_OUTPUT_RE = re.compile(
    r"^(?:\[[^\]\n]{1,32}\]:\s*)?Observed output of cmd [^\n]* execution:\s*",
    re.IGNORECASE,
)
_FILE_OUTPUT_HEADER_RE = re.compile(r"^\[file:\s*([^\]\n]+)\]\s*\n")
_PRELOADED_SKILL_BLOCK_RE = re.compile(
    r"\n+---\n\[PRE-LOADED SKILL:[\s\S]*$",
    re.IGNORECASE,
)
_BACKGROUND_JOB_ID_LINE_RE = re.compile(
    r"(?im)^job_id:\s*\S+\s*$"
)
_BACKGROUND_JOB_MONITOR_BLOCK_RE = re.compile(
    r"(?is)\n*NEXT STEPS\s*[—-]\s*monitor this job:.*$"
)


def _strip_runtime_injected_blocks(text: str) -> str:
    """Remove runtime-injected helper blocks before building evidence compact."""
    return _PRELOADED_SKILL_BLOCK_RE.sub("", text)


def _is_bounded_user_evidence(message: dict[str, Any]) -> bool:
    """Keep short user-stated facts, not long prior task instructions."""
    content = message.get("content", "")
    if not isinstance(content, str):
        return False
    text = _strip_runtime_injected_blocks(content).strip()
    try:
        from spoon_bot.utils.privacy import mask_secrets
        evidence_text = mask_secrets(text)
    except Exception:
        evidence_text = text
    if not evidence_text or len(evidence_text) > 80:
        return False
    non_empty_lines = [line for line in evidence_text.splitlines() if line.strip()]
    return len(non_empty_lines) <= 2


def _clip_text(text: Any, limit: int) -> str:
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except Exception:
            text = str(text)
    text = _strip_runtime_injected_blocks(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f" ...[truncated {len(text) - limit} chars]"


def _compact_multiline_text(text: Any, *, line_window: int = 4) -> str:
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except Exception:
            text = str(text)

    text = _OBSERVED_OUTPUT_RE.sub("", text.strip())
    if "command moved to background" in text.lower() or "job_id:" in text.lower():
        text = _BACKGROUND_JOB_MONITOR_BLOCK_RE.sub("", text)
        text = _BACKGROUND_JOB_ID_LINE_RE.sub(
            "job_id: [volatile runtime job id omitted; verify live state instead]",
            text,
        )
    file_header = _FILE_OUTPUT_HEADER_RE.match(text)
    if file_header:
        return (
            f"[file: {file_header.group(1)}] "
            "historical file reference only; read the current local file again for exact text"
        )
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= line_window * 2 + 2:
        return "\n".join(lines)
    omitted = len(lines) - (line_window * 2)
    return "\n".join(
        lines[:line_window]
        + [f"...[{omitted} line(s) omitted]"]
        + lines[-line_window:]
    )


def _compact_tool_call(tool_call: Any, *, limit: int) -> str:
    if not isinstance(tool_call, dict):
        return _clip_text(tool_call, limit)

    function = tool_call.get("function") or {}
    if isinstance(function, dict):
        name = str(function.get("name") or tool_call.get("name") or "tool")
        arguments = function.get("arguments") or tool_call.get("arguments") or ""
    else:
        name = str(getattr(function, "name", None) or tool_call.get("name") or "tool")
        arguments = getattr(function, "arguments", "") or tool_call.get("arguments") or ""
    call_id = tool_call.get("id") or tool_call.get("tool_call_id") or ""
    prefix = f"{name}"
    if call_id:
        prefix += f"#{call_id}"
    if arguments:
        return f"{prefix}({_clip_text(sanitize_tool_arguments_for_history(name, arguments), limit)})"
    return prefix


def _format_session_event(message: dict[str, Any], *, limit: int) -> str:
    role = str(message.get("role") or "").lower()
    content = message.get("content", "")

    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            calls = [
                _compact_tool_call(tool_call, limit=max(80, limit // 2))
                for tool_call in tool_calls[:5]
            ]
            if len(tool_calls) > 5:
                calls.append(f"...[{len(tool_calls) - 5} more tool call(s)]")
            return f"Assistant tool calls: {'; '.join(calls)}"
        excerpt = _clip_text(content, limit)
        return f"Assistant: {excerpt}" if excerpt else ""

    if role == "tool":
        name = str(message.get("name") or message.get("tool_name") or "tool")
        call_id = str(message.get("tool_call_id") or message.get("id") or "")
        excerpt = _clip_text(_compact_multiline_text(content), limit)
        if not excerpt:
            return ""
        suffix = f" #{call_id}" if call_id else ""
        return f"Tool `{name}`{suffix}: {excerpt}"

    if role == "user":
        excerpt = _clip_text(content, min(limit, 220))
        return f"User evidence: {excerpt}" if excerpt else ""

    excerpt = _clip_text(content, limit)
    return f"{role or 'message'}: {excerpt}" if excerpt else ""


def _messages_before_current(raw_messages: Any, current_message: str) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list):
        return []

    messages = [msg for msg in raw_messages if isinstance(msg, dict)]
    if messages:
        latest = messages[-1]
        if (
            str(latest.get("role") or "").lower() == "user"
            and str(latest.get("content") or "") == str(current_message)
        ):
            messages = messages[:-1]
    return messages


def _completed_session_messages(raw_messages: Any, current_message: str) -> list[dict[str, Any]]:
    messages = _messages_before_current(raw_messages, current_message)

    completed: list[dict[str, Any]] = []
    include_current_turn = False
    for msg in messages:
        role = str(msg.get("role") or "").lower()
        state = str(msg.get("turn_state") or msg.get("state") or "").lower()
        if role == "user":
            include_current_turn = state not in {
                "pending",
                "interrupted",
                "superseded",
                "cancelled",
                "canceled",
            }
            if include_current_turn:
                completed.append(msg)
            continue
        if include_current_turn:
            completed.append(msg)
    return completed


def _completed_session_messages_without_active_turn(raw_messages: Any) -> list[dict[str, Any]]:
    """Return completed messages while dropping the currently pending turn."""
    if not isinstance(raw_messages, list):
        return []

    messages = [msg for msg in raw_messages if isinstance(msg, dict)]
    active_start: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if str(message.get("role") or "").lower() != "user":
            continue
        state = str(message.get("turn_state") or message.get("state") or "").lower()
        if state == "pending":
            active_start = index
            break
        break

    if active_start is not None:
        messages = messages[:active_start]

    completed: list[dict[str, Any]] = []
    include_current_turn = False
    for message in messages:
        role = str(message.get("role") or "").lower()
        state = str(message.get("turn_state") or message.get("state") or "").lower()
        if role == "user":
            include_current_turn = state not in {
                "pending",
                "interrupted",
                "superseded",
                "cancelled",
                "canceled",
            }
            if include_current_turn:
                completed.append(message)
            continue
        if include_current_turn:
            completed.append(message)
    return completed


def _interrupted_user_evidence_lines(
    raw_messages: Any,
    current_message: str,
    *,
    max_items: int = 6,
) -> list[str]:
    """Keep short user-stated facts from aborted turns without making them executable."""
    messages = _messages_before_current(raw_messages, current_message)

    evidence: list[str] = []
    for message in messages:
        if str(message.get("role") or "").lower() != "user":
            continue
        state = str(message.get("turn_state") or message.get("state") or "").lower()
        if state not in {"interrupted", "superseded", "cancelled", "canceled"}:
            continue
        if not _is_bounded_user_evidence(message):
            continue
        excerpt = _clip_text(message.get("content", ""), 220)
        if excerpt:
            evidence.append(f"{state}: {excerpt}")

    return evidence[-max_items:]


def _interrupted_turn_evidence_lines(
    raw_messages: Any,
    current_message: str,
    *,
    max_items: int = 10,
) -> list[str]:
    """Keep persisted evidence from aborted turns as historical context."""
    messages = _messages_before_current(raw_messages, current_message)

    evidence: list[str] = []
    seen: set[str] = set()
    include_current_turn = False
    for message in messages:
        role = str(message.get("role") or "").lower()
        if role == "user":
            state = str(message.get("turn_state") or message.get("state") or "").lower()
            include_current_turn = state in {
                "interrupted",
                "superseded",
                "cancelled",
                "canceled",
            }
            continue
        if not include_current_turn:
            continue
        is_incomplete_marker = role == "assistant" and message.get("incomplete") is True
        if not is_incomplete_marker and not _message_is_tool_backed_evidence(message):
            continue
        event = _format_session_event(message, limit=360)
        if not event or event in seen:
            continue
        evidence.append(event)
        seen.add(event)

    return evidence[-max_items:]


def _latest_prior_user_task_line(
    raw_messages: Any,
    current_message: str,
    *,
    limit: int = 700,
) -> str:
    """Return the latest prior user request for explicit continuation-only turns."""
    messages = _messages_before_current(raw_messages, current_message)
    for message in reversed(messages):
        if str(message.get("role") or "").lower() != "user":
            continue
        state = str(message.get("turn_state") or message.get("state") or "").lower()
        if state in {"superseded", "cancelled", "canceled"}:
            continue
        excerpt = _clip_text(message.get("content", ""), limit)
        if not excerpt:
            continue
        skill_names = [
            str(item.get("name") or "").strip()
            for item in (message.get("invoked_skills") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        ]
        skill_suffix = (
            f" | selected skills: {', '.join(dict.fromkeys(skill_names))}"
            if skill_names
            else ""
        )
        return f"{state or 'previous'}: {excerpt}{skill_suffix}"
    return ""


def _latest_prior_user_turn_messages(
    raw_messages: Any,
    current_message: str,
) -> list[dict[str, Any]]:
    """Return only the latest prior user-selected turn for bounded continuation."""
    messages = _messages_before_current(raw_messages, current_message)
    start_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if str(message.get("role") or "").lower() != "user":
            continue
        state = str(message.get("turn_state") or message.get("state") or "").lower()
        if state in {"superseded", "cancelled", "canceled"}:
            continue
        start_index = index
        break
    if start_index is None:
        return []

    selected: list[dict[str, Any]] = []
    for message in messages[start_index:]:
        state = str(message.get("turn_state") or message.get("state") or "").lower()
        if state in {"superseded", "cancelled", "canceled"}:
            continue
        selected.append(message)
    return selected


def _group_completed_turns(messages: list[dict[str, Any]]) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    turns: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    current_user: dict[str, Any] | None = None
    current_events: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role") or "").lower()
        if role == "user":
            if current_user is not None and current_events:
                turns.append((current_user, current_events))
            current_user = message
            current_events = []
            continue
        if current_user is not None:
            current_events.append(message)

    if current_user is not None and current_events:
        turns.append((current_user, current_events))
    return turns


def _summarize_completed_turn(
    user_message: dict[str, Any],
    events: list[dict[str, Any]],
) -> str:
    _ = user_message
    last_assistant = ""
    for message in reversed(events):
        if str(message.get("role") or "").lower() != "assistant":
            continue
        excerpt = _clip_text(message.get("content", ""), 220)
        if excerpt:
            last_assistant = excerpt
            break

    visible_tool_events = _visible_recent_tool_events(events)
    tool_counts: Counter[str] = Counter()
    for message in visible_tool_events:
        name = str(message.get("name") or message.get("tool_name") or "tool")
        tool_counts[name] += 1

    if not last_assistant:
        fallback_events: list[str] = []
        for message in reversed(visible_tool_events):
            fallback = _format_session_event(message, limit=140)
            if fallback:
                fallback_events.append(fallback)
            if len(fallback_events) >= 2:
                break
        if fallback_events:
            last_assistant = "Recent tool evidence: " + " || ".join(reversed(fallback_events))

    parts: list[str] = []
    if last_assistant:
        parts.append(f"Outcome: {last_assistant}")
    if tool_counts:
        tool_summary = ", ".join(
            f"{name} x{count}"
            for name, count in tool_counts.most_common(3)
        )
        if sum(tool_counts.values()) > sum(count for _, count in tool_counts.most_common(3)):
            tool_summary += ", ..."
        parts.append(f"Tools: {tool_summary}")
    return " | ".join(parts)


def _select_turn_summaries_with_budget(
    turns: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    char_budget: int,
) -> list[str]:
    selected: list[str] = []
    used = 0
    for user_message, events in reversed(turns[-DEFAULT_MAX_SESSION_TURNS:]):
        summary = _summarize_completed_turn(user_message, events)
        if not summary:
            continue
        cost = len(summary) + 3
        if selected and used + cost > char_budget:
            break
        selected.append(summary)
        used += cost
    return list(reversed(selected))


def _select_events_with_budget(
    messages: list[dict[str, Any]],
    *,
    char_budget: int,
) -> list[str]:
    for event_limit in EVENT_CLIP_LEVELS:
        events = [
            event
            for event in (
                _format_session_event(message, limit=event_limit)
                for message in messages
            )
            if event
        ]
        total = sum(len(event) + 3 for event in events)
        if total <= char_budget:
            return events

    selected: list[str] = []
    used = 0
    for message in reversed(messages):
        event = _format_session_event(message, limit=EVENT_CLIP_LEVELS[-1])
        if not event:
            continue
        cost = len(event) + 3
        if selected and used + cost > char_budget:
            break
        selected.append(event)
        used += cost
    return list(reversed(selected))


def _message_is_tool_backed_evidence(message: dict[str, Any]) -> bool:
    """Return True for persisted messages that carry concrete tool evidence."""
    role = str(message.get("role") or "").lower()
    if role == "tool":
        return True
    if role != "assistant":
        return False
    if message.get("tool_calls"):
        return True
    if str(message.get("message_kind") or "") == "tool_trace":
        return True
    content = message.get("content", "")
    return isinstance(content, str) and content.lstrip().startswith("[Tool execution summary]")


def _tool_call_payload(tool_call: Any) -> tuple[str, Any]:
    """Return a tool call's name and arguments without assuming provider shape."""
    if not isinstance(tool_call, dict):
        name = str(getattr(tool_call, "name", "") or "")
        function = getattr(tool_call, "function", None)
        if function is not None:
            name = str(getattr(function, "name", None) or name)
            return name, getattr(function, "arguments", None)
        return name, getattr(tool_call, "arguments", None)

    function = tool_call.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or ""), function.get("arguments")
    return str(tool_call.get("name") or ""), tool_call.get("arguments")


def _turn_tool_call_arguments(events: list[dict[str, Any]]) -> dict[str, tuple[str, Any]]:
    """Index assistant tool-call arguments by tool_call_id within a turn."""
    indexed: dict[str, tuple[str, Any]] = {}
    for message in events:
        if str(message.get("role") or "").lower() != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_call_id = str(tool_call.get("id") or tool_call.get("tool_call_id") or "")
            if not tool_call_id:
                continue
            name, arguments = _tool_call_payload(tool_call)
            indexed[tool_call_id] = (name, arguments)
    return indexed


def _tool_message_category(
    message: dict[str, Any],
    tool_calls_by_id: dict[str, tuple[str, Any]],
) -> str:
    """Classify a tool result using persisted call arguments when available."""
    tool_name = str(message.get("name") or message.get("tool_name") or "").strip()
    arguments: Any = None
    tool_call_id = str(message.get("tool_call_id") or message.get("id") or "")
    if tool_call_id and tool_call_id in tool_calls_by_id:
        call_name, arguments = tool_calls_by_id[tool_call_id]
        if call_name:
            tool_name = call_name
    return classify_tool_invocation_category(tool_name or "tool", arguments)


def _tool_result_failed(message: dict[str, Any]) -> bool:
    """Return True for generic guardrail/error outputs."""
    if message.get("guardrail_stop") is True:
        return True
    status = str(message.get("status") or "").strip().casefold()
    if status in {"failed", "error", "terminated", "cancelled", "canceled"}:
        return True
    for key in ("returncode", "return_code", "exit_code"):
        if key not in message:
            continue
        value = message.get(key)
        if value in (None, "", "running"):
            continue
        try:
            if int(value) != 0:
                return str(message.get("tool_outcome") or "") != "already_satisfied"
        except (TypeError, ValueError):
            continue

    text = _OBSERVED_OUTPUT_RE.sub("", str(message.get("content") or "").strip())
    normalized = text.strip().casefold()
    if not normalized:
        return True
    failure_prefixes = (
        "current-session fact check required:",
        "read_file_cache_hit:",
        "rejected:",
        "repeated read skipped:",
        "error:",
        "stop_tool_loop:",
        "security error:",
        "traceback",
    )
    return normalized.startswith(failure_prefixes)


def _assistant_summary_failed(text: Any) -> bool:
    """Return True when an assistant summary is an internal non-progress marker."""
    return _tool_result_failed({"content": text})


def _is_meta_history_tool_event(message: dict[str, Any]) -> bool:
    """Return True for history-recovery tool output that would recursively nest history."""
    name = str(message.get("name") or message.get("tool_name") or "").strip()
    return name == "search_history"


def _visible_recent_tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return tool results that are useful as compact evidence for later turns."""
    visible: list[dict[str, Any]] = []
    for event in events:
        if str(event.get("role") or "").lower() != "tool":
            continue
        if _tool_result_failed(event):
            continue
        if _is_meta_history_tool_event(event):
            continue
        visible.append(event)
    return visible


def _message_visible_in_compact(message: dict[str, Any]) -> bool:
    """Return True when a persisted message should be rendered into compact context."""
    role = str(message.get("role") or "").lower()
    if role == "tool":
        return not _tool_result_failed(message) and not _is_meta_history_tool_event(message)
    if role == "assistant":
        if message.get("tool_calls"):
            return False
        return not _assistant_summary_failed(message.get("content", ""))
    return True


def _turn_stateful_tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return successful state-changing tool result messages for a turn."""
    tool_calls_by_id = _turn_tool_call_arguments(events)
    selected: list[dict[str, Any]] = []
    for message in events:
        if str(message.get("role") or "").lower() != "tool":
            continue
        if _tool_message_category(message, tool_calls_by_id) != "stateful":
            continue
        if _tool_result_failed(message):
            continue
        selected.append(message)
    return selected


def _turn_skill_names(user_message: dict[str, Any]) -> list[str]:
    names = [
        str(item.get("name") or "").strip()
        for item in (user_message.get("invoked_skills") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    return list(dict.fromkeys(names))


def _format_substantive_turn_line(
    user_message: dict[str, Any],
    events: list[dict[str, Any]],
    *,
    limit: int,
) -> str:
    """Format one completed turn whose tool evidence shows real progress."""
    stateful_events = _turn_stateful_tool_events(events)
    if not stateful_events:
        return ""

    user_excerpt = _clip_text(user_message.get("content", ""), min(limit, 260))
    skill_names = _turn_skill_names(user_message)
    parts: list[str] = []
    if user_excerpt:
        parts.append(f"User request: {user_excerpt}")
    if skill_names:
        parts.append(f"Selected skills: {', '.join(skill_names)}")

    latest_progress = _format_session_event(stateful_events[-1], limit=limit)
    if latest_progress:
        parts.append(f"Latest stateful tool result: {latest_progress}")

    assistant_summary = ""
    for event in reversed(events):
        if str(event.get("role") or "").lower() != "assistant":
            continue
        if event.get("tool_calls"):
            continue
        assistant_summary = _clip_text(event.get("content", ""), min(limit, 260))
        if assistant_summary:
            break
    if assistant_summary:
        parts.append(f"Assistant summary: {assistant_summary}")

    return " | ".join(parts)


def _select_recent_substantive_turn_lines(
    turns: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    *,
    char_budget: int,
    max_items: int = 6,
) -> list[str]:
    """Return recent completed turns whose tool evidence made real progress."""
    selected: list[str] = []
    seen: set[str] = set()
    used = 0
    for user_message, events in reversed(turns):
        line = _format_substantive_turn_line(
            user_message,
            events,
            limit=520,
        )
        if not line or line in seen:
            continue
        cost = len(line) + 3
        if selected and used + cost > char_budget:
            break
        selected.append(line)
        seen.add(line)
        used += cost
        if len(selected) >= max_items:
            break
    return selected


def build_recent_session_turns_payload(
    raw_messages: Any,
    *,
    limit: int = 8,
    max_content_length: int = 1000,
) -> dict[str, Any]:
    """Build structured recent-turn evidence for history recovery tools."""
    completed = _completed_session_messages_without_active_turn(raw_messages)
    turns = _group_completed_turns(completed)
    selected: list[dict[str, Any]] = []
    substantive_turns: list[dict[str, Any]] = []
    latest_substantive: dict[str, Any] | None = None

    scan_limit = max(int(limit or 0), 80)
    for scanned, (user_message, events) in enumerate(reversed(turns), start=1):
        stateful_events = _turn_stateful_tool_events(events)
        assistant_summary = ""
        for event in reversed(events):
            if str(event.get("role") or "").lower() != "assistant":
                continue
            if event.get("tool_calls"):
                continue
            if _assistant_summary_failed(event.get("content", "")):
                continue
            assistant_summary = _clip_text(
                event.get("content", ""),
                min(max_content_length, 700),
            )
            if assistant_summary:
                break
        visible_tool_events = _visible_recent_tool_events(events)

        turn_payload = {
            "user_request": _clip_text(
                user_message.get("content", ""),
                min(max_content_length, 700),
            ),
            "timestamp": user_message.get("timestamp"),
            "turn_state": user_message.get("turn_state") or user_message.get("state"),
            "invoked_skills": _turn_skill_names(user_message),
            "has_stateful_progress": bool(stateful_events),
            "latest_stateful_tool_result": (
                _format_session_event(
                    stateful_events[-1],
                    limit=min(max_content_length, 900),
                )
                if stateful_events
                else ""
            ),
            "recent_tool_results": [
                _format_session_event(event, limit=min(max_content_length, 520))
                for event in visible_tool_events[-3:]
                if _format_session_event(event, limit=min(max_content_length, 520))
            ],
            "assistant_summary": assistant_summary,
        }

        if stateful_events:
            if latest_substantive is None:
                latest_substantive = turn_payload
            if len(substantive_turns) < max(1, limit):
                substantive_turns.append(turn_payload)

        if len(selected) < max(1, limit):
            selected.append(turn_payload)

        if (
            len(selected) >= max(1, limit)
            and latest_substantive is not None
        ):
            break
        if scanned >= scan_limit:
            break

    return {
        "latest_substantive_turn": latest_substantive,
        "substantive_turns": substantive_turns,
        "total": len(selected),
        "turns": selected,
        "note": (
            "Use substantive_turns as same-session prior-work evidence. "
            "Pick the turn or turns relevant to the newest request; read-only "
            "diagnostic turns and assistant summaries are secondary and should "
            "not override stateful tool results."
        ),
    }


def _select_recent_tool_evidence_lines(
    messages: list[dict[str, Any]],
    *,
    char_budget: int,
    max_items: int = 14,
) -> list[str]:
    """Return newest-first tool evidence that can resume or audit prior work."""
    selected: list[str] = []
    seen: set[str] = set()
    used = 0

    for message in reversed(messages):
        if not _message_is_tool_backed_evidence(message):
            continue
        if not _message_visible_in_compact(message):
            continue
        event = _format_session_event(message, limit=360)
        if not event or event in seen:
            continue
        cost = len(event) + 3
        if selected and used + cost > char_budget:
            break
        selected.append(event)
        seen.add(event)
        used += cost
        if len(selected) >= max_items:
            break

    return selected


def build_session_compact_context(
    session: Any,
    current_message: str,
    *,
    max_messages: int = DEFAULT_MAX_SESSION_MESSAGES,
    char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET,
    resume_latest_user_turn: bool = False,
) -> str:
    """Build an ordered compact block from the current session transcript."""
    if session is None:
        return ""

    try:
        raw_messages = (
            session.get_messages()
            if hasattr(session, "get_messages")
            else session.get_history()
        )
    except Exception:
        return ""

    completed = (
        _latest_prior_user_turn_messages(raw_messages, current_message)
        if resume_latest_user_turn
        else _completed_session_messages(raw_messages, current_message)
    )
    interrupted_evidence = _interrupted_user_evidence_lines(raw_messages, current_message)
    interrupted_turn_evidence = _interrupted_turn_evidence_lines(raw_messages, current_message)
    if not completed and not interrupted_evidence and not interrupted_turn_evidence:
        return ""

    try:
        from spoon_bot.utils.privacy import mask_secrets
    except Exception:
        mask_secrets = lambda value: value  # type: ignore[assignment]

    lines: list[str] = [
        "## Current Session Compact",
        "Source: ordered same-session transcript compact, not long-term memory.",
        "Use it only when the newest request asks about prior work, prior tool results, or continuing this session.",
        "The newest user request remains authoritative for task selection and output requirements.",
        "Never start or continue actions that are implied only by this compact; use it as evidence after the newest request selects the task.",
        "An empty long-term memory search does not mean the same-session transcript is empty.",
        "Completed turn summaries below contain only assistant/tool outcomes, not prior user instructions.",
        "If a continuation anchor is shown above, treat it as the only prior user request selected by the newest continuation request.",
    ]

    latest_prior = _latest_prior_user_task_line(raw_messages, current_message)
    if resume_latest_user_turn:
        anchor = latest_prior
        if anchor:
            lines.append("Continuation anchor selected by the newest continuation-only request:")
            lines.append(f"- Latest prior user request: {mask_secrets(anchor)}")
            lines.append(
                "Resume only one bounded continuation unit from this anchor and current live state. "
                "If selected skills are shown, continue that skill family rather than another earlier skill."
            )

    if interrupted_evidence:
        lines.append(
            "Interrupted/superseded user evidence "
            "(facts only; do not execute unless the newest request explicitly resumes it):"
        )
        for item in interrupted_evidence:
            lines.append(f"- {mask_secrets(item)}")

    if interrupted_turn_evidence:
        lines.append(
            "Interrupted/superseded turn evidence "
            "(historical only; verify live state before any new side effect):"
        )
        for item in interrupted_turn_evidence:
            lines.append(f"- {mask_secrets(item)}")

    completed_turns = _group_completed_turns(completed)

    substantive_turns = _select_recent_substantive_turn_lines(
        completed_turns,
        char_budget=max(1_000, min(5_500, char_budget // 3)),
    )
    if substantive_turns:
        lines.append(
            "Latest substantive tool-backed turns "
            "(newest first; use the entries relevant to the newest prior-result "
            "question before read-only diagnostics and assistant summaries):"
        )
        for event in substantive_turns:
            lines.append(f"- {mask_secrets(event)}")

    active_evidence = _select_recent_tool_evidence_lines(
        completed,
        char_budget=max(800, min(3_500, char_budget // 4)),
    )
    if active_evidence:
        lines.append(
            "Recent raw tool-backed evidence (newest first; read-only "
            "diagnostic/log searches can be stale and should not override "
            "the substantive turn outcomes above):"
        )
        lines.append(
            "- Start from the newest relevant evidence here, then verify live "
            "state with tools before any additional external side effect."
        )
        for event in active_evidence:
            lines.append(f"- {mask_secrets(event)}")

    turn_summaries = _select_turn_summaries_with_budget(
        completed_turns,
        char_budget=max(600, min(4_000, char_budget // 4)),
    )
    if turn_summaries:
        lines.append("Recent completed turn summaries:")
        lines.append("(assistant wording only; tool evidence above is authoritative)")
        for summary in turn_summaries:
            lines.append(f"- {mask_secrets(summary)}")

    task_scoped_user_ids: set[int] = set()
    for user_message, events in completed_turns:
        invoked = user_message.get("invoked_skills")
        has_tool_activity = any(
            str(event.get("role") or "").lower() == "tool"
            or bool(event.get("tool_calls"))
            for event in events
        )
        if invoked or has_tool_activity:
            task_scoped_user_ids.add(id(user_message))

    transcript_tail = [
        message
        for message in completed[-max_messages:]
        if _message_visible_in_compact(message)
        if str(message.get("role") or "").lower() != "user"
        or (
            id(message) not in task_scoped_user_ids
            and _is_bounded_user_evidence(message)
        )
    ]
    event_budget = max(1_000, char_budget - sum(len(line) + 1 for line in lines))
    events = _select_events_with_budget(transcript_tail, char_budget=event_budget)
    if events:
        lines.append("Ordered transcript compact:")
        for event in events:
            lines.append(f"- {mask_secrets(event)}")

    return "\n".join(lines).strip()
