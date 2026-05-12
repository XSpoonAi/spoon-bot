"""Same-session transcript compaction helpers.

This module mirrors Claude Code's post-compact shape at a smaller scale:
preserve a compact, ordered view of the active session transcript without
putting prompt/domain routing rules in the agent loop.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from typing import Any


DEFAULT_MAX_SESSION_MESSAGES = 80
DEFAULT_MAX_SESSION_TURNS = 6
DEFAULT_CONTEXT_CHAR_BUDGET = 24_000
EVENT_CLIP_LEVELS = (360, 220, 140)

_OBSERVED_OUTPUT_RE = re.compile(r"^Observed output of cmd [^\n]* execution:\s*")
_PRELOADED_SKILL_BLOCK_RE = re.compile(
    r"\n+---\n\[PRE-LOADED SKILL:[\s\S]*$",
    re.IGNORECASE,
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
    if not text or len(text) > 320:
        return False
    sentence_parts = [
        part.strip()
        for part in re.split(r"[.!?。！？\n]+", text)
        if part.strip()
    ]
    return len(sentence_parts) <= 2


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
        return f"{prefix}({_clip_text(arguments, limit)})"
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


def _completed_session_messages(raw_messages: Any, current_message: str) -> list[dict[str, Any]]:
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

    completed: list[dict[str, Any]] = []
    for msg in messages:
        state = str(msg.get("turn_state") or msg.get("state") or "").lower()
        if state in {"interrupted", "superseded", "cancelled", "canceled"}:
            continue
        completed.append(msg)
    return completed


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

    tool_counts: Counter[str] = Counter()
    for message in events:
        if str(message.get("role") or "").lower() != "tool":
            continue
        name = str(message.get("name") or message.get("tool_name") or "tool")
        tool_counts[name] += 1

    if not last_assistant:
        fallback_events: list[str] = []
        for message in reversed(events):
            if str(message.get("role") or "").lower() != "tool":
                continue
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


def build_session_compact_context(
    session: Any,
    current_message: str,
    *,
    max_messages: int = DEFAULT_MAX_SESSION_MESSAGES,
    char_budget: int = DEFAULT_CONTEXT_CHAR_BUDGET,
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

    completed = _completed_session_messages(raw_messages, current_message)
    if not completed:
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
    ]

    turn_summaries = _select_turn_summaries_with_budget(
        _group_completed_turns(completed),
        char_budget=max(600, min(4_000, char_budget // 4)),
    )
    if turn_summaries:
        lines.append("Recent completed turn summaries:")
        for summary in turn_summaries:
            lines.append(f"- {mask_secrets(summary)}")

    task_scoped_user_ids: set[int] = set()
    for user_message, events in _group_completed_turns(completed):
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
