"""Small, domain-neutral helpers for turn boundary handling.

The loop should not route by product names, prompt phrases, or CLI-specific
``NEXT`` text.  These helpers look only at structured tool events and keep the
LLM responsible for deciding the next real action from the current context.
"""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from spoon_bot.agent.tools.shell import ShellTool

_SETUP_OR_READ_ONLY_TOOLS = frozenset({
    "web_fetch",
    "web_search",
    "skill_marketplace",
    "self_upgrade",
    "read_file",
    "list_dir",
    "grep",
    "search_history",
})

_STATEFUL_PROGRESS_TOOLS = frozenset({
    "write_file",
    "edit_file",
    "service_expose",
    "spawn",
    "cron",
})

_SKILL_CONTRACT_MARKERS = (
    "skill-ref",
    "[skill.md execution contract]",
    "[skill.md execution summary]",
)

_USER_SUMMARY_PREFIXES = (
    "read it aloud:",
    "final answer:",
    "user summary:",
)


def dominant_non_latin_scripts(text: str | None) -> list[str]:
    """Return dominant non-Latin Unicode script labels in text."""
    counts: dict[str, int] = {}
    for char in str(text or ""):
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        script = name.split(" ", 1)[0]
        if script == "LATIN":
            continue
        counts[script] = counts.get(script, 0) + 1

    if not counts:
        return []
    max_count = max(counts.values())
    floor = max(2, max_count // 3)
    return sorted(script for script, count in counts.items() if count >= floor)


def _stringify_payload(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (dict, list)):
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)
    return str(payload)


def _unwrap_payload(payload: Any) -> Any:
    for _ in range(3):
        if not isinstance(payload, dict):
            break
        for key in (
            "result",
            "output",
            "content",
            "text",
            "message",
            "model_result",
            "model_output",
            "model_content",
        ):
            if key in payload:
                payload = payload.get(key)
                break
        else:
            break
    return payload


def _stream_tool_event_name(event: dict[str, Any]) -> str:
    metadata = dict(event.get("metadata") or {})
    return str(metadata.get("name") or metadata.get("tool") or "").strip().lower()


def _stream_tool_event_arguments(event: dict[str, Any]) -> Any:
    metadata = dict(event.get("metadata") or {})
    for key in ("arguments", "input", "args"):
        if key not in metadata:
            continue
        value = metadata.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            try:
                return json.loads(stripped)
            except Exception:
                return stripped
        if value:
            return value
    return None


def _stream_tool_event_text(event: dict[str, Any]) -> str:
    metadata = dict(event.get("metadata") or {})
    payload = (
        metadata.get("full_output")
        or metadata.get("full_result")
        or metadata.get("full_content")
        or metadata.get("output")
        or metadata.get("result")
        or metadata.get("content")
        or metadata.get("model_result")
        or metadata.get("model_output")
        or metadata.get("model_content")
        or event.get("result")
        or event.get("output")
        or event.get("content")
        or event.get("delta")
    )
    return _stringify_payload(_unwrap_payload(payload))


def _contains_workspace_skill_path(text: str) -> bool:
    marker = "skills/"
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            return False
        suffix = text[index + len(marker):]
        slash_index = suffix.find("/")
        if slash_index > 0:
            skill_name = suffix[:slash_index]
            if skill_name not in {".", ".."} and all(
                char.isalnum() or char in {"-", "_", "."}
                for char in skill_name
            ):
                return True
        start = index + len(marker)


def _tool_result_indicates_workspace_skill_activity(event: dict[str, Any]) -> bool:
    tool_name = _stream_tool_event_name(event)
    text = _stream_tool_event_text(event).replace("\\", "/").casefold()
    arguments_text = _stringify_payload(_stream_tool_event_arguments(event)).replace("\\", "/").casefold()
    if tool_name == "skill_marketplace":
        return True
    if any(marker in text for marker in _SKILL_CONTRACT_MARKERS):
        return True
    if _contains_workspace_skill_path(arguments_text):
        return True
    if tool_name in _SETUP_OR_READ_ONLY_TOOLS and _contains_workspace_skill_path(text):
        return True
    return False


def should_run_skill_contract_check(tool_result_events: list[dict[str, Any]]) -> bool:
    """Return True when a workspace skill contract is active in this turn."""
    return any(
        _tool_result_indicates_workspace_skill_activity(event)
        for event in tool_result_events
    )


def _shell_event_command_is_read_only(event: dict[str, Any]) -> bool:
    arguments = _stream_tool_event_arguments(event)
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or "").strip()
    elif isinstance(arguments, str):
        command = arguments.strip()
    else:
        command = ""
    return bool(command) and ShellTool.command_is_plain_read_only_inspection(command)


def _tool_event_is_setup_or_read_only(event: dict[str, Any]) -> bool:
    tool_name = _stream_tool_event_name(event)
    if not tool_name:
        return False
    if tool_name in _SETUP_OR_READ_ONLY_TOOLS:
        return True
    if tool_name == "shell":
        return _shell_event_command_is_read_only(event)
    return False


def _tool_event_has_stateful_progress(event: dict[str, Any]) -> bool:
    tool_name = _stream_tool_event_name(event)
    if not tool_name:
        return False
    if tool_name in _STATEFUL_PROGRESS_TOOLS:
        return True
    if tool_name == "shell":
        text = " ".join(_stream_tool_event_text(event).strip().casefold().split())
        if text in {"", "(no output)", "no output"}:
            return False
    return not _tool_event_is_setup_or_read_only(event)


def tool_events_have_stateful_progress(
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True when the turn has evidence of a mutating or running action."""
    return any(_tool_event_has_stateful_progress(event) for event in tool_result_events)


def tool_events_are_read_only(
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True when every observed tool result is setup/inspection only."""
    return bool(tool_result_events) and all(
        _tool_event_is_setup_or_read_only(event)
        for event in tool_result_events
    )


def final_answer_is_raw_tool_evidence(
    final_content: str | None,
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Detect a final answer that is just leaked tool/file output.

    This is intentionally format-based and domain-neutral. It does not infer
    user intent or parse business vocabulary; it only prevents a raw
    read/list/search artifact from being treated as a user-facing completion.
    """
    final_text = str(final_content or "").strip()
    if not final_text:
        return False

    normalized_final = " ".join(final_text.split())
    if normalized_final.startswith("[file:") or normalized_final.startswith("Observed output of cmd"):
        return True

    for event in reversed(tool_result_events[-8:]):
        tool_text = _stream_tool_event_text(event).strip()
        if not tool_text:
            continue
        normalized_tool = " ".join(tool_text.split())
        if not normalized_tool:
            continue
        if normalized_final == normalized_tool:
            return True
        if (
            len(normalized_final) >= 160
            and normalized_tool.startswith(normalized_final[:160])
        ):
            return True
    return False


def read_only_tool_turn_needs_continuation(
    final_content: str | None,
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True when a tool-backed turn stopped at inspection evidence.

    The continuation prompt still lets the model answer immediately if the
    latest request was truly read-only. This helper only rejects a raw
    inspection artifact as terminal content.
    """
    if not tool_events_are_read_only(tool_result_events):
        return False
    if tool_events_have_stateful_progress(tool_result_events):
        return False
    return final_answer_is_raw_tool_evidence(final_content, tool_result_events)


def skill_contract_has_progress(
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True after a skill-backed turn performs a non-read action."""
    if not should_run_skill_contract_check(tool_result_events):
        return False
    return any(_tool_event_has_stateful_progress(event) for event in tool_result_events)


def skill_contract_inspection_stalled_after_progress(
    tool_result_events: list[dict[str, Any]],
    *,
    min_trailing_events: int = 2,
) -> bool:
    """Detect repeated setup/read-only events after state-changing progress."""
    if not should_run_skill_contract_check(tool_result_events):
        return False

    saw_progress = False
    trailing_read_only = 0
    for event in tool_result_events:
        if _tool_event_has_stateful_progress(event):
            saw_progress = True
            trailing_read_only = 0
            continue
        if saw_progress and _tool_event_is_setup_or_read_only(event):
            trailing_read_only += 1

    return trailing_read_only >= max(1, int(min_trailing_events))


def _latest_non_empty_tool_event(
    tool_result_events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for event in reversed(tool_result_events):
        if _stream_tool_event_text(event).strip():
            return event
    return tool_result_events[-1] if tool_result_events else None


def _iter_user_summary_lines(tool_outputs: list[str]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for output in tool_outputs:
        for raw_line in str(output or "").splitlines():
            stripped = raw_line.strip()
            lower = stripped.casefold()
            for prefix in _USER_SUMMARY_PREFIXES:
                if not lower.startswith(prefix):
                    continue
                summary = " ".join(stripped[len(prefix):].strip().split())
                if summary and summary not in seen:
                    seen.add(summary)
                    lines.append(summary)
                break
    return lines


def latest_tool_event_has_user_summary_marker(
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True when the latest non-empty tool result gives a user summary."""
    latest = _latest_non_empty_tool_event(tool_result_events)
    return bool(latest and _iter_user_summary_lines([_stream_tool_event_text(latest)]))


def _tool_event_shell_job_fields(event: dict[str, Any]) -> dict[str, str]:
    """Return simple key/value fields from the shell job text format."""
    if _stream_tool_event_name(event) != "shell":
        return {}

    fields: dict[str, str] = {}
    for raw_line in _stream_tool_event_text(event).splitlines():
        key, separator, value = raw_line.partition(":")
        if not separator:
            continue
        key = " ".join(key.strip().casefold().split())
        value = " ".join(value.strip().casefold().split())
        if key and value:
            fields[key] = value
    return fields


def latest_tool_event_has_active_background_job(
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True when the latest shell evidence is a managed job still running."""
    latest = _latest_non_empty_tool_event(tool_result_events)
    if not latest:
        return False

    fields = _tool_event_shell_job_fields(latest)
    if not fields:
        return False

    has_job_id = bool(fields.get("job_id"))
    status = fields.get("status", "")
    returncode = fields.get("returncode", "")
    return has_job_id and (
        status == "running" or status.startswith("running ") or returncode == "running"
    )


def _skill_contract_was_loaded(tool_result_events: list[dict[str, Any]]) -> bool:
    """Return True once the model has received an actual skill contract."""
    for event in tool_result_events:
        tool_name = _stream_tool_event_name(event)
        text = _stream_tool_event_text(event).replace("\\", "/").casefold()
        arguments_text = _stringify_payload(_stream_tool_event_arguments(event)).replace("\\", "/").casefold()
        if any(marker in text for marker in _SKILL_CONTRACT_MARKERS):
            return True
        if tool_name == "read_file" and (
            _contains_workspace_skill_path(text)
            or _contains_workspace_skill_path(arguments_text)
        ):
            return True
    return False


def latest_tool_event_from_skill_continuation(
    tool_result_events: list[dict[str, Any]],
) -> bool:
    latest = _latest_non_empty_tool_event(tool_result_events)
    if not latest:
        return False
    metadata = dict(latest.get("metadata") or {})
    return str(metadata.get("repair") or "").strip() == "skill_contract_continuation"


def skill_contract_needs_continuation(
    final_content: str | None,
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True only for structural partial skill turns.

    This intentionally does not parse prompt language, game names, CLI next-step
    text, or domain-specific status.  It asks for another model step only when
    the turn touched a skill and does not yet have terminal tool evidence.
    """
    if not should_run_skill_contract_check(tool_result_events):
        return False
    if latest_tool_event_has_active_background_job(tool_result_events):
        return True

    final_text = str(final_content or "").strip()
    if not final_text or final_text in {"No results", "NO_CONCISE_TOOL_EVIDENCE"}:
        return True
    if not _skill_contract_was_loaded(tool_result_events):
        return False
    has_progress = skill_contract_has_progress(tool_result_events)
    if not has_progress:
        return True
    latest = _latest_non_empty_tool_event(tool_result_events)
    if latest is not None and _tool_event_is_setup_or_read_only(latest):
        return True
    if latest_tool_event_from_skill_continuation(tool_result_events):
        return False
    if latest_tool_event_has_user_summary_marker(tool_result_events):
        return skill_contract_inspection_stalled_after_progress(tool_result_events)
    return False


def latest_tool_event_has_next_command(tool_result_events: list[dict[str, Any]]) -> bool:
    """Do not route from CLI ``NEXT`` text; leave it as model evidence."""
    return False


def _compact_tool_output(text: str, *, limit: int = 1200) -> str:
    compact = "\n".join(line.rstrip() for line in str(text or "").splitlines())
    compact = compact.strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 24].rstrip() + "\n[output truncated]"


def build_user_facing_tool_evidence_answer(
    tool_outputs: list[str],
    *,
    incomplete: bool = False,
    user_message: str | None = None,
) -> str:
    """Build a conservative fallback without inferring workflow completion."""
    summaries = _iter_user_summary_lines(tool_outputs)
    if summaries:
        return summaries[-1]
    for output in reversed(tool_outputs):
        compact = _compact_tool_output(output, limit=500)
        if compact and compact.casefold() not in {"(no output)", "no output"}:
            return compact
    return (
        "The skill turn produced tool evidence but no final answer."
        if incomplete
        else "NO_CONCISE_TOOL_EVIDENCE"
    )


def build_user_facing_tool_event_answer(
    tool_result_events: list[dict[str, Any]],
    *,
    incomplete: bool = False,
    user_message: str | None = None,
) -> str:
    if (
        tool_result_events
        and tool_events_are_read_only(tool_result_events)
        and not tool_events_have_stateful_progress(tool_result_events)
    ):
        lines = [
            "The turn only produced read-only tool evidence; no workspace mutation or service action was verified."
        ]
        for event in tool_result_events[-8:]:
            tool_name = _stream_tool_event_name(event) or "tool"
            args = _stream_tool_event_arguments(event)
            args_text = _stringify_payload(args).strip()
            path = ""
            if isinstance(args, dict):
                path = str(args.get("path") or args.get("file_path") or "").strip()
            elif args_text:
                try:
                    parsed = json.loads(args_text)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    path = str(parsed.get("path") or parsed.get("file_path") or "").strip()
            detail = f"- {tool_name}"
            if path:
                detail += f": {path}"
            elif args_text:
                detail += ": " + _compact_tool_output(args_text, limit=180)
            lines.append(detail)
        if incomplete:
            lines.append("The requested task did not complete before the agent stopped.")
        return "\n".join(lines)

    return build_user_facing_tool_evidence_answer(
        [_stream_tool_event_text(event) for event in tool_result_events],
        incomplete=incomplete,
        user_message=user_message,
    )


def build_tool_event_synthesis_brief(
    tool_outputs: list[Any],
    *,
    incomplete: bool = False,
    user_message: str | None = None,
) -> str:
    lines = [
        "[FINAL ANSWER SYNTHESIS INPUT]",
        "Use the tool evidence below. The user request is not evidence.",
        "Do not invent completed work, accounts, ids, balances, or registrations.",
        "Write a concise conversational conclusion for a person. Use local commands, "
        "paths, status tokens, and exception text as evidence, but unless the user "
        "explicitly requested raw commands or logs, explain what they mean instead "
        "of copying them literally.",
        "If tool evidence says a remote action is already satisfied, already claimed, "
        "already joined, already registered, or otherwise completed, do not present "
        "repeating that same action as the next step. Report the current blocker or "
        "verified outcome instead.",
        f"Workflow status: {'incomplete_or_paused' if incomplete else 'evidence_available'}",
    ]
    if incomplete:
        lines.append(
            "If evidence shows a running background job or a partial workflow, "
            "state that it is still running/partial instead of marking the "
            "requested workflow complete."
        )
    if user_message:
        lines.extend(["", "Newest user request:", str(user_message).strip()])
        scripts = dominant_non_latin_scripts(user_message)
        if scripts:
            lines.extend([
                "",
                "Required response language:",
                "The newest user request uses dominant non-Latin Unicode "
                f"script(s): {', '.join(scripts)}. Do not answer in English "
                "unless the user explicitly requested English.",
            ])
    terminal_summaries = _iter_user_summary_lines([
        _stream_tool_event_text(output) if isinstance(output, dict) else str(output or "")
        for output in tool_outputs
    ])
    if terminal_summaries:
        lines.extend([
            "",
            "Terminal user-facing summaries from tools:",
        ])
        for summary in terminal_summaries[-6:]:
            lines.append(f"- {summary}")
        lines.append(
            "These are completed tool results. Report them as verified outcomes; "
            "do not transform old setup examples or command lists into a new "
            "offer to execute the same workflow."
        )
    for index, output in enumerate(tool_outputs[-12:], start=1):
        if isinstance(output, dict):
            output = _stream_tool_event_text(output)
        compact = _compact_tool_output(str(output or ""))
        if compact:
            lines.extend(["", f"Tool evidence {index}:", compact])
    return "\n".join(lines)


def build_tool_evidence_synthesis_brief(
    tool_outputs: list[Any],
    *,
    incomplete: bool = False,
    user_message: str | None = None,
) -> str:
    """Backward-compatible name for generic tool-evidence synthesis."""
    return build_tool_event_synthesis_brief(
        tool_outputs,
        incomplete=incomplete,
        user_message=user_message,
    )


def final_answer_denies_available_tool_evidence(
    final_content: str | None,
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Do not override the model with text-token heuristics."""
    return False
