"""Turn-boundary verifier helpers.

These helpers keep workflow-specific completion checks out of the main agent
loop.  They operate on structured tool evidence instead of prompt routing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from spoon_bot.agent.request_hints import extract_shell_command_candidates


_LONG_0X_VALUE_RE = re.compile(r"\b0x[0-9a-fA-F]{64}\b")
_OBSERVED_OUTPUT_PREFIX_RE = re.compile(
    r"^Observed output of cmd [^\n]* execution:\s*",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(r"<([A-Za-z0-9_-]+)>")
_USER_SUMMARY_PREFIXES = (
    "read it aloud:",
    "final answer:",
    "user summary:",
)
_STATUS_LINE_MARKERS = (
    "complete",
    "completed",
    "success",
    "passed",
    "installed",
    "registered",
    "joined",
    "final_state",
    "result=",
    "status=",
    "phase=",
    "rank=",
    "score=",
    "reward=",
)
_ERROR_LINE_MARKERS = (
    "error:",
    "http 4",
    "http 5",
    "exit code:",
    "eaddrinuse",
    "timed out",
    "timeout",
    "failed",
    "already claimed",
    "insufficient funds",
    "urgent:",
)
_LOCAL_SERVICE_URL_RE = re.compile(
    r"\b(?:https?|wss?|ws)://(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::\d+)?\b",
    re.IGNORECASE,
)
_LOCAL_SERVICE_HOSTPORT_RE = re.compile(
    r"\b(?:localhost|127\.0\.0\.1|0\.0\.0\.0):\d{2,5}\b",
    re.IGNORECASE,
)
_PUBLIC_URL_RE = re.compile(
    r"\bhttps://(?!localhost\b|127\.0\.0\.1\b|0\.0\.0\.0\b|\[::1\])[-a-z0-9.]+\.[a-z]{2,}(?::\d+)?(?:/[^\s`'\"<>)]*)?",
    re.IGNORECASE,
)
_SERVICE_WORDS = (
    "api",
    "app",
    "backend",
    "browser",
    "frontend",
    "http",
    "preview",
    "server",
    "service",
    "websocket",
    "服务",
    "服务器",
    "网页",
    "浏览器",
    "应用",
    "聊天室",
    "实时",
)
_LOCAL_ONLY_WORDS = (
    "local only",
    "local-only",
    "localhost only",
    "no public",
    "without public",
    "do not expose",
    "don't expose",
    "本地即可",
    "只要本地",
    "不需要公网",
    "不要公网",
    "不用公网",
    "不要暴露",
    "无需暴露",
)


def _contains_workspace_skill_path(text: str) -> bool:
    """Return True for normalized workspace paths under skills/<skill-name>/."""
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


def _stringify_stream_payload(payload: Any) -> str:
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


def _stream_tool_event_name(event: dict[str, Any]) -> str:
    metadata = dict(event.get("metadata") or {})
    return str(metadata.get("name") or metadata.get("tool") or "").strip().lower()


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
    return _stringify_stream_payload(payload)


def _tool_result_indicates_workspace_skill_activity(event: dict[str, Any]) -> bool:
    tool_name = _stream_tool_event_name(event)
    text = _stream_tool_event_text(event)
    normalized = text.replace("\\", "/").casefold()

    if tool_name == "skill_marketplace":
        return "skill " in normalized and any(
            word in normalized
            for word in ("installed", "already installed", "updated", "removed")
        )

    if _contains_workspace_skill_path(normalized):
        return True

    return False


def should_run_skill_contract_check(tool_result_events: list[dict[str, Any]]) -> bool:
    """Return True when a workspace skill contract is in play this turn."""
    return any(
        _tool_result_indicates_workspace_skill_activity(event)
        for event in tool_result_events
    )


def extract_tool_next_commands(
    tool_result_events: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[str]:
    """Extract structural follow-up shell commands from tool output."""
    commands: list[str] = []
    seen: set[str] = set()
    for event in tool_result_events:
        text = _stream_tool_event_text(event)
        for command in extract_shell_command_candidates(text, limit=limit * 2):
            if not command or command in seen:
                continue
            seen.add(command)
            commands.append(command)
    return commands[-limit:]


def latest_tool_event_has_next_command(tool_result_events: list[dict[str, Any]]) -> bool:
    """Return True when the most recent non-empty tool output still exposes a follow-up command."""
    for event in reversed(tool_result_events):
        text = _stream_tool_event_text(event)
        if not text.strip():
            continue
        return bool(extract_shell_command_candidates(text, limit=1))
    return False


def _simple_scalar_tool_result_text(event: dict[str, Any]) -> str:
    text = _stream_tool_event_text(event).strip()
    if not text or "\n" in text or len(text) > 80:
        return ""
    lower = text.casefold()
    if lower.startswith(("success:", "error:", "warning:")):
        return ""
    if _PLACEHOLDER_RE.search(text):
        return ""
    return text


def extract_resolved_tool_next_commands(
    tool_result_events: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[str]:
    """Return structural follow-up commands with simple placeholders filled."""
    command_entries: list[tuple[int, str]] = []
    scalar_entries: list[tuple[int, str]] = []
    for index, event in enumerate(tool_result_events):
        text = _stream_tool_event_text(event)
        for command in extract_shell_command_candidates(text, limit=limit * 2):
            if command:
                command_entries.append((index, command))
        scalar = _simple_scalar_tool_result_text(event)
        if scalar:
            scalar_entries.append((index, scalar))

    resolved: list[str] = []
    seen: set[str] = set()
    for command_index, command in command_entries:
        updated = command
        placeholders = [match.group(0) for match in _PLACEHOLDER_RE.finditer(command)]
        for placeholder in placeholders:
            placeholder_name = placeholder.strip("<>").casefold()
            if placeholder_name != "answer":
                continue
            replacement = next(
                (
                    scalar
                    for scalar_index, scalar in scalar_entries
                    if scalar_index > command_index
                ),
                "",
            )
            if replacement:
                updated = updated.replace(placeholder, replacement)
        if updated in seen:
            continue
        seen.add(updated)
        resolved.append(updated)
    return resolved[-limit:]


def has_pending_placeholder_next_command(tool_result_events: list[dict[str, Any]]) -> bool:
    """Return True when a follow-up command still has unresolved placeholders."""
    original_commands = extract_tool_next_commands(tool_result_events)
    resolved_commands = extract_resolved_tool_next_commands(tool_result_events)
    if not original_commands or not resolved_commands:
        return False
    latest_original = original_commands[-1]
    latest_resolved = resolved_commands[-1]
    if "<" not in latest_original or ">" not in latest_original:
        return False
    return True


def _clean_user_visible_evidence_line(line: str) -> str:
    cleaned = " ".join(str(line or "").strip().split())
    cleaned = _LONG_0X_VALUE_RE.sub("<omitted 0x id>", cleaned)
    cleaned = cleaned.replace("0x***masked_private_key***", "<redacted 0x value>")
    cleaned = cleaned.replace("***masked_private_key***", "<redacted 0x value>")
    return cleaned


def _iter_user_summary_lines(tool_outputs: list[str]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for output in tool_outputs:
        for raw_line in str(output or "").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            lower = stripped.casefold()
            for prefix in _USER_SUMMARY_PREFIXES:
                if not lower.startswith(prefix):
                    continue
                summary = _clean_user_visible_evidence_line(
                    stripped[len(prefix):].strip()
                )
                if summary and summary not in seen:
                    seen.add(summary)
                    lines.append(summary)
                break
    return lines


def tool_events_have_user_summary_marker(tool_result_events: list[dict[str, Any]]) -> bool:
    """Return True when tool output carries an explicit user-facing summary."""
    return bool(_iter_user_summary_lines([
        _stream_tool_event_text(event)
        for event in tool_result_events
    ]))


def latest_tool_event_has_user_summary_marker(
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True when the latest non-empty tool result carries a final summary."""
    for event in reversed(tool_result_events):
        text = _stream_tool_event_text(event)
        if not text.strip():
            continue
        return bool(_iter_user_summary_lines([text]))
    return False


def tool_events_have_skill_install_completion(
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True when skill-management evidence proves installation is done."""
    for event in tool_result_events:
        if _stream_tool_event_name(event) != "skill_marketplace":
            continue
        text = _stream_tool_event_text(event).casefold()
        if "skill " not in text:
            continue
        if "success:" in text and "installed" in text:
            return True
        if "already installed" in text or "is already installed" in text:
            return True
    return False


def _public_url_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        value = payload.get("public_url") or payload.get("url")
        if isinstance(value, str) and _PUBLIC_URL_RE.search(value):
            return value.strip()
        for nested_key in ("verification", "public_readiness", "result"):
            nested = payload.get(nested_key)
            nested_url = _public_url_from_payload(nested)
            if nested_url:
                return nested_url
    if isinstance(payload, list):
        for item in payload:
            nested_url = _public_url_from_payload(item)
            if nested_url:
                return nested_url
    return ""


def _parse_json_payload(text: str) -> Any | None:
    stripped = str(text or "").strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def tool_events_have_verified_public_url(tool_result_events: list[dict[str, Any]]) -> bool:
    """Return True when service exposure evidence contains a verified public URL."""
    for event in tool_result_events:
        if _stream_tool_event_name(event) != "service_expose":
            continue
        text = _stream_tool_event_text(event)
        payload = _parse_json_payload(text)
        if isinstance(payload, dict):
            success = payload.get("success")
            public_url = payload.get("public_url")
            if success is True and isinstance(public_url, str) and _PUBLIC_URL_RE.search(public_url):
                return True
            nested_url = _public_url_from_payload(payload)
            if success is True and nested_url:
                return True
        if re.search(r'"success"\s*:\s*true', text, re.IGNORECASE) and re.search(
            r'"public_url"\s*:\s*"https://(?!localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])',
            text,
            re.IGNORECASE,
        ):
            return True
    return False


def _explicitly_local_only(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(marker in lowered for marker in _LOCAL_ONLY_WORDS)


def _mentions_browser_service(text: str) -> bool:
    lowered = str(text or "").casefold()
    return any(word in lowered for word in _SERVICE_WORDS)


def _mentions_local_service_only(text: str) -> bool:
    if _LOCAL_SERVICE_URL_RE.search(text) or _LOCAL_SERVICE_HOSTPORT_RE.search(text):
        return True
    lowered = str(text or "").casefold()
    return (
        ("localhost" in lowered or "0.0.0.0" in lowered or "127.0.0.1" in lowered)
        and any(marker in lowered for marker in ("listen", "listening", "port", "监听", "端口"))
    )


def final_response_needs_public_service_exposure_recovery(
    user_request: str,
    final_content: str,
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True when a browser/service task ended with only local access.

    The check is evidence-based: it does not route by one product prompt.  It
    only fires when the newest request/final answer is about a browser-facing
    service, the user did not ask for local-only behavior, and the answer still
    exposes loopback/local service coordinates without verified public exposure
    tool evidence.
    """
    if tool_events_have_verified_public_url(tool_result_events):
        return False
    request_text = str(user_request or "")
    answer_text = str(final_content or "")
    combined = f"{request_text}\n{answer_text}"
    if _explicitly_local_only(combined):
        return False
    if not _mentions_browser_service(combined):
        return False
    if _mentions_local_service_only(answer_text):
        return True
    for event in reversed(tool_result_events[-8:]):
        text = _stream_tool_event_text(event)
        if _mentions_local_service_only(text):
            return True
    return False


def build_skill_install_completion_answer(
    tool_result_events: list[dict[str, Any]],
) -> str:
    """Build a concise answer from skill installation evidence."""
    for event in reversed(tool_result_events):
        if _stream_tool_event_name(event) != "skill_marketplace":
            continue
        text = _stream_tool_event_text(event)
        lowered = text.casefold()
        if "skill " not in lowered:
            continue
        if not (
            ("success:" in lowered and "installed" in lowered)
            or "already installed" in lowered
            or "is already installed" in lowered
        ):
            continue

        skill_name = ""
        match = re.search(r"Skill\s+'([^']+)'", text)
        if match:
            skill_name = match.group(1).strip()
        status = "already installed" if "already installed" in lowered else "installed"
        if skill_name:
            return f"Completed.\n\nSkill '{skill_name}' {status}."
        return f"Completed.\n\nSkill {status}."
    return "Completed.\n\nSkill installation completed."


def _iter_status_evidence_lines(tool_outputs: list[str], *, limit: int = 5) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for output in reversed(tool_outputs):
        for raw_line in reversed(str(output or "").splitlines()):
            stripped = raw_line.strip()
            if not stripped:
                continue
            stripped = _OBSERVED_OUTPUT_PREFIX_RE.sub("", stripped).strip()
            if not stripped:
                continue
            lower = stripped.casefold()
            if lower.startswith("next:") or "observed output of cmd" in lower:
                continue
            if lower.startswith(("rule ", "rule_", "run $", "use ", "do not ")):
                continue
            if not any(marker in lower for marker in _STATUS_LINE_MARKERS):
                continue
            cleaned = _clean_user_visible_evidence_line(stripped)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            lines.append(cleaned)
            if len(lines) >= limit:
                return list(reversed(lines))
    return list(reversed(lines))


def _iter_error_evidence_lines(tool_outputs: list[str], *, limit: int = 3) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for output in reversed(tool_outputs):
        for raw_line in str(output or "").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            stripped = _OBSERVED_OUTPUT_PREFIX_RE.sub("", stripped).strip()
            if not stripped:
                continue
            lower = stripped.casefold()
            if lower.startswith(("traceback", "at ", "file \"", "{", "}", "node.js ")):
                continue
            if "no test specified" in lower and "exit 1" in lower:
                continue
            if not any(marker in lower for marker in _ERROR_LINE_MARKERS):
                continue
            cleaned = _clean_user_visible_evidence_line(stripped)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            lines.append(cleaned)
            if len(lines) >= limit:
                return lines
    return lines


def _iter_next_evidence_lines(tool_outputs: list[str], *, limit: int = 2) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for output in reversed(tool_outputs):
        for raw_line in reversed(str(output or "").splitlines()):
            stripped = raw_line.strip()
            if not stripped.casefold().startswith("next:"):
                continue
            next_step = stripped.partition(":")[2].strip()
            if not next_step:
                continue
            cleaned = _clean_user_visible_evidence_line(next_step)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            lines.append(cleaned)
            if len(lines) >= limit:
                return list(reversed(lines))
    return list(reversed(lines))


def build_user_facing_tool_evidence_answer(
    tool_outputs: list[str],
    *,
    incomplete: bool = False,
) -> str:
    """Build a concise final answer from tool evidence without raw transcripts."""
    summary_lines = _iter_user_summary_lines(tool_outputs)
    if summary_lines:
        return "Completed.\n\n" + summary_lines[-1]

    next_lines = _iter_next_evidence_lines(tool_outputs)
    if incomplete and next_lines:
        status_lines = _iter_status_evidence_lines(tool_outputs, limit=3)
        parts = ["The tool workflow stopped before a final answer."]
        if status_lines:
            parts.append(
                "Latest tool evidence:\n"
                + "\n".join(f"- {line}" for line in status_lines)
            )
        parts.append(
            "Pending next step:\n"
            + "\n".join(f"- {line}" for line in next_lines)
        )
        return "\n\n".join(parts)

    error_lines = _iter_error_evidence_lines(tool_outputs)
    if error_lines:
        bullets = "\n".join(f"- {line}" for line in error_lines)
        return "The tool workflow stopped before a final answer.\n\nLatest blocker:\n" + bullets

    status_lines = _iter_status_evidence_lines(tool_outputs)
    if status_lines:
        bullets = "\n".join(f"- {line}" for line in status_lines)
        if incomplete:
            return "The tool workflow stopped before a final answer.\n\nLatest tool evidence:\n" + bullets
        return "Completed from the latest tool evidence.\n\n" + bullets

    return (
        "The tool workflow stopped before a final answer. Internal tool details "
        "were suppressed because no concise user-facing result could be derived."
    )


def build_user_facing_tool_event_answer(
    tool_result_events: list[dict[str, Any]],
    *,
    incomplete: bool = False,
) -> str:
    """Build a user-facing evidence answer from stream tool result events."""
    return build_user_facing_tool_evidence_answer([
        _stream_tool_event_text(event)
        for event in tool_result_events
    ], incomplete=incomplete)
