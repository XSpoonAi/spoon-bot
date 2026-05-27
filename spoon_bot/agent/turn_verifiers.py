"""Turn-boundary verifier helpers.

These helpers keep workflow-specific completion checks out of the main agent
loop.  They operate on structured tool evidence instead of prompt routing.
"""

from __future__ import annotations

import json
import re
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

from spoon_bot.agent.request_hints import extract_shell_command_candidates

_LONG_0X_VALUE_RE = re.compile(r"\b0x[0-9a-fA-F]{64}\b")
_OBSERVED_OUTPUT_PREFIX_RE = re.compile(
    r"^(?:\[[^\]\n]{1,32}\]:\s*)?Observed output of cmd [^\n]* execution:\s*",
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
_ACTIONABLE_FOLLOWUP_EXECUTABLES = frozenset({
    "bash",
    "bun",
    "cargo",
    "cd",
    "curl",
    "deno",
    "docker",
    "git",
    "go",
    "make",
    "modal",
    "node",
    "npm",
    "npx",
    "pnpm",
    "poetry",
    "pytest",
    "python",
    "python3",
    "sh",
    "uv",
    "wget",
    "wrangler",
})
_PROSE_FOLLOWUP_STARTS = frozenset({
    "answer",
    "choose",
    "install",
    "print",
    "read",
    "replace",
    "select",
    "use",
    "wait",
})
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
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097f]")
_THAI_RE = re.compile(r"[\u0e00-\u0e7f]")
_BACKGROUND_JOB_ID_RE = re.compile(r"\bjob_id[:=]\s*([A-Za-z0-9_.:-]+)")
_BACKGROUND_POLL_GUARD_MARKER = "repeated background job polling suppressed"
_LOOP_GUARD_MARKERS = (
    "duplicate tool invocation suppressed",
    "repeated side-effecting tool series suppressed",
    _BACKGROUND_POLL_GUARD_MARKER,
    "duplicate action was skipped",
    "repeated external action was skipped",
)
_EVIDENCE_DENIAL_RE = re.compile(
    r"(?i)\b(?:"
    r"no evidence|"
    r"no steps (?:are )?shown|"
    r"not evidenced|"
    r"cannot confirm|"
    r"can't confirm|"
    r"could not confirm|"
    r"nothing .* evidence"
    r")\b|没有[^。.\n]{0,24}证据|无法确认|不能确认"
)
_INSTALL_COMPLETION_DENIAL_RE = re.compile(
    r"(?i)\b(?:"
    r"no evidence[^.\n]{0,80}(?:install|installed|dependencies)|"
    r"no explicit install(?:ation)?|"
    r"install(?:ation)? (?:step )?(?:was )?(?:not|never) (?:recorded|observed|shown|completed)|"
    r"skill (?:was )?(?:not|never) installed"
    r")\b|没有[^。.\n]{0,16}安装证据|未看到[^。.\n]{0,16}安装"
)
_BALANCE_ZERO_DENIAL_RE = re.compile(
    r"(?i)\b(?:"
    r"balance (?:is )?(?:zero|empty|missing)|"
    r"gas and gld balance is zero|"
    r"wallet (?:is )?(?:zero|empty|unfunded)"
    r")\b|余额[^。.\n]{0,16}(?:为零|为空|不足)"
)
_BALANCE_PAIR_RE = re.compile(
    r"\bGAS=(?P<gas>\d+(?:\.\d+)?)\s+GLD=(?P<gld>\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
_EVIDENCE_FACT_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9_:-]{0,32}=[^\s,.;)]+|"
    r"\b[A-Z]{2,}(?:_[A-Z0-9]+)*\b|"
    r"\b\d{2,}(?:/\d+)?\b"
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


def _is_actionable_tool_next_command(command: str) -> bool:
    """Return True for executable follow-up commands, not prose instructions."""
    normalized = " ".join(str(command or "").strip().split())
    if not normalized:
        return False
    lowered = normalized.casefold()
    if lowered.startswith(("read_file(", "write_file(", "replace only ")):
        return False

    words = normalized.split()
    first = words[0].strip("'\"`").casefold()
    if not first:
        return False
    if first in _PROSE_FOLLOWUP_STARTS:
        return False
    if first in _ACTIONABLE_FOLLOWUP_EXECUTABLES:
        return True
    if first.startswith(("./", "../", ".\\", "..\\", "/", "\\")):
        return True
    if "/" in first or "\\" in first:
        return True
    if re.search(r"\.(?:bat|cmd|cjs|exe|js|mjs|ps1|py|sh)\b", first):
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
            if (
                not command
                or command in seen
                or not _is_actionable_tool_next_command(command)
            ):
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
        if any(
            _is_actionable_tool_next_command(command)
            for command in extract_shell_command_candidates(text, limit=3)
        ):
            return True
        normalized = text.casefold()
        if (
            _iter_user_summary_lines([text])
            or any(marker in normalized for marker in _STATUS_LINE_MARKERS)
        ):
            return False
        continue
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
            if command and _is_actionable_tool_next_command(command):
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
    placeholder_event_index: int | None = None
    for index, event in enumerate(tool_result_events):
        text = _stream_tool_event_text(event)
        for command in extract_shell_command_candidates(text, limit=10):
            if (
                "<" in command
                and ">" in command
                and _is_actionable_tool_next_command(command)
            ):
                placeholder_event_index = index

    if placeholder_event_index is None:
        return False

    for event in tool_result_events[placeholder_event_index + 1 :]:
        text = _stream_tool_event_text(event)
        later_commands = [
            command
            for command in extract_shell_command_candidates(text, limit=3)
            if _is_actionable_tool_next_command(command)
        ]
        if any("<" not in command or ">" not in command for command in later_commands):
            return False
        if later_commands:
            continue
        normalized = text.casefold()
        if any(marker in normalized for marker in _STATUS_LINE_MARKERS):
            return False

    resolved_commands = extract_resolved_tool_next_commands(tool_result_events)
    if not resolved_commands:
        return False
    return True


def _clean_user_visible_evidence_line(line: str) -> str:
    cleaned = " ".join(str(line or "").strip().split())
    cleaned = _OBSERVED_OUTPUT_PREFIX_RE.sub("", cleaned).strip()
    cleaned = _LONG_0X_VALUE_RE.sub("<omitted 0x id>", cleaned)
    cleaned = cleaned.replace("0x***masked_private_key***", "<redacted 0x value>")
    cleaned = cleaned.replace("***masked_private_key***", "<redacted 0x value>")
    return cleaned


def _looks_like_structured_evidence_line(line: str) -> bool:
    stripped = str(line or "").strip()
    if not stripped or len(stripped) > 220:
        return False
    if stripped.casefold().startswith(_USER_SUMMARY_PREFIXES):
        return False
    if re.match(r"^[A-Za-z][A-Za-z0-9 _./-]{0,48}:\s+\S", stripped):
        if stripped.casefold().startswith(("q:", "question:")):
            return False
        return True
    if "=" in stripped and not stripped.casefold().startswith(("q:", "question:")):
        return True
    return False


def _response_language_instruction(user_message: str | None) -> str:
    if not isinstance(user_message, str) or not user_message.strip():
        return (
            "Required response language: infer it from the newest user request "
            "and do not copy the language of tool evidence by default."
        )

    text = user_message.strip()
    script_hints = [
        (_CJK_RE, "CJK characters"),
        (_HIRAGANA_KATAKANA_RE, "Japanese kana"),
        (_HANGUL_RE, "Hangul"),
        (_CYRILLIC_RE, "Cyrillic script"),
        (_ARABIC_RE, "Arabic script"),
        (_DEVANAGARI_RE, "Devanagari script"),
        (_THAI_RE, "Thai script"),
    ]
    for pattern, label in script_hints:
        if pattern.search(text):
            return (
                "Required response language: match the newest user request. "
                f"The newest request contains {label}, so answer in that language/script "
                "unless the user explicitly requested another language or a machine-readable format. "
                "Do not answer in English just because earlier requests or tool evidence are English."
            )

    return (
        "Required response language: match the newest user request as written. "
        "If the newest request is a short continuation, use that continuation's "
        "language instead of older requests or tool evidence."
    )


def _match_user_language_for_direct_evidence(line: str, user_message: str | None) -> str:
    if not isinstance(user_message, str) or not _CJK_RE.search(user_message):
        return line
    if _CJK_RE.search(line):
        return line
    return f"已完成：{line}"


def _user_message_contains_cjk(user_message: str | None) -> bool:
    return isinstance(user_message, str) and bool(_CJK_RE.search(user_message))


def _format_latest_evidence_answer(
    bullets: str,
    *,
    incomplete: bool,
    user_message: str | None,
) -> str:
    if _user_message_contains_cjk(user_message):
        if incomplete:
            return "我先停在当前状态。\n\n最新工具证据：\n" + bullets
        return "已完成。\n\n最新工具证据：\n" + bullets
    if incomplete:
        return "I paused before a final answer.\n\nLatest tool evidence:\n" + bullets
    return "Completed from the latest tool evidence.\n\n" + bullets


def _format_pending_next_answer(
    status_lines: list[str],
    next_lines: list[str],
    *,
    user_message: str | None,
) -> str:
    if _user_message_contains_cjk(user_message):
        parts = ["我先停在当前状态。"]
        if status_lines:
            parts.append(
                "最新工具证据：\n"
                + "\n".join(f"- {line}" for line in status_lines)
            )
        parts.append(
            "待执行的下一步：\n"
            + "\n".join(f"- {line}" for line in next_lines)
        )
        return "\n\n".join(parts)

    parts = ["I paused before a final answer."]
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


def _format_latest_blocker_answer(
    bullets: str,
    *,
    user_message: str | None,
) -> str:
    if _user_message_contains_cjk(user_message):
        return "我先停在当前状态。\n\n最新阻塞：\n" + bullets
    return "I paused before a final answer.\n\nLatest blocker:\n" + bullets


def _format_no_concise_result_answer(user_message: str | None) -> str:
    if _user_message_contains_cjk(user_message):
        return "我还无法从当前工具证据里整理出可靠的简短结果。"
    return (
        "I could not derive a concise user-facing result from the available tool evidence."
    )


def _tool_outputs_have_loop_guard(tool_outputs: list[str]) -> bool:
    for output in reversed(tool_outputs):
        lowered = str(output or "").casefold()
        if any(marker in lowered for marker in _LOOP_GUARD_MARKERS):
            return True
    return False


def _loop_guard_user_message(user_message: str | None) -> str:
    if isinstance(user_message, str) and _CJK_RE.search(user_message):
        return "已暂停：为避免重复执行同一类外部操作，我停在了当前安全状态。请继续时我会从最新状态接着处理。"
    return (
        "Paused at the current safe state to avoid repeating the same external "
        "action. Send a continuation request to proceed from the latest state."
    )


def _background_poll_guard_user_message(
    tool_outputs: list[str],
    user_message: str | None,
) -> str:
    job_id = _background_poll_guard_job_id(tool_outputs)
    if not job_id:
        return ""

    if isinstance(user_message, str) and _CJK_RE.search(user_message):
        return (
            "后台命令仍在运行，连续检查没有新的状态或输出。我已停止继续轮询，"
            f"避免一直调用。当前 job：{job_id}。"
        )
    return (
        "The background command is still running and repeated checks produced "
        f"no new status or output. I stopped polling to avoid a loop. Current job: {job_id}."
    )


def _background_poll_guard_job_id(tool_outputs: list[str]) -> str:
    for output in reversed(tool_outputs):
        text = str(output or "")
        lowered = text.casefold()
        if _BACKGROUND_POLL_GUARD_MARKER not in lowered:
            continue
        match = _BACKGROUND_JOB_ID_RE.search(text)
        if match:
            return match.group(1).strip()
    return ""


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


def _latest_user_summary_lines(tool_outputs: list[str]) -> list[str]:
    for output in reversed(tool_outputs):
        if not str(output or "").strip():
            continue
        return _iter_user_summary_lines([output])
    return []


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
        if isinstance(value, str) and _is_public_https_url(value):
            return value.strip()
        for nested in payload.values():
            nested_url = _public_url_from_payload(nested)
            if nested_url:
                return nested_url
    if isinstance(payload, list):
        for item in payload:
            nested_url = _public_url_from_payload(item)
            if nested_url:
                return nested_url
    return ""


def _is_public_https_url(value: str) -> bool:
    """Return True for a structured non-loopback HTTPS URL value."""
    candidate = str(value or "").strip()
    parsed = urlparse(candidate)
    if parsed.scheme != "https" or not parsed.hostname:
        return False

    hostname = parsed.hostname.casefold()
    if hostname in {"localhost", "0.0.0.0"} or hostname.endswith(".localhost"):
        return False

    try:
        parsed_ip = ip_address(hostname)
    except ValueError:
        return True
    return not (
        parsed_ip.is_loopback
        or parsed_ip.is_unspecified
        or parsed_ip.is_private
        or parsed_ip.is_link_local
    )


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
            if success is True and isinstance(public_url, str) and _is_public_https_url(public_url):
                return True
            nested_url = _public_url_from_payload(payload)
            if success is True and nested_url:
                return True
    return False


def _latest_balance_pair_is_nonzero(tool_outputs: list[str]) -> bool:
    """Return True when the latest observed GAS/GLD pair is funded."""
    for output in reversed(tool_outputs):
        matches = list(_BALANCE_PAIR_RE.finditer(str(output or "")))
        if not matches:
            continue
        latest = matches[-1]
        try:
            gas = float(latest.group("gas"))
            gld = float(latest.group("gld"))
        except Exception:
            return False
        return gas > 0 or gld > 0
    return False


def _tool_output_contains_recovery_status(output: str) -> bool:
    """Detect a later non-error status that supersedes older transient errors."""
    for raw_line in str(output or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        stripped = _OBSERVED_OUTPUT_PREFIX_RE.sub("", stripped).strip()
        if not stripped:
            continue
        lower = stripped.casefold()
        if lower.startswith(("error:", "stderr:", "exit code:")):
            continue
        if any(marker in lower for marker in ("success", "installed", "registered", "joined")):
            return True
        if "status=funded" in lower or "bound invitation code:" in lower:
            return True
        balance_match = _BALANCE_PAIR_RE.search(stripped)
        if balance_match:
            try:
                if float(balance_match.group("gas")) > 0 or float(balance_match.group("gld")) > 0:
                    return True
            except Exception:
                continue
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
        output_lines: list[str] = []
        for raw_line in str(output or "").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            stripped = _OBSERVED_OUTPUT_PREFIX_RE.sub("", stripped).strip()
            if not stripped:
                continue
            lower = stripped.casefold()
            if "stop_tool_loop" in lower:
                continue
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
            output_lines.append(cleaned)
        if output_lines:
            lines.extend(output_lines)
            if len(lines) >= limit:
                return lines[:limit]
        elif _tool_output_contains_recovery_status(output):
            return lines
    return lines


def _iter_next_evidence_lines(tool_outputs: list[str], *, limit: int = 2) -> list[str]:
    pending: list[str] = []
    for output in tool_outputs:
        for raw_line in str(output or "").splitlines():
            stripped = raw_line.strip()
            lower = stripped.casefold()
            if lower.startswith("next:"):
                next_step = stripped.partition(":")[2].strip()
                cleaned = _clean_user_visible_evidence_line(next_step)
                if cleaned:
                    pending.append(cleaned)
                continue
            if pending and (
                any(marker in lower for marker in _STATUS_LINE_MARKERS)
                or any(lower.startswith(prefix) for prefix in _USER_SUMMARY_PREFIXES)
            ):
                pending.clear()

    lines: list[str] = []
    seen: set[str] = set()
    for line in reversed(pending):
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= limit:
            break
    return list(reversed(lines))


def _iter_chronological_evidence_lines(
    tool_outputs: list[str],
    *,
    limit: int = 20,
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for output in tool_outputs:
        for raw_line in str(output or "").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            stripped = _OBSERVED_OUTPUT_PREFIX_RE.sub("", stripped).strip()
            if not stripped:
                continue
            lower = stripped.casefold()
            if "stop_tool_loop" in lower:
                continue
            if _PLACEHOLDER_RE.search(stripped):
                continue
            if lower.startswith("next:") or "observed output of cmd" in lower:
                continue
            if lower.startswith(("rule ", "rule_", "run $", "use ", "do not ")):
                continue
            if (
                any(marker in lower for marker in _STATUS_LINE_MARKERS)
                or _looks_like_structured_evidence_line(stripped)
            ):
                cleaned = _clean_user_visible_evidence_line(stripped)
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    lines.append(cleaned)

    if len(lines) <= limit:
        return lines
    return lines[-limit:]


def final_answer_denies_available_tool_evidence(
    final_content: str | None,
    tool_result_events: list[dict[str, Any]],
) -> bool:
    """Return True when a draft denies evidence that tool events actually contain."""
    if not isinstance(final_content, str) or not final_content.strip():
        return False
    if (
        _INSTALL_COMPLETION_DENIAL_RE.search(final_content)
        and tool_events_have_skill_install_completion(tool_result_events)
    ):
        return True
    tool_outputs = [_stream_tool_event_text(event) for event in tool_result_events]
    if (
        _BALANCE_ZERO_DENIAL_RE.search(final_content)
        and _latest_balance_pair_is_nonzero(tool_outputs)
    ):
        return True
    if not _EVIDENCE_DENIAL_RE.search(final_content):
        return False

    evidence_lines = (
        _iter_user_summary_lines(tool_outputs)
        + _iter_status_evidence_lines(tool_outputs, limit=8)
        + _iter_chronological_evidence_lines(tool_outputs, limit=12)
    )
    unique_lines = [line for index, line in enumerate(evidence_lines) if line not in evidence_lines[:index]]
    if len(unique_lines) < 2:
        return False

    lowered_final = final_content.casefold()
    fact_tokens: set[str] = set()
    for line in unique_lines:
        for match in _EVIDENCE_FACT_RE.finditer(line):
            token = match.group(0).strip(".,;:()[]{}").casefold()
            if len(token) >= 2:
                fact_tokens.add(token)
    if any(token and token in lowered_final for token in fact_tokens):
        return False

    return True


def build_user_facing_tool_evidence_answer(
    tool_outputs: list[str],
    *,
    incomplete: bool = False,
    user_message: str | None = None,
) -> str:
    """Build a concise final answer from tool evidence without raw transcripts."""
    summary_lines = _latest_user_summary_lines(tool_outputs)
    if summary_lines:
        return _match_user_language_for_direct_evidence(
            summary_lines[-1],
            user_message,
        )

    background_poll_message = _background_poll_guard_user_message(
        tool_outputs,
        user_message,
    )
    if background_poll_message:
        return background_poll_message

    next_lines = _iter_next_evidence_lines(tool_outputs)
    if incomplete and next_lines:
        status_lines = _iter_status_evidence_lines(tool_outputs, limit=3)
        return _format_pending_next_answer(
            status_lines,
            next_lines,
            user_message=user_message,
        )

    if _tool_outputs_have_loop_guard(tool_outputs):
        return _loop_guard_user_message(user_message)

    error_lines = _iter_error_evidence_lines(tool_outputs)
    if error_lines:
        bullets = "\n".join(f"- {line}" for line in error_lines)
        return _format_latest_blocker_answer(bullets, user_message=user_message)

    status_lines = _iter_status_evidence_lines(tool_outputs)
    if status_lines:
        bullets = "\n".join(f"- {line}" for line in status_lines)
        return _format_latest_evidence_answer(
            bullets,
            incomplete=incomplete,
            user_message=user_message,
        )

    return _format_no_concise_result_answer(user_message)


def build_tool_evidence_synthesis_brief(
    tool_outputs: list[str],
    *,
    incomplete: bool = False,
    user_message: str | None = None,
) -> str:
    """Build a compact evidence packet for model-written final-answer synthesis."""
    lines: list[str] = [
        "[FINAL ANSWER SYNTHESIS INPUT]",
        "Write the final answer as natural language for the newest user.",
        "Use only the verified evidence below; do not invent missing progress.",
        "Treat the newest user request as intent only, not as evidence of completed work.",
        "Do not claim clone/install/faucet/register/join steps completed unless they appear in evidence.",
        "If skill_marketplace reports the requested skill installed, do not deny skill installation just because no manual git clone appears.",
        "If a register command is documented as ERC-8004 registration and evidence says Registered AGENT_ID=<assigned id>, treat the assigned id as successful registration unless a tool explicitly accepts a user-supplied id argument.",
        "If the active skill contract exposes an auto-selection command such as join <spot>, do not say a game id is required solely because the user asked for the latest game; use verified join evidence if present or report that join is still pending.",
        "When newer evidence contradicts an older transient error or older state for the same workflow, trust the newer evidence and do not report the older state as a current blocker.",
        "Do not provide generic troubleshooting guides, manual install instructions, option menus, or invented example commands.",
        "Do not ask the user to provide optional next inputs or choose among options; if the verified workflow is done or blocked, state that status and stop.",
        "If settlement, final result, or equivalent terminal evidence exists, treat that as the outcome for this turn instead of inviting another run.",
        "If a next step is needed, use only exact NEXT commands or blockers found in the evidence.",
        "Match the newest user's language unless they requested a specific output format.",
        _response_language_instruction(user_message),
        "Do not expose raw JSON, internal guardrail markers, tool transcripts, or internal planning.",
        f"Workflow status: {'incomplete_or_paused' if incomplete else 'evidence_available'}",
    ]
    if isinstance(user_message, str) and user_message.strip():
        lines.extend(["", "Newest user request:", user_message.strip()])

    summary_lines = _latest_user_summary_lines(tool_outputs)
    if summary_lines:
        lines.extend(["", "Latest direct user-facing evidence:"])
        lines.extend(f"- {line}" for line in summary_lines)

    chronological_lines = _iter_chronological_evidence_lines(tool_outputs)
    if chronological_lines:
        lines.extend(["", "Verified chronological evidence:"])
        lines.extend(f"- {line}" for line in chronological_lines)

    background_message = _background_poll_guard_user_message(tool_outputs, None)
    if background_message:
        lines.extend(["", "Runtime guardrail state:"])
        lines.append(
            "- Background job polling stopped because repeated checks produced no new status or output."
        )
        job_id = _background_poll_guard_job_id(tool_outputs)
        if job_id:
            lines.append(f"- Current background job id: {job_id}")
    elif _tool_outputs_have_loop_guard(tool_outputs):
        lines.extend(["", "Runtime guardrail state:"])
        lines.append(
            "- A repeated tool/action loop was stopped; no additional duplicate action was run."
        )

    error_lines = _iter_error_evidence_lines(tool_outputs)
    if error_lines:
        lines.extend(["", "Latest blockers:"])
        lines.extend(f"- {line}" for line in error_lines[:3])

    status_lines = _iter_status_evidence_lines(tool_outputs)
    if status_lines:
        lines.extend(["", "Latest status evidence:"])
        lines.extend(f"- {line}" for line in status_lines[-5:])

    next_lines = _iter_next_evidence_lines(tool_outputs)
    if next_lines:
        lines.extend(["", "Pending next step evidence:"])
        lines.extend(f"- {line}" for line in next_lines[-2:])

    if len(lines) <= 8:
        fallback = build_user_facing_tool_evidence_answer(
            tool_outputs,
            incomplete=incomplete,
            user_message=user_message,
        )
        lines.extend(["", "Sanitized fallback evidence:", fallback])

    return "\n".join(lines)


def _iter_tool_event_milestone_lines(
    tool_result_events: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[str]:
    """Extract high-signal workflow milestones with tool identity preserved."""
    lines: list[str] = []
    seen: set[str] = set()
    for event in tool_result_events:
        tool_name = _stream_tool_event_name(event)
        text = _stream_tool_event_text(event)
        lowered = text.casefold()
        if tool_name == "skill_marketplace" and (
            ("success:" in lowered and "installed" in lowered)
            or "already installed" in lowered
            or "is already installed" in lowered
        ):
            skill_name = ""
            match = re.search(r"Skill\s+'([^']+)'", text)
            if match:
                skill_name = match.group(1).strip()
            line = (
                f"skill_marketplace installed skill {skill_name}."
                if skill_name
                else "skill_marketplace installed the requested skill."
            )
            if line not in seen:
                seen.add(line)
                lines.append(line)
            continue

        for raw_line in str(text or "").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            stripped = _OBSERVED_OUTPUT_PREFIX_RE.sub("", stripped).strip()
            if not stripped:
                continue
            lower = stripped.casefold()
            if lower.startswith(("next:", "q:", "question:", "read it aloud:")):
                continue
            keep = False
            if "skills reloaded" in lower or "installed at:" in lower:
                keep = True
            elif "status=funded" in lower or "bound invitation code:" in lower:
                keep = True
            elif _BALANCE_PAIR_RE.search(stripped):
                keep = True
            elif any(marker in lower for marker in ("registered agent_id=", "joined", "settlement", "final_state")):
                keep = True
            if not keep:
                continue
            cleaned = _clean_user_visible_evidence_line(stripped)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                lines.append(cleaned)
    if len(lines) <= limit:
        return lines
    return lines[-limit:]


def build_user_facing_tool_event_answer(
    tool_result_events: list[dict[str, Any]],
    *,
    incomplete: bool = False,
    user_message: str | None = None,
) -> str:
    """Build a user-facing evidence answer from stream tool result events."""
    return build_user_facing_tool_evidence_answer([
        _stream_tool_event_text(event)
        for event in tool_result_events
    ], incomplete=incomplete, user_message=user_message)


def build_tool_event_synthesis_brief(
    tool_result_events: list[dict[str, Any]],
    *,
    incomplete: bool = False,
    user_message: str | None = None,
) -> str:
    """Build a compact evidence packet from stream tool result events."""
    brief = build_tool_evidence_synthesis_brief([
        _stream_tool_event_text(event)
        for event in tool_result_events
    ], incomplete=incomplete, user_message=user_message)
    milestone_lines = _iter_tool_event_milestone_lines(tool_result_events)
    if not milestone_lines:
        return brief
    return (
        brief
        + "\n\nVerified tool milestones:\n"
        + "\n".join(f"- {line}" for line in milestone_lines)
    )
