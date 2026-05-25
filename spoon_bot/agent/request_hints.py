"""Generic request hint helpers for the agent loop.

The agent loop should stay a state machine.  This module derives small,
request-scoped hints from the newest user message and the current skill
catalog without classifying product-specific routes.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_EXECUTABLE_SUFFIXES = tuple(
    suffix.casefold()
    for suffix in (os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD;.PS1;.SH")
    .replace(";", os.pathsep)
    .split(os.pathsep)
    if suffix
)


def tokenize_request_matching_text(text: str) -> set[str]:
    """Return normalized ASCII tokens for lightweight catalog matching."""
    tokens: set[str] = set()
    buffer: list[str] = []
    for char in str(text or "").casefold():
        if char.isascii() and (char.isalnum() or char in {"_", "-"}):
            buffer.append(char)
            continue
        if len(buffer) >= 2 and buffer[0].isalnum():
            tokens.add("".join(buffer))
        buffer = []
    if len(buffer) >= 2 and buffer[0].isalnum():
        tokens.add("".join(buffer))
    return tokens


def extract_urls_from_text(text: str) -> list[str]:
    """Extract explicit HTTP(S) URLs without classifying request intent."""
    urls: list[str] = []
    seen: set[str] = set()
    for raw in str(text or "").replace("\n", " ").split():
        candidate = raw.strip("`'\"<>()[]{}").rstrip(".,;:，。；")
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        urls.append(candidate)
    return urls[:12]


def _clean_extracted_value(value: str) -> str:
    return str(value or "").strip("`'\"<>()[]{}").rstrip(".,;:，。；")


def _looks_like_user_supplied_value(value: str) -> bool:
    if not 3 <= len(value) <= 80:
        return False
    return all(char.isalnum() or char in {"_", "-"} for char in value)


def _last_label_fragment(label: str) -> str:
    fragment = str(label or "").strip()
    for separator in (".", "。", ";", "；", "\n", "\r"):
        if separator in fragment:
            fragment = fragment.rsplit(separator, 1)[-1].strip()
    return fragment


def extract_explicit_request_values_from_text(text: str) -> list[dict[str, Any]]:
    """Extract user-labeled values from simple ``label: value`` request facts."""
    values: list[dict[str, Any]] = []
    seen: set[str] = set()
    source = str(text or "")
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for separator in (":", "：", "="):
            if separator not in line:
                continue
            label, _, tail = line.partition(separator)
            label = _last_label_fragment(label)
            labels = sorted(tokenize_request_matching_text(label))
            if not labels:
                break
            tail_words = tail.strip().split()
            if not tail_words:
                break
            value = _clean_extracted_value(tail_words[0])
            if not _looks_like_user_supplied_value(value):
                break
            key = value.casefold()
            if key in seen:
                break
            seen.add(key)
            values.append({
                "value": value,
                "labels": labels,
                "label": label,
            })
            break
    return values[:12]


def _split_shell_words(command: str) -> list[str]:
    try:
        return shlex.split(str(command or ""), posix=True)
    except ValueError:
        return str(command or "").split()


def _strip_shell_prompt(line: str) -> str:
    stripped = str(line or "").strip()
    if not stripped:
        return ""
    if stripped[:2] in {"$ ", "> "}:
        return stripped[2:].strip()
    if stripped.casefold().startswith("ps"):
        marker = stripped.find(">")
        if marker >= 0:
            return stripped[marker + 1 :].strip()
    return stripped


def _strip_list_marker(line: str) -> str:
    stripped = str(line or "").strip()
    if stripped[:2] in {"- ", "* ", "+ "}:
        return stripped[2:].strip()
    digit_end = 0
    while digit_end < len(stripped) and stripped[digit_end].isdigit():
        digit_end += 1
    if (
        digit_end
        and digit_end < len(stripped)
        and stripped[digit_end] in {".", ")"}
    ):
        return stripped[digit_end + 1 :].strip()
    return stripped


def _looks_like_executable_token(token: str) -> bool:
    token = str(token or "").strip().strip("'\"")
    if not token:
        return False
    if urlparse(token).scheme in {"http", "https"} or "://" in token:
        return False
    if token.startswith(("./", "../", ".\\", "..\\", "/", "\\")):
        return True
    if "/" in token or "\\" in token:
        return True
    lowered = token.casefold()
    if _EXECUTABLE_SUFFIXES and lowered.endswith(_EXECUTABLE_SUFFIXES):
        return True
    if shutil.which(token):
        return True
    return bool(
        len(token) >= 2
        and token == token.casefold()
        and all(char.isalnum() or char in {"_", "-", ".", "+"} for char in token)
    )


def _looks_like_shell_command(candidate: str, *, explicit_shell_context: bool) -> bool:
    command = " ".join(str(candidate or "").strip().split())
    if not command or "\n" in command:
        return False
    words = _split_shell_words(command)
    if not words:
        return False

    first = words[0]
    if not _looks_like_executable_token(first):
        return False

    if explicit_shell_context:
        return True

    if shutil.which(first):
        return True
    if first.startswith(("./", "../", ".\\", "..\\", "/", "\\")):
        return True
    if "/" in first or "\\" in first:
        return True
    lowered = first.casefold()
    return bool(_EXECUTABLE_SUFFIXES and lowered.endswith(_EXECUTABLE_SUFFIXES))


def _iter_structural_command_candidates(text: str):
    in_fence = False
    fence_marker = ""
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if in_fence and marker == fence_marker:
                in_fence = False
                fence_marker = ""
            elif not in_fence:
                in_fence = True
                fence_marker = marker
            continue

        if in_fence:
            yield _strip_shell_prompt(stripped), True
            continue

        line = _strip_shell_prompt(_strip_list_marker(stripped))
        explicit_prompt = line != _strip_list_marker(stripped)
        yield line, explicit_prompt

    inline_parts = str(text or "").split("`")
    for index in range(1, len(inline_parts), 2):
        candidate = inline_parts[index]
        if "\n" not in candidate:
            yield candidate, True


def _iter_embedded_command_candidates(line: str):
    """Yield colon-delimited suffixes that may contain an executable command."""
    stripped = " ".join(str(line or "").strip().split())
    if not stripped:
        return
    yield stripped
    search = stripped
    while ":" in search:
        _, _, tail = search.partition(":")
        candidate = tail.strip()
        if candidate:
            yield candidate
        search = candidate


def extract_shell_command_candidates(text: str, *, limit: int = 8) -> list[str]:
    """Extract shell-like command candidates from free-form text generically."""
    commands: list[str] = []
    seen: set[str] = set()

    def _add(raw: str, *, explicit: bool) -> None:
        candidate = " ".join(str(raw or "").strip().split())
        candidate = candidate.strip("`").rstrip("。；;")
        if not candidate:
            return
        if not _looks_like_shell_command(candidate, explicit_shell_context=explicit):
            return
        if candidate in seen:
            return
        seen.add(candidate)
        commands.append(candidate)

    for candidate, explicit in _iter_structural_command_candidates(text):
        if 3 <= len(candidate) <= 400:
            _add(candidate, explicit=explicit)
        for embedded in _iter_embedded_command_candidates(candidate):
            if embedded == candidate:
                continue
            if 3 <= len(embedded) <= 400:
                _add(embedded, explicit=True)
        if len(commands) >= limit:
            break
    return commands[:limit]


def extract_exact_shell_commands_from_request(message: str) -> list[str]:
    """Extract shell commands that the user explicitly wrote in the request."""
    return extract_shell_command_candidates(message, limit=4)


def request_restricts_to_exact_shell_commands(
    message: str,
    exact_commands: list[str],
) -> bool:
    """Return True when the request consists only of shell command content."""
    if not exact_commands:
        return False
    remaining = str(message or "")
    for command in exact_commands:
        remaining = remaining.replace(command, " ")
    leftovers = tokenize_request_matching_text(remaining)
    return not leftovers


def request_needs_current_session_fact_check(message: str) -> bool:
    """Return True when the user is asking about this conversation's prior actions."""
    text = str(message or "").casefold()
    if not text.strip():
        return False

    tokens = tokenize_request_matching_text(text)
    has_actor_cue = (
        "你" in text
        or bool(tokens & {"you", "assistant", "agent", "bot"})
    )
    temporal_or_dispute_cues = (
        "刚刚",
        "刚才",
        "刚",
        "之前",
        "前面",
        "上面",
        "忘记",
        "不是",
        "previous",
        "earlier",
        "before",
        "last turn",
        "just did",
        "you said",
        "forgot",
        "contradict",
        "what happened",
        "what did",
    )
    return has_actor_cue and any(cue in text for cue in temporal_or_dispute_cues)


def format_current_session_fact_check_context(message: str) -> str:
    """Format the fact-check boundary for prior-action follow-up requests."""
    if not request_needs_current_session_fact_check(message):
        return ""
    return (
        "[CURRENT SESSION FACT CHECK REQUIRED]: This request asks about prior "
        "actions/results in the current conversation. First use "
        "search_history(scope='current') for exact same-session user/tool facts. "
        "`search_history` is an active tool in the default tool set; call it "
        "directly and do not search the filesystem for it. Do not claim it is "
        "unavailable unless an actual tool call returns that error. "
        "Do not use long-term memory as a substitute for the session transcript. "
        "External or live-state tools may be used afterward for current state, "
        "but they do not prove what happened earlier unless the matching "
        "tool call/result is found in current-session history.\n"
    )


def format_explicit_request_values_context(message: str) -> str:
    """Format user-labeled request facts without classifying their meaning."""
    values = extract_explicit_request_values_from_text(message)
    if not values:
        return ""
    lines = ["[STRUCTURED USER REQUEST FACTS]:"]
    for entry in values[:8]:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip()
        value = str(entry.get("value") or "").strip()
        if not label or not value:
            continue
        lines.append(f"- {label}: {value}")
    if len(lines) == 1:
        return ""
    lines.append("")
    return "\n".join(lines)


def format_local_executable_skill_context(hints: dict[str, Any]) -> str:
    """Format request-scoped guidance for matched local executable skills."""
    local_skills = hints.get("local_executable_skills") if isinstance(hints, dict) else None
    if not isinstance(local_skills, list) or not local_skills:
        return ""

    lines = [
        "[LOCAL SKILL EXECUTION CONTEXT]: The newest request matches installed "
        "local skill procedures. Candidate SKILL.md files and extracted command "
        "forms are listed below as request-scoped context. Treat the listed "
        "metadata as evidence, not a hidden route.",
    ]
    for skill in local_skills[:3]:
        if not isinstance(skill, dict):
            continue
        name = str(skill.get("name") or "local skill")
        location = str(skill.get("location") or "").strip()
        commands = skill.get("commands")
        command_text = ""
        if isinstance(commands, list) and commands:
            compact_commands = [
                " ".join(str(command or "").strip().split())
                for command in commands[:16]
                if str(command or "").strip()
            ]
            if compact_commands:
                command_text = "; commands: " + " | ".join(compact_commands)
        location_text = f" at {location}" if location else ""
        lines.append(f"- {name}{location_text}{command_text}")
    lines.append("")
    return "\n".join(lines)


def format_exact_shell_command_context(message: str) -> str:
    """Format exact command constraints for the active request prompt."""
    exact_commands = extract_exact_shell_commands_from_request(message)
    if not exact_commands:
        return ""
    lines = ["[EXACT SHELL COMMANDS REQUESTED]:"]
    lines.extend(
        f"{index}. {command}"
        for index, command in enumerate(exact_commands, start=1)
    )
    if request_restricts_to_exact_shell_commands(message, exact_commands):
        lines.append(
            "Restriction: the user asked to run only these shell commands; "
            "do not substitute or add other shell commands."
        )
    else:
        lines.append(
            "Run exact commands as written before trying nearby variants; "
            "do not treat them as loose examples."
        )
    lines.append("")
    return "\n".join(lines)


def extract_skill_command_hints(skill_md: Path) -> tuple[list[str], list[str]]:
    """Extract runnable command hints and referenced URLs from a SKILL.md file."""
    try:
        content = skill_md.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return [], []

    commands: list[str] = []
    seen_commands: set[str] = set()

    cli_value = ""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped[:3].casefold() == "cli":
            continue
        rest = stripped[3:].lstrip()
        if rest.startswith(":"):
            rest = rest[1:].lstrip()
        if not rest.startswith("="):
            continue
        cli_value = rest[1:].strip()
        break

    def _push_command(command: str) -> None:
        normalized = " ".join(str(command or "").strip().split())
        if not normalized or normalized in seen_commands:
            return
        seen_commands.add(normalized)
        commands.append(normalized)

    def _normalize_skill_command_hint(candidate: str) -> str:
        stripped = str(candidate or "").strip()
        if not stripped:
            return ""
        if stripped.casefold().startswith("run "):
            if re.search(r"\bagain\b", stripped, re.IGNORECASE):
                return ""
            stripped = stripped[4:].strip()
        return stripped

    if cli_value:
        _push_command(cli_value)

    def _markdown_heading_text(line: str) -> str | None:
        stripped_line = line.strip()
        level = 0
        while level < len(stripped_line) and stripped_line[level] == "#":
            level += 1
        if not 1 <= level <= 6:
            return None
        if level >= len(stripped_line) or stripped_line[level] not in {" ", "\t"}:
            return None
        return stripped_line[level:].strip()

    command_section: list[str] = []
    in_commands_section = False
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        heading = _markdown_heading_text(stripped)
        if heading and heading.casefold().split()[:1] == ["commands"]:
            in_commands_section = True
            continue
        if in_commands_section and heading:
            break
        if in_commands_section:
            command_section.append(raw_line)

    scan_texts: list[str] = []
    if command_section:
        explicit_run_lines = [
            raw_line
            for raw_line in content.splitlines()
            if _strip_list_marker(raw_line.strip()).casefold().startswith("run ")
        ]
        if explicit_run_lines:
            scan_texts.append("\n".join(explicit_run_lines))
        scan_texts.append("\n".join(command_section))
    else:
        scan_texts.append(content)

    for scan_text in scan_texts:
        for candidate, explicit in _iter_structural_command_candidates(scan_text):
            stripped = _normalize_skill_command_hint(candidate)
            if not stripped:
                continue
            if stripped.startswith("$CLI "):
                expanded = stripped.replace("$CLI", cli_value or "$CLI", 1)
                _push_command(expanded)
                continue
            if _looks_like_shell_command(stripped, explicit_shell_context=explicit):
                _push_command(stripped)

    urls = extract_urls_from_text(content)
    return commands[:16], urls[:12]


def build_request_execution_hints(
    message: str,
    skill_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build generic execution hints from request text and skill metadata."""
    message_text = str(message or "")
    request_tokens = tokenize_request_matching_text(message_text)
    local_executable_skills: list[dict[str, Any]] = []

    for candidate in skill_candidates:
        name = str(candidate.get("name") or "")
        if not name:
            continue
        name_tokens = tokenize_request_matching_text(name.replace("-", " "))
        desc_tokens = tokenize_request_matching_text(
            " ".join(
                part
                for part in (
                    candidate.get("description") or "",
                    candidate.get("when_to_use") or "",
                )
                if part
            )
        )
        overlap = len(request_tokens & (name_tokens | desc_tokens))
        direct_name_match = name.casefold() in message_text.casefold()
        distinctive_name_match = any(
            len(token) >= 4 and token in name_tokens
            for token in request_tokens
        )
        if not direct_name_match and overlap < 2 and not distinctive_name_match:
            continue

        skill_md = candidate.get("skill_md")
        if not isinstance(skill_md, Path):
            continue
        commands, urls = extract_skill_command_hints(skill_md)
        if not commands:
            continue

        is_organized = bool(candidate.get("is_organized"))
        skill_rel = f"skills/{name}/SKILL.md" if is_organized else f"{name}/SKILL.md"
        local_executable_skills.append({
            "name": name,
            "location": skill_rel,
            "commands": commands,
            "urls": urls,
            "score": (
                (100 if direct_name_match else 0)
                + (25 if distinctive_name_match else 0)
                + overlap
            ),
        })

    local_executable_skills.sort(
        key=lambda item: int(item.get("score") or 0),
        reverse=True,
    )
    explicit_urls = extract_urls_from_text(message_text)
    explicit_request_values = extract_explicit_request_values_from_text(message_text)
    exact_commands = extract_exact_shell_commands_from_request(message_text)
    return {
        "explicit_request_urls": explicit_urls,
        "explicit_request_values": explicit_request_values,
        "exact_shell_commands": exact_commands,
        "restrict_to_exact_shell_commands": request_restricts_to_exact_shell_commands(
            message_text,
            exact_commands,
        ),
        "current_session_fact_check_required": (
            request_needs_current_session_fact_check(message_text)
        ),
        "local_executable_skills": local_executable_skills[:3],
    }
