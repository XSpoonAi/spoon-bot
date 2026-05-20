"""Generic request hint helpers for the agent loop.

The agent loop should stay a state machine.  This module derives small,
request-scoped hints from the newest user message and the current skill
catalog without classifying product-specific routes.
"""

from __future__ import annotations

import os
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


def extract_exact_shell_commands_from_request(message: str) -> list[str]:
    """Extract shell commands that the user explicitly wrote in the request."""
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

    for candidate, explicit in _iter_structural_command_candidates(message):
        if 3 <= len(candidate) <= 300:
            _add(candidate, explicit=explicit)
    return commands[:4]


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

    if cli_value:
        _push_command(cli_value)

    for candidate, explicit in _iter_structural_command_candidates(content):
        stripped = candidate.strip()
        if not stripped:
            continue
        if stripped.startswith("$CLI "):
            expanded = stripped.replace("$CLI", cli_value or "$CLI", 1)
            _push_command(expanded)
            continue
        if _looks_like_shell_command(stripped, explicit_shell_context=explicit):
            _push_command(stripped)

    urls = extract_urls_from_text(content)
    return commands[:8], urls[:12]


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
        if not direct_name_match and overlap < 2:
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
            "score": (100 if direct_name_match else 0) + overlap,
        })

    local_executable_skills.sort(
        key=lambda item: int(item.get("score") or 0),
        reverse=True,
    )
    exact_commands = extract_exact_shell_commands_from_request(message_text)
    return {
        "explicit_request_urls": extract_urls_from_text(message_text),
        "exact_shell_commands": exact_commands,
        "restrict_to_exact_shell_commands": request_restricts_to_exact_shell_commands(
            message_text,
            exact_commands,
        ),
        "local_executable_skills": local_executable_skills[:3],
    }
