"""Generic request hint helpers for the agent loop.

The agent loop should stay a state machine.  This module derives small,
request-scoped hints from the newest user message and the current skill
catalog without classifying product-specific routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse


_SHELL_COMMAND_PREFIXES = (
    "node ",
    "python ",
    "python3 ",
    "uv ",
    "bash ",
    "sh ",
    "curl ",
    "cast ",
    "git ",
    "npm ",
    "pnpm ",
    "yarn ",
)
_EXACT_COMMAND_ONLY_MARKERS = (
    "只执行",
    "仅执行",
    "只运行",
    "仅运行",
    "only run",
    "only execute",
    "run only",
    "execute only",
    "do not run other",
    "do not execute other",
    "do not add other",
    "不要运行其他",
    "不要执行其他",
    "不要额外",
)
_REMOTE_LOOKUP_TOKENS = {
    "web_fetch",
    "web_search",
    "curl",
    "http",
    "https",
    "api",
    "doc",
    "docs",
    "documentation",
    "search",
    "browse",
    "lookup",
    "fetch",
}
_REMOTE_LOOKUP_PHRASES = ("look up", "官网", "文档", "接口", "网页", "搜索")


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


def request_explicitly_needs_remote_lookup(message: str) -> bool:
    """Return True when the latest request clearly asks for web/API lookup."""
    text = str(message or "").casefold()
    if "http://" in text or "https://" in text:
        return True
    tokens = tokenize_request_matching_text(text)
    return bool(tokens & _REMOTE_LOOKUP_TOKENS) or any(
        phrase in text for phrase in _REMOTE_LOOKUP_PHRASES
    )


def extract_exact_shell_commands_from_request(message: str) -> list[str]:
    """Extract shell commands that the user explicitly wrote in the request."""
    commands: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        candidate = " ".join(str(raw or "").strip().split())
        candidate = candidate.strip("`").rstrip("。；;")
        if not candidate:
            return
        if not candidate.casefold().startswith(_SHELL_COMMAND_PREFIXES):
            return
        if candidate in seen:
            return
        seen.add(candidate)
        commands.append(candidate)

    inline_parts = str(message or "").split("`")
    for index in range(1, len(inline_parts), 2):
        candidate = inline_parts[index]
        if 3 <= len(candidate) <= 300 and "\n" not in candidate:
            _add(candidate)

    for line in str(message or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[:2] in {"- ", "* ", "+ "}:
            stripped = stripped[2:].strip()
        else:
            digit_end = 0
            while digit_end < len(stripped) and stripped[digit_end].isdigit():
                digit_end += 1
            if (
                digit_end
                and digit_end < len(stripped)
                and stripped[digit_end] in {".", ")"}
            ):
                stripped = stripped[digit_end + 1:].strip()
        _add(stripped)
    return commands[:4]


def request_restricts_to_exact_shell_commands(
    message: str,
    exact_commands: list[str],
) -> bool:
    """Return True when the newest request says only those commands should run."""
    if not exact_commands:
        return False
    text = str(message or "").casefold()
    return any(marker in text for marker in _EXACT_COMMAND_ONLY_MARKERS)


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

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("$CLI "):
            expanded = stripped.replace("$CLI", cli_value or "$CLI", 1)
            _push_command(expanded)
            continue
        if stripped.casefold().startswith(_SHELL_COMMAND_PREFIXES):
            _push_command(stripped)

    urls: list[str] = []
    seen_urls: set[str] = set()
    for raw in content.replace("\n", " ").split():
        candidate = raw.strip("`'\"<>()[]{}").rstrip(".,;:，。；")
        parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if candidate in seen_urls:
            continue
        seen_urls.add(candidate)
        urls.append(candidate)
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
        "allow_remote_probe": request_explicitly_needs_remote_lookup(message_text),
        "exact_shell_commands": exact_commands,
        "restrict_to_exact_shell_commands": request_restricts_to_exact_shell_commands(
            message_text,
            exact_commands,
        ),
        "local_executable_skills": local_executable_skills[:3],
    }
