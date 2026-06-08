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
_COMMAND_INTRO_TRAILING_WORDS = frozenset({
    "call",
    "cmd",
    "command",
    "exec",
    "execute",
    "invoke",
    "run",
    "try",
})
_EXPLICIT_TOOL_MARKERS = frozenset({"mcp", "tool", "tools"})
_EXPLICIT_TOOL_IGNORED_TOKENS = frozenset({
    "a",
    "an",
    "and",
    "by",
    "call",
    "check",
    "for",
    "from",
    "get",
    "in",
    "invoke",
    "run",
    "the",
    "to",
    "use",
    "using",
    "with",
})
_ASCII_CONTINUATION_TOKENS = frozenset({
    "again",
    "continue",
    "go",
    "going",
    "it",
    "keep",
    "previous",
    "resume",
    "retry",
    "same",
    "that",
})


def _safe_url_scheme(value: str) -> str:
    try:
        return urlparse(value).scheme
    except ValueError:
        return ""


def _looks_urlish(value: str) -> bool:
    token = str(value or "").strip()
    if "://" in token:
        return True
    return _safe_url_scheme(token) in {"http", "https"}


def tokenize_request_matching_text(text: str) -> set[str]:
    """Return normalized tokens for lightweight catalog/label matching."""
    tokens: set[str] = set()
    buffer: list[str] = []
    buffer_is_ascii: bool | None = None
    for char in str(text or "").casefold():
        if char.isalnum() or char == "_":
            char_is_ascii = char.isascii()
            if buffer and buffer_is_ascii is not None and char_is_ascii != buffer_is_ascii:
                if len(buffer) >= 2 and buffer[0].isalnum():
                    tokens.add("".join(buffer))
                buffer = []
            buffer.append(char)
            buffer_is_ascii = char_is_ascii
            continue
        if len(buffer) >= 2 and buffer[0].isalnum():
            tokens.add("".join(buffer))
        buffer = []
        buffer_is_ascii = None
    if len(buffer) >= 2 and buffer[0].isalnum():
        tokens.add("".join(buffer))
    return tokens


def ordered_request_matching_tokens(text: str) -> list[str]:
    """Return request tokens in order using the same normalization as matching."""
    tokens: list[str] = []
    buffer: list[str] = []
    buffer_is_ascii: bool | None = None
    for char in str(text or "").casefold():
        if char.isalnum() or char == "_":
            char_is_ascii = char.isascii()
            if buffer and buffer_is_ascii is not None and char_is_ascii != buffer_is_ascii:
                if len(buffer) >= 2 and buffer[0].isalnum():
                    tokens.append("".join(buffer))
                buffer = []
            buffer.append(char)
            buffer_is_ascii = char_is_ascii
            continue
        if len(buffer) >= 2 and buffer[0].isalnum():
            tokens.append("".join(buffer))
        buffer = []
        buffer_is_ascii = None
    if len(buffer) >= 2 and buffer[0].isalnum():
        tokens.append("".join(buffer))
    return tokens


def request_is_bare_continuation(text: str) -> bool:
    """Return True for short messages that only ask to continue prior work."""
    value = str(text or "").strip()
    if not value:
        return False
    if extract_urls_from_text(value):
        return False
    if extract_exact_shell_commands_from_request(value):
        return False
    tokens = ordered_request_matching_tokens(value)
    if not tokens:
        return len(value) <= 12
    if len(tokens) > 3:
        return False
    ascii_tokens = [token for token in tokens if token.isascii()]
    if ascii_tokens:
        return all(token in _ASCII_CONTINUATION_TOKENS for token in ascii_tokens)
    return len(value) <= 12


def _normalize_tool_identifier(value: str) -> str:
    normalized = []
    for char in str(value or "").casefold():
        if char.isalnum():
            normalized.append(char)
    return "".join(normalized)


def _is_explicit_tool_name_candidate(token: str) -> bool:
    value = str(token or "").casefold()
    if len(value) < 2:
        return False
    if value in _EXPLICIT_TOOL_MARKERS or value in _EXPLICIT_TOOL_IGNORED_TOKENS:
        return False
    return any(char.isalnum() for char in value)


def extract_explicit_tool_requests_from_text(
    text: str,
    available_tool_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract explicit tool/MCP name requests and mark unavailable names.

    This is tool-catalog matching, not task routing.  It only applies when the
    user names a capability adjacent to a generic marker such as "MCP" or
    "tool"; if that named capability is unavailable, other tools should not be
    used as a substitute.
    """
    tokens = ordered_request_matching_tokens(text)
    if not tokens or not any(token in _EXPLICIT_TOOL_MARKERS for token in tokens):
        return []

    available = {
        _normalize_tool_identifier(name)
        for name in (available_tool_names or [])
        if str(name or "").strip()
    }
    requests: list[dict[str, Any]] = []
    seen: set[str] = set()

    for index, token in enumerate(tokens):
        if token not in _EXPLICIT_TOOL_MARKERS:
            continue

        candidates: list[str] = []
        if index > 0:
            candidates.append(tokens[index - 1])
        if index + 1 < len(tokens):
            candidates.append(tokens[index + 1])

        for candidate in candidates:
            if not _is_explicit_tool_name_candidate(candidate):
                continue
            normalized = _normalize_tool_identifier(candidate)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            requests.append({
                "name": candidate,
                "available": normalized in available,
            })

    return requests[:8]


def extract_urls_from_text(text: str) -> list[str]:
    """Extract explicit HTTP(S) URLs without classifying request intent."""
    urls: list[str] = []
    seen: set[str] = set()
    for raw in str(text or "").replace("\n", " ").split():
        candidate = raw.strip("`'\"<>()[]{}").rstrip(".,;:，。；")
        try:
            parsed = urlparse(candidate)
        except ValueError:
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        urls.append(candidate)
    return urls[:12]


def _remove_explicit_urls_from_text(text: str, urls: list[str]) -> str:
    scrubbed = str(text or "")
    for url in urls:
        scrubbed = scrubbed.replace(url, " ")
    return scrubbed


def _iter_request_value_token_spans(text: str):
    """Yield request value-like token spans without assuming an English grammar."""
    source = str(text or "")
    buffer: list[str] = []
    start_index = 0
    buffer_is_ascii: bool | None = None

    def flush(end_index: int):
        nonlocal buffer, start_index, buffer_is_ascii
        if not buffer:
            return None
        raw = "".join(buffer)
        value = raw.strip("_-")
        item = (value, start_index, end_index) if value else None
        buffer = []
        buffer_is_ascii = None
        return item

    for index, char in enumerate(source):
        if char.isalnum():
            char_is_ascii = char.isascii()
            if buffer and buffer_is_ascii is not None and char_is_ascii != buffer_is_ascii:
                item = flush(index)
                if item is not None:
                    yield item
                start_index = index
            elif not buffer:
                start_index = index
            buffer.append(char)
            buffer_is_ascii = char_is_ascii
            continue
        if char in {"_", "-"} and buffer:
            buffer.append(char)
            continue
        item = flush(index)
        if item is not None:
            yield item

    item = flush(len(source))
    if item is not None:
        yield item


def _ordered_label_tokens(text: str) -> list[str]:
    """Return request-match tokens in source order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for token, _start, _end in _iter_request_value_token_spans(text):
        for normalized in sorted(tokenize_request_matching_text(token)):
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _nearby_value_labels(
    source: str,
    *,
    start: int,
    end: int,
    value: str,
) -> tuple[list[str], str]:
    """Infer labels from text near a user-provided value.

    This is intentionally structural rather than intent-specific: a value only
    becomes actionable later when a tool schema or SKILL.md command documents a
    matching parameter label.
    """
    line_start = source.rfind("\n", 0, start) + 1
    line_end = source.find("\n", end)
    if line_end < 0:
        line_end = len(source)
    line = source[line_start:line_end]
    local_start = start - line_start
    local_end = end - line_start
    spans = list(_iter_request_value_token_spans(line))
    value_index = -1
    for index, (token, token_start, token_end) in enumerate(spans):
        if token_start == local_start and token_end == local_end:
            value_index = index
            break
        if token == value and token_start <= local_start <= token_end:
            value_index = index
            break
    if value_index < 0:
        context = f"{line[:local_start]} {line[local_end:]}"
    else:
        def _bounded_side(direction: int, *, limit: int) -> list[str]:
            selected: list[tuple[str, int, int]] = []
            cursor = value_index + direction
            previous_start = spans[value_index][1]
            previous_end = spans[value_index][2]
            while 0 <= cursor < len(spans) and len(selected) < limit:
                token, token_start, token_end = spans[cursor]
                if direction > 0:
                    gap = line[previous_end:token_start]
                else:
                    gap = line[token_end:previous_start]
                if any(char in gap for char in ".。!！?？,，;；:：\n\r"):
                    break
                if _looks_like_user_supplied_value(token, allow_plain_alpha=False):
                    break
                selected.append((token, token_start, token_end))
                previous_start = token_start
                previous_end = token_end
                cursor += direction
            if direction < 0:
                selected.reverse()
            return [token for token, _token_start, _token_end in selected]

        after = _bounded_side(1, limit=2)
        before = _bounded_side(-1, limit=4)
        context = " ".join(after or before)
    value_tokens = tokenize_request_matching_text(value)
    labels = sorted({
        token
        for token in _ordered_label_tokens(context)
        if token not in value_tokens
        and not _looks_like_user_supplied_value(token, allow_plain_alpha=False)
    })[:12]
    return labels, " ".join(str(context or "").split()[:6])


def _clean_extracted_value(value: str) -> str:
    return str(value or "").strip("`'\"<>()[]{}").rstrip(".,;:，。；")


def _last_label_fragment(label: str) -> str:
    fragment = str(label or "").strip()
    for separator in (".", "。", ";", "；", "\n", "\r"):
        if separator in fragment:
            fragment = fragment.rsplit(separator, 1)[-1].strip()
    return fragment


def _append_explicit_request_value(
    values: list[dict[str, Any]],
    seen: set[str],
    *,
    label: str,
    value: str,
    allow_plain_alpha: bool = False,
) -> None:
    label = _last_label_fragment(label)
    labels = sorted(tokenize_request_matching_text(label))
    value = _clean_extracted_value(value)
    if (
        not labels
        or not _looks_like_user_supplied_value(
            value,
            allow_plain_alpha=allow_plain_alpha,
        )
    ):
        return
    key = value.casefold()
    if key in seen:
        return
    seen.add(key)
    values.append({
        "value": value,
        "labels": labels,
        "label": label,
    })


def _has_value_marker(value: str) -> bool:
    text = str(value or "")
    if any(char.isdigit() for char in text):
        return True
    letters = [
        char
        for char in text
        if char.isalpha() and char.lower() != char.upper()
    ]
    return bool(letters and all(char.upper() == char for char in letters))


def _looks_like_user_supplied_value(
    value: str,
    *,
    allow_plain_alpha: bool = False,
) -> bool:
    if not 3 <= len(value) <= 80:
        return False
    if not all(char.isalnum() or char in {"_", "-"} for char in value):
        return False
    return allow_plain_alpha or _has_value_marker(value)


def extract_explicit_request_values_from_text(text: str) -> list[dict[str, Any]]:
    """Extract explicit request facts without language-specific label routing."""
    values: list[dict[str, Any]] = []
    seen: set[str] = set()
    source = str(text or "")
    explicit_urls = extract_urls_from_text(source)
    scrubbed_source = _remove_explicit_urls_from_text(source, explicit_urls)
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
            _append_explicit_request_value(
                values,
                seen,
                label=label,
                value=tail_words[0],
                allow_plain_alpha=True,
            )
            break

    for value, start, end in _iter_request_value_token_spans(scrubbed_source):
        if value.casefold() in seen:
            continue
        if not _looks_like_user_supplied_value(value):
            continue
        labels, label = _nearby_value_labels(
            scrubbed_source,
            start=start,
            end=end,
            value=value,
        )
        if not labels or not label:
            continue
        seen.add(value.casefold())
        values.append({
            "value": value,
            "labels": labels,
            "label": label,
        })

    return values[:12]


def _split_shell_words(command: str) -> list[str]:
    try:
        return shlex.split(str(command or ""), posix=True)
    except ValueError:
        return str(command or "").split()


def _shell_word_is_placeholder(token: str) -> bool:
    stripped = str(token or "").strip().strip("'\"")
    if not stripped:
        return False
    return (
        (stripped.startswith("<") and stripped.endswith(">"))
        or (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
        or ("<" in stripped and ">" in stripped)
        or ("{" in stripped and "}" in stripped)
    )


def _shell_word_is_structural_argument(token: str) -> bool:
    stripped = str(token or "").strip().strip("'\"")
    if not stripped:
        return True
    if stripped in {"&&", "||", "|", ";", "&"}:
        return True
    if stripped.startswith("-"):
        return True
    if _shell_word_is_placeholder(stripped):
        return True
    if _looks_urlish(stripped):
        return True
    if any(char in stripped for char in ("/", "\\", ".", "=", ":", "$", "@")):
        return True
    if any(char.isdigit() for char in stripped):
        return True
    if not all(char.isalpha() or char in {"_", "-"} for char in stripped):
        return True
    return False


def _shell_word_is_plain_prose_token(token: str) -> bool:
    stripped = str(token or "").strip().strip("'\"")
    if not stripped or _shell_word_is_structural_argument(stripped):
        return False
    return any(char.isalpha() for char in stripped)


def _command_starts_with_words(words: list[str], prefix_words: list[str]) -> bool:
    if not prefix_words or len(words) < len(prefix_words):
        return False
    return words[: len(prefix_words)] == prefix_words


def _command_hint_has_prose_tail(
    command: str,
    *,
    executable_prefix: str | None = None,
) -> bool:
    """Return True when an extracted command shape includes prose explanation.

    This is structural cleanup for SKILL command summaries. It intentionally
    avoids vocabulary-specific checks and instead rejects command aliases whose
    arguments contain a long run of plain language tokens, especially after a
    documented placeholder.
    """
    words = _split_shell_words(command)
    if not words:
        return False

    prefix_words = _split_shell_words(executable_prefix or "")
    if prefix_words and _command_starts_with_words(words, prefix_words):
        argument_words = words[len(prefix_words) :]
    else:
        argument_words = words[1:]

    plain_run = 0
    plain_after_placeholder = 0
    seen_placeholder = False
    for word in argument_words:
        if word in {"&&", "||", "|", ";", "&"}:
            plain_run = 0
            plain_after_placeholder = 0
            seen_placeholder = False
            continue
        if _shell_word_is_placeholder(word):
            plain_run = 0
            plain_after_placeholder = 0
            seen_placeholder = True
            continue
        if _shell_word_is_plain_prose_token(word):
            plain_run += 1
            if seen_placeholder:
                plain_after_placeholder += 1
            if plain_run >= 4 or plain_after_placeholder >= 3:
                return True
            continue
        plain_run = 0
        if not _shell_word_is_structural_argument(word):
            plain_after_placeholder = 0
    return False


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
    if _looks_urlish(token):
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
        and any(char.isalpha() for char in token)
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

    if len(words) > 1 and all(
        _shell_word_is_plain_prose_token(word)
        for word in words[1:]
    ):
        return False

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
    for match in re.finditer(":", stripped):
        label = stripped[: match.start()].strip()
        candidate = stripped[match.end() :].strip()
        if candidate and _looks_like_command_intro_label(label):
            yield candidate
    for separator in ("->", "=>"):
        if separator not in stripped:
            continue
        label, _, candidate = stripped.partition(separator)
        label = label.strip()
        candidate = candidate.strip()
        if not candidate or len(label) > 48:
            continue
        yield candidate


def _iter_usage_retry_command_candidates(text: str):
    """Yield documented retry commands after CLI option parsing failures."""
    normalized = str(text or "")
    lowered = normalized.casefold()
    if not any(
        marker in lowered
        for marker in (
            "unknown option",
            "unrecognized option",
            "invalid option",
        )
    ):
        return

    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if not stripped.casefold().startswith("usage:"):
            continue
        candidate = stripped.partition(":")[2].strip()
        candidate = re.sub(r"\s+\[[^\]]+\]", "", candidate).strip()
        candidate = " ".join(candidate.split())
        if not candidate or any(marker in candidate for marker in ("<", ">", "{", "}")):
            continue
        yield candidate


def _extract_shell_command_aliases(text: str) -> dict[str, str]:
    """Extract simple command aliases from instruction text."""
    aliases: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        match = re.match(r"^(?:\$?CLI)\s*(?::=|=)\s*(.+)$", stripped)
        if not match:
            continue
        command = " ".join(match.group(1).strip().split())
        if _looks_like_shell_command(command, explicit_shell_context=True):
            aliases["$CLI"] = command
    return aliases


def _looks_like_command_intro_label(label: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", str(label or "").casefold())
    if not words:
        return False
    return words[-1] in _COMMAND_INTRO_TRAILING_WORDS


def extract_shell_command_candidates(text: str, *, limit: int = 8) -> list[str]:
    """Extract shell-like command candidates from free-form text generically."""
    commands: list[str] = []
    seen: set[str] = set()
    aliases = _extract_shell_command_aliases(text)

    def _add(raw: str, *, explicit: bool) -> None:
        candidate = " ".join(str(raw or "").strip().split())
        candidate = candidate.strip("`").rstrip("。；;")
        if candidate.casefold().startswith("run "):
            candidate = candidate[4:].strip()
            explicit = True
        expanded_prefix = ""
        for alias, replacement in aliases.items():
            if candidate == alias:
                candidate = replacement
                explicit = True
                expanded_prefix = replacement
                break
            if candidate.startswith(alias + " "):
                candidate = replacement + candidate[len(alias):]
                explicit = True
                expanded_prefix = replacement
                break
        if not candidate:
            return
        if expanded_prefix and _command_hint_has_prose_tail(
            candidate,
            executable_prefix=expanded_prefix,
        ):
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
    if len(commands) < limit:
        for candidate in _iter_usage_retry_command_candidates(text):
            if 3 <= len(candidate) <= 400:
                _add(candidate, explicit=True)
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
    """Return True when the request is structurally a prior-fact question.

    The blocking guard stays narrow: it prevents live-state tools from being
    used as proof for explicit questions before same-session evidence is
    checked.  Broader short follow-ups are handled by final-answer synthesis
    without blocking live-state tools.
    """
    text = str(message or "").strip()
    if not text:
        return False
    if len(text) > 120 or "\n" in text or "```" in text:
        return False
    if extract_urls_from_text(text):
        return False
    if extract_exact_shell_commands_from_request(text):
        return False
    if any(marker in text for marker in ("/", "\\", "./", "../")):
        return False
    tokens = tokenize_request_matching_text(text)
    if len(tokens) > 8:
        return False
    return any(char in text for char in ("?", "？", "¿", "؟"))


def request_prefers_session_evidence_synthesis(message: str) -> bool:
    """Return True for compact source-less turns that benefit from recall.

    This is not a router and it does not block tools.  It only tells the final
    answer stage to combine current tool evidence with the same-session compact
    so short follow-ups cannot be answered from an unrelated live-state field.
    """
    text = str(message or "").strip()
    if not text:
        return False
    if len(text) > 120 or "\n" in text or "```" in text:
        return False
    if extract_urls_from_text(text):
        return False
    if extract_exact_shell_commands_from_request(text):
        return False
    if any(marker in text for marker in ("/", "\\", "./", "../")):
        return False
    tokens = tokenize_request_matching_text(text)
    return len(tokens) <= 8


def format_explicit_request_urls_context(message: str) -> str:
    """Format explicit user-supplied URLs as source facts, not routes."""
    urls = extract_urls_from_text(message)
    if not urls:
        return ""
    lines = ["[EXPLICIT USER URLS]:"]
    for url in urls[:8]:
        lines.append(f"- {url}")
    lines.append(
        "Treat these as literal user-provided source facts. Do not infer local "
        "paths from URLs. If the newest request asks for a URL-backed action, "
        "pass the URL to the appropriate active tool whose contract covers that "
        "action; otherwise inspect it with normal tools."
    )
    lines.append("")
    return "\n".join(lines)


def format_current_session_fact_check_context(message: str) -> str:
    """Format the fact-check boundary for prior-action follow-up requests."""
    return (
        "[SESSION FACT LOOKUP BOUNDARY]: When the newest request depends on "
        "what happened earlier in this conversation, answer from same-session "
        "evidence first. Prefer the Current Session Compact and call "
        "search_history(mode='recent', scope='current') when the user asks "
        "about the last completed action/result; use targeted "
        "search_history(scope='current') only when you already have a stable "
        "id, path, tx hash, or exact output fragment. Broad workspace "
        "searches, persisted skill logs, and generated debug snapshots can "
        "contain stale cross-session data; inspect them only after same-session "
        "evidence identifies a specific path, id, or artifact to verify.\n"
    )


def format_explicit_request_values_context(message: str) -> str:
    """Format user-labeled request facts without classifying their meaning."""
    values = extract_explicit_request_values_from_text(message)
    if not values:
        return ""
    lines = ["[STRUCTURED USER REQUEST FACTS]:"]
    lines.append(
        "Use these user-provided facts when a matching tool or skill "
        "parameter is documented; do not ask the user to repeat them. Treat "
        "each value as request context, not automatically as a CLI argument. "
        "Pass a value only into a tool schema, SKILL.md placeholder, or "
        "documented option whose label matches the fact. If the active skill "
        "or tool exposes no matching slot, do not invent one; continue from "
        "verified tool evidence and report the observed value or concrete "
        "blocker."
    )
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


def format_explicit_tool_request_context(hints: dict[str, Any]) -> str:
    """Format a generic boundary for unavailable explicitly named tools/MCPs."""
    requests = hints.get("explicit_tool_requests") if isinstance(hints, dict) else None
    if not isinstance(requests, list) or not requests:
        return ""

    unavailable = [
        str(request.get("name") or "").strip()
        for request in requests
        if isinstance(request, dict)
        and str(request.get("name") or "").strip()
        and not bool(request.get("available"))
    ]
    if not unavailable:
        return ""

    lines = ["[EXPLICIT TOOL REQUEST BOUNDARY]:"]
    lines.append(
        "The newest request explicitly names tool/MCP capability below, but "
        "no available registered tool or MCP has that name. Do not satisfy "
        "that request by substituting another tool such as web_search, "
        "web_fetch, shell, or an ad-hoc API call. Report that the requested "
        "capability is unavailable unless the newest request separately says "
        "to use an alternative tool."
    )
    for name in unavailable[:6]:
        lines.append(f"- {name}")
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

    def _extract_markdown_section_lines(heading_name: str) -> list[str]:
        section: list[str] = []
        in_section = False
        for raw_line in content.splitlines():
            stripped = raw_line.strip()
            heading = _markdown_heading_text(stripped)
            if heading and heading.casefold().split()[:1] == [heading_name.casefold()]:
                in_section = True
                continue
            if in_section and heading:
                break
            if in_section:
                section.append(raw_line)
        return section

    def _iter_fenced_blocks(markdown: str):
        in_fence = False
        fence_marker = ""
        fence_language = ""
        block_lines: list[str] = []
        for raw_line in str(markdown or "").splitlines():
            stripped = raw_line.strip()
            if stripped.startswith(("```", "~~~")):
                marker = stripped[:3]
                if in_fence and marker == fence_marker:
                    yield fence_language, "\n".join(block_lines)
                    in_fence = False
                    fence_marker = ""
                    fence_language = ""
                    block_lines = []
                    continue
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                    language_parts = stripped[3:].strip().split(None, 1)
                    fence_language = (
                        language_parts[0].casefold() if language_parts else ""
                    )
                    block_lines = []
                    continue
            if in_fence:
                block_lines.append(raw_line)

    def _fence_language_is_shell(language: str) -> bool:
        normalized = str(language or "").strip().casefold()
        return normalized in {"", "bash", "sh", "shell", "zsh", "powershell", "ps1"}

    command_section = _extract_markdown_section_lines("commands")

    scan_texts: list[str] = []
    if command_section:
        scan_texts.append("\n".join(command_section))

    for language, block in _iter_fenced_blocks(content):
        if not _fence_language_is_shell(language):
            continue
        if command_section and block in scan_texts:
            continue
        scan_texts.append(block)

    if not scan_texts and cli_value:
        scan_texts.append("")

    for scan_text in scan_texts:
        for candidate, explicit in _iter_structural_command_candidates(scan_text):
            stripped = _normalize_skill_command_hint(candidate)
            if not stripped:
                continue
            if stripped.startswith("$CLI "):
                expanded = stripped.replace("$CLI", cli_value or "$CLI", 1)
                if _command_hint_has_prose_tail(
                    expanded,
                    executable_prefix=cli_value or None,
                ):
                    continue
                _push_command(expanded)
                continue
            if _looks_like_shell_command(stripped, explicit_shell_context=explicit):
                _push_command(stripped)

    urls = extract_urls_from_text(content)
    return commands[:16], urls[:12]

def build_request_execution_hints(
    message: str,
    skill_candidates: list[dict[str, Any]],
    available_tool_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Any]:
    """Build generic execution hints from explicit request text only.

    Hints are used by low-level tools for protocol hygiene, not for routing
    workflows.  Do not infer selected skills, catalog targets, game/site names,
    or side-effect permissions here; the model should decide those from the
    real user request, the installed skill catalog, SKILL.md, and tool output.
    """
    message_text = str(message or "")
    explicit_urls = extract_urls_from_text(message_text)
    executable_skill_candidate_count = 0
    local_executable_skills: list[dict[str, Any]] = []
    for candidate in skill_candidates:
        skill_md = candidate.get("skill_md")
        if not isinstance(skill_md, Path):
            continue
        commands, _urls = extract_skill_command_hints(skill_md)
        if commands:
            executable_skill_candidate_count += 1
            local_executable_skills.append({
                "name": str(candidate.get("name") or "").strip(),
                "commands": commands[:8],
                "skill_md": str(skill_md),
            })

    explicit_request_values = extract_explicit_request_values_from_text(message_text)
    exact_commands = extract_exact_shell_commands_from_request(message_text)
    explicit_tool_requests = extract_explicit_tool_requests_from_text(
        message_text,
        available_tool_names,
    )
    unavailable_explicit_tool_requests = [
        request
        for request in explicit_tool_requests
        if isinstance(request, dict) and not bool(request.get("available"))
    ]
    current_session_fact_check_required = (
        request_needs_current_session_fact_check(message_text)
    )
    session_evidence_synthesis_preferred = (
        request_prefers_session_evidence_synthesis(message_text)
    )

    return {
        "explicit_request_urls": explicit_urls,
        "explicit_request_values": explicit_request_values,
        "local_executable_skills": local_executable_skills,
        "exact_shell_commands": exact_commands,
        "explicit_tool_requests": explicit_tool_requests,
        "unavailable_explicit_tool_requests": unavailable_explicit_tool_requests,
        "restrict_to_exact_shell_commands": request_restricts_to_exact_shell_commands(
            message_text,
            exact_commands,
        ),
        "current_session_fact_check_required": current_session_fact_check_required,
        "session_evidence_synthesis_preferred": session_evidence_synthesis_preferred,
        "skill_candidate_count": len(skill_candidates),
        "executable_skill_candidate_count": executable_skill_candidate_count,
    }
