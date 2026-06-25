"""Task-local execution context for tool ownership scoping and stream capture."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4

from spoon_bot.agent.request_hints import extract_shell_command_candidates
from spoon_bot.agent.execution_ledger import record_tool_capture_in_ledger

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
_TOOL_INVOCATION_STATE_BY_OWNER: dict[str, dict[str, Any]] = {}
_TOOL_INVOCATION_STATE_FALLBACK: dict[str, Any] | None = None
_TOOL_INVOCATION_STATE_LOCK = Lock()
_REQUEST_EXECUTION_HINTS: ContextVar[dict[str, Any] | None] = ContextVar(
    "request_execution_hints",
    default=None,
)
_TOOL_WORKSPACE: ContextVar[str | None] = ContextVar(
    "tool_workspace",
    default=None,
)
_TOOL_RUN_ID: ContextVar[str | None] = ContextVar(
    "tool_run_id",
    default=None,
)
_REQUEST_EXECUTION_HINTS_BY_OWNER: dict[str, dict[str, Any]] = {}
_REQUEST_EXECUTION_HINTS_FALLBACK: dict[str, Any] | None = None
_REQUEST_EXECUTION_HINTS_LOCK = Lock()
_CANCELLED_TOOL_RUNS: set[str] = set()
_CANCELLED_TOOL_RUNS_LOCK = Lock()


@dataclass
class CapturedToolOutput:
    """Full tool output captured for websocket streaming."""

    scope_id: str
    owner: str
    tool_name: str
    arguments_key: str
    summary_output: str = ""
    full_output: str = ""
    progress_recorded: bool = False
    guardrail_stop: bool = False
    guardrail_reason: str = ""
    guardrail_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


_TOOL_OUTPUT_LOCK = Lock()
_ACTIVE_TOOL_INVOCATIONS: dict[str, CapturedToolOutput] = {}
_COMPLETED_TOOL_OUTPUTS: dict[str, list[CapturedToolOutput]] = defaultdict(list)
_MAX_CAPTURED_TOOL_OUTPUTS_PER_SCOPE = 256
_DUPLICATE_TOOL_INVOCATION_MESSAGE = (
    "STOP_TOOL_LOOP: Error: duplicate tool invocation suppressed. The same tool and arguments "
    "already executed in this request; use the previous result or choose "
    "different arguments. Do not repeat this same action. If the user's request "
    "has remaining distinct steps, continue with the next step from the existing "
    "tool evidence; otherwise report the previous result or blocker."
)
_REPEATED_TOOL_SERIES_MESSAGE = (
    "STOP_TOOL_LOOP: Error: repeated side-effecting tool series suppressed. This request has "
    "already executed the same kind of external side effect; report the latest "
    "result or ask for an explicit count before continuing. Do not call more "
    "tools for this same action."
)
_REPEATED_BACKGROUND_JOB_POLL_MESSAGE = (
    "STOP_TOOL_LOOP: Error: repeated background job polling suppressed. "
    "job_id={job_id}. The background job produced no new status or output "
    "signal across repeated polls; report the current state instead of "
    "polling again in this request."
)
_REDUNDANT_FILE_READ_MESSAGE = (
    "READ_FILE_CACHE_HIT: requested file range already available in this "
    "request. File content already available in this request. Treat this read "
    "as complete and continue the user's remaining instructions without "
    "calling read_file again for the same path and range."
)
_REPEATED_REDUNDANT_FILE_READ_MESSAGE = (
    "Repeated read skipped: the requested file range was already provided "
    "earlier in this request; use the previous file content and continue with "
    "the next non-read action."
)
_REPEATED_SHELL_FILE_READ_MESSAGE = (
    "STOP_TOOL_LOOP: Error: repeated shell file read suppressed. The requested "
    "file range was already inspected in this request; use the previous file "
    "content and continue with the next write, edit, test, or final-answer "
    "action instead of running another shell file read."
)
_REPEATED_TOOL_FAILURE_STRATEGY_WARNING = (
    "TOOL_PROGRESS_HINT: The same tool failure pattern has repeated. Inspect the "
    "failure and change strategy instead of retrying it unchanged. If the failed "
    "tool was only an optional external dependency and the user's requested "
    "artifact can still be created with available local/code-native tools, continue "
    "with those tools and report the degraded dependency honestly."
)
_EXPLICIT_COMMAND_EVIDENCE_RE = re.compile(
    r"(?im)(?:\bRetry\b[^\n]{0,120}\bcommand\b|"
    r"\bDocumented\s+form\s*:|\bUsage\s*:|"
    r"^[A-Za-z0-9 _./-]{1,48}\s*(?:->|=>)\s*\S)"
)
_PLACEHOLDER_RE = re.compile(r"<[A-Za-z0-9_-]+>")

_READ_ONLY_OR_SETUP_TOOL_NAMES = frozenset({
    "grep",
    "list_dir",
    "read_file",
    "search_history",
    "self_upgrade",
    "skill_marketplace",
    "web_fetch",
    "web_search",
})
_STATEFUL_TOOL_NAMES = frozenset({
    "cron",
    "edit_file",
    "service_expose",
    "spawn",
    "write_file",
})
_SKILL_CONTRACT_MARKERS = (
    "skill-ref",
    "[skill.md execution contract]",
    "[skill.md execution summary]",
    "skill.md execution contract",
    "skill.md execution summary",
)
_READ_ONLY_SHELL_COMMANDS = frozenset({
    "cat",
    "cut",
    "diff",
    "du",
    "echo",
    "file",
    "find",
    "grep",
    "head",
    "ls",
    "pwd",
    "rg",
    "sed",
    "sort",
    "stat",
    "tail",
    "test",
    "wc",
    "which",
})
_SKILL_ENTRYPOINT_WRAPPERS = frozenset({
    "bash",
    "bun",
    "deno",
    "node",
    "npx",
    "npm",
    "pnpm",
    "poetry",
    "python",
    "python3",
    "sh",
    "tsx",
    "uv",
    "uvx",
    "yarn",
})
_READ_ONLY_SKILL_ACTIONS = frozenset({
    "-h",
    "--help",
    "balance",
    "balances",
    "context",
    "get",
    "help",
    "history",
    "list",
    "logs",
    "show",
    "snapshot",
    "state",
    "status",
    "summary",
    "version",
    "wallet",
    "wait",
})
_PREPARATORY_SHELL_COMMANDS = frozenset({
    "mkdir",
    "touch",
})
_PACKAGE_MANAGER_COMMANDS = frozenset({
    "bun",
    "npm",
    "pnpm",
    "yarn",
})
_PYTHON_PACKAGE_INSTALL_COMMANDS = frozenset({
    "pip",
    "pip3",
})
_CANCELLED_TOOL_RUN_MESSAGE = (
    "STOP_TOOL_LOOP: Error: request was cancelled before this tool could run. "
    "Do not execute additional tools for this cancelled request."
)


def _runtime_skill_read_only_budget() -> int:
    raw = os.getenv("SPOON_BOT_SKILL_READ_ONLY_BUDGET", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 8


def _normalize_invocation_category(category: str | None) -> str:
    value = str(category or "").strip().casefold().replace("-", "_")
    if value in {"read", "readonly", "read_only", "inspection", "inspect"}:
        return "read_only"
    if value in {"setup", "preparatory", "preparation"}:
        return "setup"
    if value in {"write", "mutation", "stateful", "progress", "side_effect"}:
        return "stateful"
    return ""


def _default_invocation_category(tool_name: str) -> str:
    normalized = str(tool_name or "").strip().casefold().replace("-", "_")
    if normalized in _STATEFUL_TOOL_NAMES:
        return "stateful"
    if normalized in _READ_ONLY_OR_SETUP_TOOL_NAMES:
        return "setup" if normalized in {"self_upgrade", "skill_marketplace"} else "read_only"
    return ""


def _arguments_shell_command(arguments: Any) -> str:
    payload = arguments
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    if isinstance(payload, dict):
        return str(payload.get("command") or "").strip()
    return ""


def _segment_is_help_or_version_inspection(segment: list[str]) -> bool:
    """Return True for generic CLI help/version probes."""
    tokens = [str(token or "").strip().casefold() for token in segment if str(token or "").strip()]
    if len(tokens) <= 1:
        return False
    inspection_tokens = {"--help", "-h", "help", "--version", "version"}
    return any(token in inspection_tokens for token in tokens[1:])


def _shell_command_looks_read_only(command: str) -> bool:
    try:
        tokens = shlex.split(str(command or ""))
    except ValueError:
        tokens = str(command or "").split()
    if not tokens:
        return True

    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in {"&&", ";", "||", "|"}:
            if segments[-1]:
                segments.append([])
            continue
        if token.startswith((">", ">>")):
            return False
        segments[-1].append(token)

    cleaned_segments = [segment for segment in segments if segment]
    if not cleaned_segments:
        return True
    for segment in cleaned_segments:
        if _skill_cli_segment_looks_read_only(segment):
            continue
        command_name = str(segment[0]).replace("\\", "/").rsplit("/", 1)[-1].casefold()
        if command_name == "cd":
            continue
        if _segment_is_help_or_version_inspection(segment):
            continue
        if command_name not in _READ_ONLY_SHELL_COMMANDS:
            return False
    return True


def _split_shell_command_segments(command: str) -> list[list[str]]:
    try:
        tokens = shlex.split(str(command or ""))
    except ValueError:
        tokens = str(command or "").split()
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in {"&&", ";", "||", "|"}:
            if segments[-1]:
                segments.append([])
            continue
        if token.startswith((">", ">>", "<")):
            return []
        segments[-1].append(token)
    return [segment for segment in segments if segment]


def _segment_command_name(segment: list[str]) -> str:
    for token in segment:
        if "=" in token and not token.startswith(("-", "./", "/", "../")):
            key, _sep, value = token.partition("=")
            if key and value and key.replace("_", "").isalnum():
                continue
        return token.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    return ""


def _segment_command_index(segment: list[str]) -> int | None:
    for index, token in enumerate(segment):
        if "=" in token and not token.startswith(("-", "./", "/", "../")):
            key, _sep, value = token.partition("=")
            if key and value and key.replace("_", "").isalnum():
                continue
        return index
    return None


def _token_looks_like_skill_entrypoint(token: str) -> bool:
    normalized = str(token or "").replace("\\", "/").casefold()
    if "/skills/" not in normalized and not normalized.startswith("skills/"):
        return False
    if normalized.endswith("/skill.md") or normalized.endswith("skill.md"):
        return False
    return True


def _skill_cli_segment_looks_read_only(segment: list[str]) -> bool:
    command_index = _segment_command_index(segment)
    if command_index is None:
        return False
    command_name = _segment_command_name(segment)
    if command_name not in _SKILL_ENTRYPOINT_WRAPPERS:
        return False

    entrypoint_index: int | None = None
    for index in range(command_index + 1, len(segment)):
        token = segment[index]
        if str(token or "").startswith("-"):
            continue
        if _token_looks_like_skill_entrypoint(token):
            entrypoint_index = index
            break
    if entrypoint_index is None:
        return False

    action_tokens = [
        str(token or "").strip().casefold()
        for token in segment[entrypoint_index + 1 :]
        if str(token or "").strip() and not str(token or "").startswith("-")
    ]
    if not action_tokens:
        return False
    if action_tokens[0] in _READ_ONLY_SKILL_ACTIONS:
        return True
    return len(action_tokens) >= 2 and action_tokens[1] in _READ_ONLY_SKILL_ACTIONS


def shell_command_is_preparatory_state_change(command: str) -> bool:
    """Return True for pure workspace setup commands that create no deliverable."""
    segments = _split_shell_command_segments(command)
    if not segments:
        return False

    saw_preparatory = False
    for segment in segments:
        command_name = _segment_command_name(segment)
        if not command_name:
            continue
        if command_name == "cd":
            continue
        if command_name in _PREPARATORY_SHELL_COMMANDS:
            saw_preparatory = True
            continue
        if command_name == "install" and "-d" in segment[1:]:
            saw_preparatory = True
            continue
        return False
    return saw_preparatory


def shell_command_is_dependency_setup(command: str) -> bool:
    """Return True for dependency install commands that are setup, not delivery."""
    segments = _split_shell_command_segments(command)
    if not segments:
        return False

    saw_install = False
    for segment in segments:
        command_name = _segment_command_name(segment)
        if not command_name:
            continue
        if command_name == "cd":
            continue
        args = [str(token or "").casefold() for token in segment[1:]]
        if command_name in _PACKAGE_MANAGER_COMMANDS:
            if not args:
                return False
            if command_name == "yarn":
                if args[0] in {"install", "add"} or args[0].startswith("--"):
                    saw_install = True
                    continue
                return False
            if args[0] in {"install", "i", "add"}:
                saw_install = True
                continue
            return False
        if command_name in _PYTHON_PACKAGE_INSTALL_COMMANDS:
            if args and args[0] == "install":
                saw_install = True
                continue
            return False
        if command_name in {"python", "python3"}:
            if len(args) >= 3 and args[0] == "-m" and args[1] == "pip" and args[2] == "install":
                saw_install = True
                continue
            return False
        if command_name == "uv":
            if args[:2] == ["pip", "install"]:
                saw_install = True
                continue
            return False
        return False
    return saw_install


def _tool_invocation_category(tool_name: str, category: str | None = None) -> str:
    return _normalize_invocation_category(category) or _default_invocation_category(tool_name)


def _tool_invocation_category_from_arguments(
    tool_name: str,
    arguments: Any,
    category: str | None = None,
) -> str:
    normalized = str(tool_name or "").strip().casefold().replace("-", "_")
    explicit = _normalize_invocation_category(category)
    if explicit:
        return explicit
    if normalized == "shell":
        command = _arguments_shell_command(arguments)
        if _shell_command_looks_read_only(command):
            return "read_only"
        if (
            shell_command_is_preparatory_state_change(command)
            or shell_command_is_dependency_setup(command)
        ):
            return "setup"
        return "stateful"
    return _default_invocation_category(normalized)


def classify_tool_invocation_category(
    tool_name: str,
    arguments: Any = None,
    *,
    category: str | None = None,
) -> str:
    """Return a generic read/setup/stateful category for runtime scheduling."""
    return _tool_invocation_category_from_arguments(
        tool_name,
        arguments,
        category,
    )


def _tool_invocation_has_substantive_progress(
    tool_name: str,
    arguments: Any,
    result: Any,
    category: str,
) -> bool:
    if category != "stateful" or _tool_failure_signal(result) is not None:
        return False
    normalized = str(tool_name or "").strip().casefold().replace("-", "_")
    if normalized != "shell":
        return True
    command = _arguments_shell_command(arguments)
    if command and shell_command_is_preparatory_state_change(command):
        return False
    return True


def _text_contains_workspace_skill_contract_path(text: str) -> bool:
    normalized = str(text or "").replace("\\", "/").casefold()
    marker = "skills/"
    start = 0
    while True:
        index = normalized.find(marker, start)
        if index < 0:
            return False
        suffix = normalized[index + len(marker):]
        slash_index = suffix.find("/")
        if slash_index > 0:
            skill_name = suffix[:slash_index]
            remainder = suffix[slash_index + 1:]
            if (
                skill_name not in {".", ".."}
                and all(char.isalnum() or char in {"-", "_", "."} for char in skill_name)
                and remainder.startswith("skill.md")
            ):
                return True
        start = index + len(marker)


def _payload_indicates_skill_contract(payload: Any) -> bool:
    text = stringify_tool_output(payload).replace("\\", "/").casefold()
    if any(marker in text for marker in _SKILL_CONTRACT_MARKERS):
        return True
    return _text_contains_workspace_skill_contract_path(text)


def _request_skill_contract_seen(state: dict[str, Any]) -> bool:
    return bool(state.get("skill_contract_seen"))


def _request_skill_stateful_progress_seen(state: dict[str, Any]) -> bool:
    return bool(state.get("skill_stateful_progress_seen"))
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


def get_tool_workspace() -> str | None:
    """Return the task-local workspace path for tools that need it."""
    workspace = _TOOL_WORKSPACE.get()
    if isinstance(workspace, str) and workspace.strip():
        return workspace.strip()
    return None


@contextmanager
def bind_tool_owner(owner: str | None) -> Iterator[str]:
    """Bind tool owner for the current task context."""
    normalized = owner.strip() if isinstance(owner, str) and owner.strip() else "default"
    token = _TOOL_OWNER.set(normalized)
    try:
        yield normalized
    finally:
        _TOOL_OWNER.reset(token)


@contextmanager
def bind_tool_workspace(workspace: str | None) -> Iterator[str | None]:
    """Bind the current agent workspace for skill/tool execution."""
    normalized = workspace.strip() if isinstance(workspace, str) and workspace.strip() else None
    token = _TOOL_WORKSPACE.set(normalized)
    try:
        yield normalized
    finally:
        _TOOL_WORKSPACE.reset(token)


@contextmanager
def bind_tool_run(run_id: str | None) -> Iterator[str | None]:
    """Bind a concrete agent run so late tool calls can honor cancellation."""
    normalized = str(run_id or "").strip() or None
    token = _TOOL_RUN_ID.set(normalized)
    try:
        yield normalized
    finally:
        _TOOL_RUN_ID.reset(token)


def mark_tool_run_cancelled(run_id: str | None) -> None:
    """Mark a concrete agent run as cancelled for all future tool calls."""
    normalized = str(run_id or "").strip()
    if not normalized:
        return
    with _CANCELLED_TOOL_RUNS_LOCK:
        _CANCELLED_TOOL_RUNS.add(normalized)


def clear_tool_run_cancelled(run_id: str | None) -> None:
    """Clear cancellation state after the concrete agent run has stopped."""
    normalized = str(run_id or "").strip()
    if not normalized:
        return
    with _CANCELLED_TOOL_RUNS_LOCK:
        _CANCELLED_TOOL_RUNS.discard(normalized)


def cancelled_tool_run_blocker() -> str | None:
    """Return a blocker when this tool invocation belongs to a cancelled run."""
    run_id = str(_TOOL_RUN_ID.get() or "").strip()
    if not run_id:
        return None
    with _CANCELLED_TOOL_RUNS_LOCK:
        cancelled = run_id in _CANCELLED_TOOL_RUNS
    return _CANCELLED_TOOL_RUN_MESSAGE if cancelled else None


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


def _failure_signal_from_mapping(payload: dict[str, Any]) -> str | None:
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
    return None


def _parse_json_mapping_from_text(text: str) -> dict[str, Any] | None:
    """Parse a JSON object embedded in a plain tool result, if one is present."""
    value = str(text or "").strip()
    if not value:
        return None

    candidates: list[str] = [value]
    first = value.find("{")
    last = value.rfind("}")
    if first >= 0 and last > first:
        candidates.append(value[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _tool_failure_signal(payload: Any) -> str | None:
    """Return a generic failure signal for loop guards, independent of paths/routes."""
    if isinstance(payload, dict):
        signal = _failure_signal_from_mapping(payload)
        if signal is not None:
            return signal

    text = stringify_tool_output(payload).lower()
    if not text.strip():
        return None
    parsed = _parse_json_mapping_from_text(text)
    if parsed is not None:
        signal = _failure_signal_from_mapping(parsed)
        if signal is not None:
            return signal
    if re.search(r"exit code:\s*[1-9]\d*", text):
        return "nonzero_exit"
    if "stop_tool_loop" in text:
        return "stop_tool_loop"
    if text.lstrip().startswith("rejected:"):
        return "rejected"
    if text.lstrip().startswith("security error:"):
        return "security_error"
    if text.lstrip().startswith("error:"):
        return "error_prefix"
    if "traceback" in text or "exception" in text:
        return "runtime_error"
    return None


def _tool_failure_fingerprint(payload: Any) -> str:
    """Return a compact generic fingerprint for repeated no-progress failures."""
    parsed = payload if isinstance(payload, dict) else _parse_json_mapping_from_text(
        stringify_tool_output(payload)
    )
    if isinstance(parsed, dict):
        for key in ("error", "message", "detail"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                text = value.casefold()
                break
        else:
            text = stringify_tool_output(parsed).casefold()
    else:
        text = stringify_tool_output(payload).casefold()
    text = re.sub(r"\b0x[0-9a-f]{16,}\b", "0x#", text)
    text = re.sub(r"\b\d+\b", "#", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    return text[:240]


_MODAL_VOLUME_PATH_RE = re.compile(r"^/__modal/volumes/[^/]+(?:/(.*))?$")


def _normalize_runtime_path_alias(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    match = _MODAL_VOLUME_PATH_RE.match(text)
    if not match:
        return text
    remainder = (match.group(1) or "").lstrip("/")
    return "/workspace" + (f"/{remainder}" if remainder else "")


def _normalize_tool_argument_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return _normalize_runtime_path_alias(payload)
    if isinstance(payload, list):
        return [_normalize_tool_argument_payload(item) for item in payload]
    if isinstance(payload, dict):
        return {
            key: _normalize_tool_argument_payload(value)
            for key, value in payload.items()
        }
    return payload


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
            return _normalize_runtime_path_alias(text)
        return normalize_tool_arguments(parsed)
    if isinstance(arguments, (dict, list)):
        try:
            normalized = _normalize_tool_argument_payload(arguments)
            return json.dumps(normalized, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(arguments)
    return str(arguments)


_BODY_ARGUMENT_KEYS = frozenset({
    "body",
    "code",
    "content",
    "data",
    "html",
    "markdown",
    "newtext",
    "oldtext",
    "payload",
    "source",
    "text",
})
_BODY_ARGUMENT_THRESHOLD = 256
_GENERIC_ARGUMENT_THRESHOLD = 4096


def _argument_key_name(key: Any) -> str:
    return "".join(
        char for char in str(key or "") if char not in {"_", "-", " "}
    ).casefold()


def _summarize_argument_text(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8", errors="replace")
    return {
        "content_ref": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        "chars": len(text),
        "bytes": len(encoded),
        "lines": text.count("\n") + 1 if text else 0,
        "source": "tool argument body stored outside conversation history; use read_file for current file text",
    }


def _sanitize_tool_argument_value(tool_name: str, key: Any, value: Any) -> Any:
    normalized_key = _argument_key_name(key)
    normalized_tool = str(tool_name or "").strip().casefold().replace("-", "_")

    if isinstance(value, str):
        threshold = (
            _BODY_ARGUMENT_THRESHOLD
            if (
                normalized_key in _BODY_ARGUMENT_KEYS
                or (
                    normalized_tool == "write_file"
                    and normalized_key == "content"
                )
                or (
                    normalized_tool == "edit_file"
                    and normalized_key in {"oldtext", "newtext"}
                )
            )
            else _GENERIC_ARGUMENT_THRESHOLD
        )
        if len(value) > threshold or (
            normalized_key in _BODY_ARGUMENT_KEYS
            and value.count("\n") >= 8
        ):
            return _summarize_argument_text(value)
        return value

    if isinstance(value, dict):
        return {
            str(item_key): _sanitize_tool_argument_value(tool_name, item_key, item_value)
            for item_key, item_value in value.items()
        }

    if isinstance(value, list):
        return [
            _sanitize_tool_argument_value(tool_name, key, item)
            for item in value
        ]

    return value


def sanitize_tool_arguments_for_history(tool_name: str, arguments: Any) -> str:
    """Return tool arguments safe for websocket/session history display.

    This keeps identifying fields such as paths, commands, flags, and small
    values, but replaces large body-like fields with a digest. The live tool
    invocation still receives the original arguments; this function is only for
    transcript/UI/context compaction.
    """
    payload = arguments
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return ""
        try:
            payload = json.loads(text)
        except Exception:
            if len(text) > _GENERIC_ARGUMENT_THRESHOLD:
                return json.dumps(
                    _summarize_argument_text(text),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            return _normalize_runtime_path_alias(text)

    sanitized = _sanitize_tool_argument_value(tool_name, "", payload)
    if isinstance(sanitized, (dict, list)):
        try:
            return json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(sanitized)
    return str(sanitized)


def _new_tool_invocation_state(
    *,
    max_repeats: int,
    max_series_repeats: int,
    max_consecutive_failures: int,
) -> dict[str, Any]:
    return {
        "counts": defaultdict(int),
        "last_exact_key": None,
        "last_exact_repeats": 0,
        "series_counts": defaultdict(int),
        "consecutive_failures": [],
        "failure_pattern_counts": defaultdict(int),
        "file_read_coverage": defaultdict(list),
        "max_repeats": max(1, int(max_repeats or 1)),
        "max_series_repeats": max(1, int(max_series_repeats or 1)),
        "max_consecutive_failures": max(1, int(max_consecutive_failures or 1)),
    }


def _ensure_tool_invocation_state_shape(
    state: dict[str, Any],
    *,
    max_repeats: int,
    max_series_repeats: int,
    max_consecutive_failures: int,
) -> dict[str, Any]:
    state.setdefault("counts", defaultdict(int))
    state.setdefault("last_exact_key", None)
    state.setdefault("last_exact_repeats", 0)
    state.setdefault("series_counts", defaultdict(int))
    state.setdefault("consecutive_failures", [])
    state.setdefault("failure_pattern_counts", defaultdict(int))
    state.setdefault("file_read_coverage", defaultdict(list))
    state["max_repeats"] = max(1, int(max_repeats or 1))
    state["max_series_repeats"] = max(1, int(max_series_repeats or 1))
    state["max_consecutive_failures"] = max(1, int(max_consecutive_failures or 1))
    return state


@contextmanager
def track_tool_invocations(
    *,
    max_repeats: int = 2,
    max_series_repeats: int = 2,
    max_consecutive_failures: int = 3,
    initial_state: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Track exact tool invocations for the current request.

    This is a data-plane guard: it suppresses identical tool+argument executions
    inside one request without routing based on user prompt wording.
    """
    if isinstance(initial_state, dict):
        state = _ensure_tool_invocation_state_shape(
            initial_state,
            max_repeats=max_repeats,
            max_series_repeats=max_series_repeats,
            max_consecutive_failures=max_consecutive_failures,
        )
    else:
        state = _new_tool_invocation_state(
            max_repeats=max_repeats,
            max_series_repeats=max_series_repeats,
            max_consecutive_failures=max_consecutive_failures,
        )
    global _TOOL_INVOCATION_STATE_FALLBACK
    owner = get_tool_owner()
    missing = object()
    with _TOOL_INVOCATION_STATE_LOCK:
        previous_owner_state = _TOOL_INVOCATION_STATE_BY_OWNER.get(owner, missing)
        previous_fallback = _TOOL_INVOCATION_STATE_FALLBACK
        _TOOL_INVOCATION_STATE_BY_OWNER[owner] = state
        _TOOL_INVOCATION_STATE_FALLBACK = state
    token = _TOOL_INVOCATION_DEDUP_STATE.set(state)
    try:
        yield state
    finally:
        _TOOL_INVOCATION_DEDUP_STATE.reset(token)
        with _TOOL_INVOCATION_STATE_LOCK:
            if previous_owner_state is missing:
                _TOOL_INVOCATION_STATE_BY_OWNER.pop(owner, None)
            else:
                _TOOL_INVOCATION_STATE_BY_OWNER[owner] = previous_owner_state
            _TOOL_INVOCATION_STATE_FALLBACK = previous_fallback


def _current_tool_invocation_state() -> dict[str, Any] | None:
    state = _TOOL_INVOCATION_DEDUP_STATE.get()
    if isinstance(state, dict):
        return state
    owner = get_tool_owner()
    with _TOOL_INVOCATION_STATE_LOCK:
        owner_state = _TOOL_INVOCATION_STATE_BY_OWNER.get(owner)
        if isinstance(owner_state, dict):
            return owner_state
        if isinstance(_TOOL_INVOCATION_STATE_FALLBACK, dict):
            return _TOOL_INVOCATION_STATE_FALLBACK
    return None


def normalize_observed_cli_command(command: str) -> str:
    """Normalize a shell-emitted command so exact evidence can be matched."""
    compact = " ".join(str(command or "").strip().strip("`").split())
    compact = compact.rstrip("。；;")
    if not compact or _PLACEHOLDER_RE.search(compact):
        return ""
    try:
        tokens = shlex.split(compact)
    except ValueError:
        return compact
    if not tokens:
        return ""
    return shlex.join(tokens)


def _extract_observed_cli_commands(text: str, *, limit: int = 12) -> list[str]:
    """Extract explicit shell command evidence from a shell result."""
    output = str(text or "")
    if not output or not _EXPLICIT_COMMAND_EVIDENCE_RE.search(output):
        return []

    commands: list[str] = []
    seen: set[str] = set()
    for command in extract_shell_command_candidates(output, limit=limit * 2):
        normalized = normalize_observed_cli_command(command)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        commands.append(normalized)
        if len(commands) >= limit:
            break
    return commands


def record_shell_command_evidence(tool_name: str, output: Any) -> None:
    """Record exact shell-emitted commands as evidence, not as a route plan."""
    if str(tool_name or "").strip().casefold() != "shell":
        return
    state = _current_tool_invocation_state()
    if not isinstance(state, dict):
        return

    commands = _extract_observed_cli_commands(stringify_tool_output(output))
    if not commands:
        return

    bucket = state.setdefault("observed_cli_commands", [])
    if not isinstance(bucket, list):
        return
    seen = {
        str(item)
        for item in bucket
        if isinstance(item, str) and item.strip()
    }
    for command in commands:
        if command in seen:
            continue
        bucket.append(command)
        seen.add(command)
    if len(bucket) > 64:
        del bucket[:-64]


def get_observed_cli_commands() -> list[str]:
    """Return request-local commands emitted by previous shell output."""
    state = _current_tool_invocation_state()
    if not isinstance(state, dict):
        return []
    bucket = state.get("observed_cli_commands")
    if not isinstance(bucket, list):
        return []
    return [
        str(item)
        for item in bucket
        if isinstance(item, str) and item.strip()
    ]


def observed_cli_command_matches(command: str) -> bool:
    """Return True when a command exactly matches previous shell evidence."""
    normalized = normalize_observed_cli_command(command)
    if not normalized:
        return False
    return normalized in set(get_observed_cli_commands())


def suppress_repeated_tool_invocation(
    tool_name: str,
    arguments: Any,
    *,
    max_repeats: int | None = None,
) -> str | None:
    """Return a compact duplicate result when the same call already ran."""
    state = _current_tool_invocation_state()
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

    effective_max_repeats = (
        int(max_repeats)
        if isinstance(max_repeats, int) and max_repeats > 0
        else int(state.get("max_repeats") or 1)
    )
    effective_count = counts[key] if max_repeats is not None else repeat_count
    if effective_count <= effective_max_repeats:
        return None
    return _DUPLICATE_TOOL_INVOCATION_MESSAGE


def suppress_repeated_tool_series(tool_name: str, series_key: str | None) -> str | None:
    """Return a compact result when one request repeats the same side-effect series."""
    state = _current_tool_invocation_state()
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


def suppress_repeated_background_job_poll(
    job_id: str,
    progress_key: str,
    *,
    max_no_progress_polls: int = 2,
) -> str | None:
    """Return a compact result when one request keeps polling an unchanged job."""
    state = _current_tool_invocation_state()
    if not isinstance(state, dict):
        return None

    normalized_job_id = str(job_id or "").strip()
    normalized_progress_key = str(progress_key or "").strip()
    if not normalized_job_id or not normalized_progress_key:
        return None

    poll_state = state.setdefault("background_job_poll_state", {})
    if not isinstance(poll_state, dict):
        return None

    current = poll_state.get(normalized_job_id)
    if (
        isinstance(current, dict)
        and current.get("progress_key") == normalized_progress_key
    ):
        count = int(current.get("count") or 0) + 1
    else:
        count = 1

    poll_state[normalized_job_id] = {
        "progress_key": normalized_progress_key,
        "count": count,
    }
    if count <= max(1, int(max_no_progress_polls or 1)):
        return None
    return _REPEATED_BACKGROUND_JOB_POLL_MESSAGE.format(job_id=normalized_job_id)


def suppress_redundant_file_read(
    path_key: str,
    *,
    offset: int | None,
    limit: int | None,
    total_lines: int,
    content_fingerprint: str | None = None,
    request_key: str | None = None,
) -> str | None:
    """Return a compact result when a request re-reads covered file content."""
    state = _current_tool_invocation_state()
    if not isinstance(state, dict):
        return None

    coverage = state.get("file_read_coverage")
    if not isinstance(coverage, dict):
        return None

    normalized_path = _normalize_runtime_path_alias(str(path_key or "").strip())
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
    current_request_key = f"{normalized_path}\x1f{start}\x1f{end}\x1f{fingerprint}"
    request_counts = state.setdefault("redundant_file_read_counts", defaultdict(int))
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
            previous_count = int(request_counts[current_request_key] or 0)
            new_count = previous_count + 1
            request_counts[current_request_key] = new_count
            if new_count >= 4:
                return _REPEATED_REDUNDANT_FILE_READ_MESSAGE
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
    request_counts[current_request_key] += 1
    return None


def suppress_redundant_shell_file_read(
    path_key: str,
    *,
    offset: int | None,
    limit: int | None,
    total_lines: int,
    content_fingerprint: str | None = None,
) -> str | None:
    """Return a hard stop when shell re-reads file content already seen.

    Shell reads such as ``cat file`` and ``sed -n 1,80p file`` bypass the
    read_file tool's exact argument cache. This shares the same range coverage
    map with read_file so a request cannot loop by switching read mechanisms.
    """
    state = _current_tool_invocation_state()
    if not isinstance(state, dict):
        return None

    coverage = state.get("file_read_coverage")
    if not isinstance(coverage, dict):
        return None

    normalized_path = _normalize_runtime_path_alias(str(path_key or "").strip())
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
            return _REPEATED_SHELL_FILE_READ_MESSAGE

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
    return _normalize_runtime_path_alias(str(path_key or "").strip()).lower()


def invalidate_file_read_tracking(*path_keys: Any) -> None:
    """Invalidate read coverage and exact read counts after a file changes."""
    state = _current_tool_invocation_state()
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

    request_counts = state.get("redundant_file_read_counts")
    if isinstance(request_counts, dict):
        for key in list(request_counts.keys()):
            key_path = str(key).split("\x1f", 1)[0]
            if _normalize_file_tracking_path(key_path) in candidates:
                request_counts.pop(key, None)

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
    """Keep failure history advisory; never short-circuit the next tool call."""
    return None


def read_only_skill_inspection_budget_blocker(
    tool_name: str,
    arguments: Any = None,
    *,
    invocation_category: str | None = None,
) -> str | None:
    """Do not reject inspection/setup tools as a control-flow shortcut."""
    return None


def read_only_skill_budget_requires_stateful_tools() -> bool:
    """Return False; keep the provider's tool list evidence-driven."""
    return False


def _tool_schema_name(tool_schema: Any) -> str:
    if isinstance(tool_schema, dict):
        function = tool_schema.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                return name
        name = tool_schema.get("name")
        if isinstance(name, str):
            return name
    return ""


def _read_only_skill_budget_blocked_tool_count(tool_name: str) -> int:
    state = _current_tool_invocation_state()
    if not isinstance(state, dict):
        return 0
    blocked_counts = state.get("skill_read_only_budget_blocked_tools")
    if not isinstance(blocked_counts, dict):
        return 0
    normalized_tool = str(tool_name or "").strip().casefold().replace("-", "_")
    try:
        return int(blocked_counts.get(normalized_tool) or 0)
    except (TypeError, ValueError):
        return 0


def filter_tools_for_read_only_skill_budget(tools: Any) -> Any:
    """Do not remove setup/inspection tools solely because a read budget was spent."""
    return tools


def filter_tools_for_advancing_actions(tools: Any) -> Any:
    """Remove setup/inspection tool schemas for a required advancing action.

    Keep free-form shell available, but prefer structured state-changing tools
    first. This gives the provider concrete mutation tools before the generic
    shell escape hatch, without removing shell for workflows whose skill only
    exposes a CLI.
    """
    if not isinstance(tools, list):
        return tools

    ranked: list[tuple[int, int, Any]] = []
    for index, tool_schema in enumerate(tools):
        name = _tool_schema_name(tool_schema)
        category = _tool_invocation_category(name)
        if category in {"read_only", "setup"}:
            continue
        normalized = str(name or "").strip().casefold().replace("-", "_")
        rank = 2 if normalized == "shell" else 0
        ranked.append((rank, index, tool_schema))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ranked]


def record_tool_invocation_result(
    tool_name: str,
    result: Any,
    *,
    arguments: Any = None,
    invocation_category: str | None = None,
) -> str | None:
    """Record generic per-request tool outcome and skill progress state."""
    state = _current_tool_invocation_state()
    if not isinstance(state, dict):
        return None

    category = _tool_invocation_category_from_arguments(
        tool_name,
        arguments,
        invocation_category,
    )
    normalized_tool = str(tool_name or "").strip().casefold().replace("-", "_")
    if (
        normalized_tool == "skill_marketplace"
        or _payload_indicates_skill_contract(arguments)
        or _payload_indicates_skill_contract(result)
    ):
        state["skill_contract_seen"] = True

    if _tool_invocation_has_substantive_progress(
        normalized_tool,
        arguments,
        result,
        category,
    ):
        state["skill_stateful_progress_seen"] = True

    if (
        _request_skill_contract_seen(state)
        and not _request_skill_stateful_progress_seen(state)
        and category in {"read_only", "setup"}
    ):
        count = int(state.get("skill_read_only_observations") or 0) + 1
        state["skill_read_only_observations"] = count
        if count >= _runtime_skill_read_only_budget():
            state["skill_read_only_budget_exhausted"] = True

    failures = state.get("consecutive_failures")
    if not isinstance(failures, list):
        return None

    signal = _tool_failure_signal(result)
    if signal is None:
        failures.clear()
        return None

    series_counts = state.get("series_counts")
    if isinstance(series_counts, dict):
        series_counts.clear()

    failures.append({"tool": str(tool_name or ""), "signal": signal})
    strategy_warning: str | None = None
    pattern_counts = state.get("failure_pattern_counts")
    if isinstance(pattern_counts, dict):
        fingerprint = _tool_failure_fingerprint(result)
        if fingerprint:
            pattern_key = f"{str(tool_name or '')}\x1f{signal}\x1f{fingerprint}"
            pattern_counts[pattern_key] += 1
            max_failures = int(state.get("max_consecutive_failures") or 3)
            if 2 <= int(pattern_counts[pattern_key]) < max(2, max_failures):
                strategy_warning = _REPEATED_TOOL_FAILURE_STRATEGY_WARNING
    max_failures = int(state.get("max_consecutive_failures") or 3)
    del failures[: -max(1, max_failures)]
    return strategy_warning


def get_tracked_tool_invocation_counts() -> dict[str, int]:
    """Return per-tool execution counts for the current request."""
    state = _current_tool_invocation_state()
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
    global _REQUEST_EXECUTION_HINTS_FALLBACK
    normalized = hints if isinstance(hints, dict) else {}
    owner = get_tool_owner()
    missing = object()
    with _REQUEST_EXECUTION_HINTS_LOCK:
        previous_owner_hints = _REQUEST_EXECUTION_HINTS_BY_OWNER.get(owner, missing)
        previous_fallback = _REQUEST_EXECUTION_HINTS_FALLBACK
        _REQUEST_EXECUTION_HINTS_BY_OWNER[owner] = normalized
        _REQUEST_EXECUTION_HINTS_FALLBACK = normalized
    token = _REQUEST_EXECUTION_HINTS.set(normalized)
    try:
        yield normalized
    finally:
        _REQUEST_EXECUTION_HINTS.reset(token)
        with _REQUEST_EXECUTION_HINTS_LOCK:
            if previous_owner_hints is missing:
                _REQUEST_EXECUTION_HINTS_BY_OWNER.pop(owner, None)
            else:
                _REQUEST_EXECUTION_HINTS_BY_OWNER[owner] = previous_owner_hints
            _REQUEST_EXECUTION_HINTS_FALLBACK = previous_fallback


def get_request_execution_hints() -> dict[str, Any]:
    """Return request-scoped execution hints for the current task."""
    hints = _REQUEST_EXECUTION_HINTS.get()
    if isinstance(hints, dict):
        return hints
    owner = get_tool_owner()
    with _REQUEST_EXECUTION_HINTS_LOCK:
        owner_hints = _REQUEST_EXECUTION_HINTS_BY_OWNER.get(owner)
        if isinstance(owner_hints, dict):
            return owner_hints
        if isinstance(_REQUEST_EXECUTION_HINTS_FALLBACK, dict):
            return _REQUEST_EXECUTION_HINTS_FALLBACK
    return {}


def current_session_fact_check_blocker() -> str | None:
    """Compatibility hook for older tools; same-session checks no longer block.

    The request context and final-answer synthesis already provide the
    same-session evidence path. Blocking live tools from this low-level helper
    incorrectly treats current-state questions as prior-history disputes and
    causes repeated alternate tool attempts.
    """
    return None


def explicit_unavailable_tool_request_blocker(tool_name: str | None) -> str | None:
    """Return a blocker when a request requires an unavailable named tool/MCP."""
    hints = get_request_execution_hints()
    requests = hints.get("unavailable_explicit_tool_requests") if isinstance(hints, dict) else None
    if not isinstance(requests, list) or not requests:
        return None

    names: list[str] = []
    seen: set[str] = set()
    for request in requests:
        if not isinstance(request, dict):
            continue
        name = str(request.get("name") or "").strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        seen.add(key)
        names.append(name)

    if not names:
        return None

    requested = ", ".join(names[:6])
    current_tool = str(tool_name or "this tool").strip() or "this tool"
    message = (
        "STOP_TOOL_LOOP: Error: requested tool unavailable. The newest request "
        f"explicitly requires unavailable tool/MCP capability: {requested}. "
        f"Do not use `{current_tool}` or any other tool as a substitute unless "
        "the user explicitly allowed an alternative. Report that the requested "
        "capability is unavailable."
    )
    mark_current_tool_invocation_guardrail(
        "explicit_unavailable_tool_request",
        message=message,
    )
    return message


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


def capture_tool_output(
    summary_output: Any,
    full_output: Any | None = None,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
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
        if isinstance(metadata, dict) and metadata:
            capture.metadata.update(metadata)
        record_shell_command_evidence(capture.tool_name, full_text)


def mark_current_tool_invocation_guardrail(
    reason: str,
    *,
    message: str | None = None,
) -> None:
    """Attach structured guardrail metadata to the active captured tool result."""
    normalized_reason = str(reason or "").strip()
    if not normalized_reason:
        return
    invocation_id = _CURRENT_TOOL_INVOCATION.get()
    if not invocation_id:
        return
    with _TOOL_OUTPUT_LOCK:
        capture = _ACTIVE_TOOL_INVOCATIONS.get(invocation_id)
        if capture is None:
            return
        capture.guardrail_stop = True
        capture.guardrail_reason = normalized_reason
        if message is not None:
            capture.guardrail_message = stringify_tool_output(message)


def mark_current_tool_invocation_progress_recorded() -> None:
    """Mark the active capture as already recorded by the tool base class."""
    invocation_id = _CURRENT_TOOL_INVOCATION.get()
    if not invocation_id:
        return
    with _TOOL_OUTPUT_LOCK:
        capture = _ACTIVE_TOOL_INVOCATIONS.get(invocation_id)
        if capture is not None:
            capture.progress_recorded = True


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
        if not capture.progress_recorded:
            record_tool_invocation_result(
                capture.tool_name,
                capture.full_output or capture.summary_output,
                arguments=capture.arguments_key,
            )
            capture.progress_recorded = True
        try:
            record_tool_capture_in_ledger(
                owner=capture.owner,
                tool_name=capture.tool_name,
                arguments=capture.arguments_key,
                summary_output=capture.summary_output,
                full_output=capture.full_output,
                category=classify_tool_invocation_category(
                    capture.tool_name,
                    capture.arguments_key,
                ),
                guardrail_stop=capture.guardrail_stop,
                guardrail_reason=capture.guardrail_reason,
                metadata=capture.metadata,
            )
        except Exception:
            pass

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
        if arguments_key:
            for index, item in enumerate(bucket):
                if (
                    (not tool_name or item.tool_name == tool_name)
                    and item.arguments_key == arguments_key
                ):
                    match_index = index
                    break
            if match_index is None:
                return None
        elif tool_name:
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
