"""Provider request context snapshotting for debugging agent behavior."""

from __future__ import annotations

import json
import os
import time
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spoon_bot.agent.tools.execution_context import (
    get_request_execution_hints,
    get_tracked_tool_invocation_counts,
    read_only_skill_budget_requires_stateful_tools,
)
from spoon_bot.utils.privacy import is_sensitive_env_var, mask_secrets


def context_snapshot_enabled() -> bool:
    """Return True when request context snapshots should be written."""
    value = (
        os.getenv("SPOON_BOT_CONTEXT_SNAPSHOT_ENABLED")
        or os.getenv("SPOON_BOT_LLM_CONTEXT_CAPTURE")
        or ""
    )
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _max_text_chars() -> int:
    raw = os.getenv("SPOON_BOT_CONTEXT_SNAPSHOT_MAX_CHARS", "40000")
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        parsed = 40000
    return max(1000, parsed)


def _include_tool_schemas() -> bool:
    value = os.getenv("SPOON_BOT_CONTEXT_SNAPSHOT_INCLUDE_TOOL_SCHEMAS", "")
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _known_secret_values() -> list[str]:
    values: list[str] = []
    for name, value in os.environ.items():
        if not value or len(value) < 8:
            continue
        if is_sensitive_env_var(name):
            values.append(value)
    return sorted(values, key=len, reverse=True)


def _mask_known_secrets(text: str) -> str:
    masked = mask_secrets(text)
    for value in _known_secret_values():
        if value in masked:
            masked = masked.replace(value, "***masked***")
    return masked


def _safe_text(value: Any, *, max_chars: int | None = None) -> str:
    limit = _max_text_chars() if max_chars is None else max_chars
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = _mask_known_secrets(text)
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}...[truncated {omitted} chars]"


def _message_field(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _serialize_message(message: Any) -> dict[str, Any]:
    role = _message_field(message, "role", "")
    content = _message_field(message, "content", "")
    synthetic = _is_synthetic_user_message(message)
    serialized: dict[str, Any] = {
        "role": str(role or ""),
        "content": _safe_text(content),
        "content_chars": len(str(content or "")),
        "synthetic": synthetic,
    }
    if synthetic and _is_internal_user_message(message):
        serialized["synthetic_reason"] = "internal_recovery"
    elif synthetic:
        serialized["synthetic_reason"] = "runtime_wrapper"
    name = _message_field(message, "name")
    if name:
        serialized["name"] = str(name)
    tool_call_id = _message_field(message, "tool_call_id")
    if tool_call_id:
        serialized["tool_call_id"] = str(tool_call_id)
    tool_calls = _message_field(message, "tool_calls")
    if tool_calls:
        serialized["tool_calls"] = _safe_payload(tool_calls, max_chars=8000)
    return serialized


def _serialize_message_summary(message: Any) -> dict[str, Any]:
    serialized = _serialize_message(message)
    content = str(serialized.get("content") or "")
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    summary = {
        "role": serialized.get("role", ""),
        "synthetic": serialized.get("synthetic", False),
        "first_line": _safe_text(first_line, max_chars=500),
        "content_excerpt": _safe_text(content, max_chars=1200),
        "content_chars": serialized.get("content_chars", 0),
    }
    if "synthetic_reason" in serialized:
        summary["synthetic_reason"] = serialized["synthetic_reason"]
    if "name" in serialized:
        summary["name"] = serialized["name"]
    if "tool_call_id" in serialized:
        summary["tool_call_id"] = serialized["tool_call_id"]
    if "tool_calls" in serialized:
        summary["tool_calls"] = _safe_payload(serialized["tool_calls"], max_chars=1200)
    return summary


def _message_role(message: Any) -> str:
    return str(_message_field(message, "role", "") or "")


def _message_content_text(message: Any) -> str:
    content = _message_field(message, "content", "")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except Exception:
        return str(content or "")


def _is_internal_user_message(message: Any) -> bool:
    return (
        _message_role(message).strip().casefold() == "user"
        and _message_content_text(message).lstrip().startswith("[INTERNAL ")
    )


_DEFAULT_NEXT_STEP_PROMPTS = {
    "Continue.",
}


def _is_synthetic_user_message(message: Any) -> bool:
    if _message_role(message).strip().casefold() != "user":
        return False
    content = _message_content_text(message).lstrip()
    return (
        content.startswith("[INTERNAL ")
        or content.startswith("[FINAL ANSWER SYNTHESIS INPUT]")
        or content.startswith("[ORIGINAL USER REQUEST]")
        or content.startswith("[TURN PRIORITY]:")
        or content.startswith("[CURRENT DATE]:")
        or content.startswith("## Active Request Context")
        or content.startswith("## Current Session Compact")
        or content in _DEFAULT_NEXT_STEP_PROMPTS
        or content.startswith("Focus on the user")
    )


def _message_diagnostics(messages: list[Any]) -> dict[str, Any]:
    role_counts: dict[str, int] = {}
    duplicate_user_content_count = 0
    seen_user_content: set[str] = set()
    latest_user = ""
    latest_non_internal_user = ""
    latest_real_user = ""
    latest_internal_user = ""

    for message in messages:
        role = _message_role(message).strip().casefold() or "unknown"
        role_counts[role] = role_counts.get(role, 0) + 1
        if role != "user":
            continue

        content = _message_content_text(message)
        if content in seen_user_content:
            duplicate_user_content_count += 1
        else:
            seen_user_content.add(content)

        first_line = content.strip().splitlines()[0] if content.strip() else ""
        latest_user = first_line
        if _is_internal_user_message(message):
            latest_internal_user = first_line
        elif not _is_synthetic_user_message(message):
            latest_non_internal_user = first_line
            latest_real_user = first_line

    internal_user_count = sum(1 for message in messages if _is_internal_user_message(message))
    synthetic_user_count = sum(1 for message in messages if _is_synthetic_user_message(message))
    real_user_count = sum(
        1
        for message in messages
        if _message_role(message).strip().casefold() == "user"
        and not _is_synthetic_user_message(message)
    )
    return {
        "role_counts": role_counts,
        "user_count": role_counts.get("user", 0),
        "assistant_count": role_counts.get("assistant", 0),
        "tool_message_count": role_counts.get("tool", 0),
        "internal_user_count": internal_user_count,
        "synthetic_user_count": synthetic_user_count,
        "real_user_count": real_user_count,
        "duplicate_user_content_count": duplicate_user_content_count,
        "latest_provider_user_first_line": _safe_text(latest_user, max_chars=600),
        "latest_user_first_line": _safe_text(latest_user, max_chars=600),
        "latest_non_internal_user_first_line": _safe_text(
            latest_non_internal_user,
            max_chars=600,
        ),
        "latest_real_user_first_line": _safe_text(latest_real_user, max_chars=600),
        "latest_internal_user_first_line": _safe_text(
            latest_internal_user,
            max_chars=600,
        ),
    }


def _safe_payload(value: Any, *, max_chars: int | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _safe_payload(item, max_chars=max_chars)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_safe_payload(item, max_chars=max_chars) for item in value]
    if isinstance(value, tuple):
        return [_safe_payload(item, max_chars=max_chars) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return _safe_text(value, max_chars=max_chars)
        return value
    return _safe_text(value, max_chars=max_chars)


def _canonical_safe_json(value: Any, *, max_chars: int | None = None) -> str:
    try:
        payload = _safe_payload(value, max_chars=max_chars)
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except Exception:
        return _safe_text(value, max_chars=max_chars)


def _stable_digest(value: Any, *, max_chars: int | None = None) -> str:
    return sha256(
        _canonical_safe_json(value, max_chars=max_chars).encode("utf-8", errors="replace")
    ).hexdigest()


def _normalized_compare_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _extract_messages(args: list[Any], kwargs: dict[str, Any]) -> Any:
    if "messages" in kwargs:
        return kwargs.get("messages")
    if args:
        return args[0]
    return []


def _looks_like_tool_schema_list(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        if isinstance(item.get("function"), dict) or isinstance(item.get("name"), str):
            continue
        return False
    return True


def _extract_tools(args: list[Any], kwargs: dict[str, Any]) -> list[Any]:
    tools = kwargs.get("tools")
    if isinstance(tools, list):
        return tools
    for item in args[1:]:
        if _looks_like_tool_schema_list(item):
            return item
    return []


def _tool_name(schema: Any) -> str:
    if not isinstance(schema, dict):
        return ""
    function = schema.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(schema.get("name") or "")


def _tool_description(schema: Any) -> str:
    if not isinstance(schema, dict):
        return ""
    function = schema.get("function")
    if isinstance(function, dict):
        return str(function.get("description") or "")
    return str(schema.get("description") or "")


def _tool_parameters(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return {}
    function = schema.get("function")
    if isinstance(function, dict):
        return function.get("parameters") or {}
    return schema.get("parameters") or {}


def _summarize_tools(tools: list[Any]) -> list[dict[str, Any]]:
    include_schema = _include_tool_schemas()
    summarized: list[dict[str, Any]] = []
    for schema in tools:
        parameters = _tool_parameters(schema)
        item: dict[str, Any] = {
            "name": _tool_name(schema),
            "description": _safe_text(_tool_description(schema), max_chars=1000),
        }
        if isinstance(parameters, dict):
            props = parameters.get("properties")
            if isinstance(props, dict):
                item["parameter_names"] = sorted(str(name) for name in props.keys())
            required = parameters.get("required")
            if isinstance(required, list):
                item["required"] = [str(name) for name in required]
        if include_schema:
            item["schema"] = _safe_payload(schema, max_chars=12000)
        summarized.append(item)
    return summarized


def _message_fingerprints(messages: list[Any]) -> list[dict[str, Any]]:
    fingerprints: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        content = _message_content_text(message)
        item: dict[str, Any] = {
            "index": index,
            "role": _message_role(message),
            "synthetic": _is_synthetic_user_message(message),
            "content_chars": len(content),
            "content_sha256": _stable_digest(content),
        }
        first_line = content.strip().splitlines()[0] if content.strip() else ""
        if first_line:
            item["first_line"] = _safe_text(first_line, max_chars=300)
        for key in ("name", "tool_call_id"):
            value = _message_field(message, key)
            if value:
                item[key] = _safe_text(value, max_chars=200)
        fingerprints.append(item)
    return fingerprints


def _request_kwargs_summary(kwargs: dict[str, Any]) -> dict[str, Any]:
    skipped = {"messages", "tools", "output_queue"}
    summary: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in skipped:
            continue
        if key == "system_msg" and isinstance(value, str):
            summary["system_msg"] = _safe_text(value)
            summary["system_msg_chars"] = len(value)
            summary["system_msg_has_active_request_context"] = (
                "## Active Request Context" in value
            )
            summary["system_msg_has_current_session_compact"] = (
                "## Current Session Compact" in value
            )
            continue
        summary[str(key)] = _safe_payload(value, max_chars=4000)
    if "output_queue" in kwargs:
        summary["output_queue"] = kwargs.get("output_queue") is not None
    return summary


def _runtime_memory_summary(agent_loop: Any) -> dict[str, Any]:
    try:
        messages = agent_loop._get_runtime_memory_messages()
    except Exception:
        messages = []
    roles: dict[str, int] = {}
    for message in messages or []:
        role = _message_field(message, "role", "")
        key = str(role or "unknown")
        roles[key] = roles.get(key, 0) + 1
    return {
        "message_count": len(messages or []),
        "roles": roles,
    }


def _request_context_summary(agent_loop: Any) -> dict[str, Any]:
    raw = getattr(agent_loop, "_current_request_context", None)
    if not isinstance(raw, dict):
        agent = getattr(agent_loop, "_agent", None)
        raw = getattr(agent, "_current_request_context", None)
    if not isinstance(raw, dict):
        return {}
    allowed = {
        "user_id",
        "session_key",
        "request_id",
        "trace_id",
        "task_id",
        "transport",
        "connection_id",
    }
    return {
        key: _safe_text(value, max_chars=1000)
        for key, value in raw.items()
        if key in allowed and value not in (None, "")
    }


def _message_turn_summary(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {"content_excerpt": _safe_text(message, max_chars=500)}
    content = message.get("content", "")
    summary: dict[str, Any] = {
        "role": _safe_text(message.get("role", ""), max_chars=80),
        "timestamp": _safe_text(message.get("timestamp", ""), max_chars=120),
        "content_excerpt": _safe_text(content, max_chars=800),
        "content_chars": len(str(content or "")),
    }
    for key in (
        "turn_id",
        "turn_state",
        "turn_state_updated_at",
        "message_id",
        "invoked_skills",
    ):
        value = message.get(key)
        if value not in (None, ""):
            summary[key] = _safe_payload(value, max_chars=1200)
    return summary


def _session_turn_summary(agent_loop: Any) -> dict[str, Any]:
    session = getattr(agent_loop, "_session", None)
    if session is None:
        return {}
    try:
        messages = session.get_messages() if hasattr(session, "get_messages") else []
    except Exception:
        messages = []
    if not isinstance(messages, list):
        messages = []

    latest_user: dict[str, Any] | None = None
    latest_pending_user: dict[str, Any] | None = None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().casefold()
        if role != "user":
            continue
        if latest_user is None:
            latest_user = message
        state = str(message.get("turn_state") or "").strip().casefold()
        if state == "pending":
            latest_pending_user = message
            break

    return {
        "session_key": _safe_text(getattr(session, "session_key", ""), max_chars=500),
        "message_count": len(messages),
        "metadata": _safe_payload(getattr(session, "metadata", {}), max_chars=2000),
        "latest_user_turn": (
            _message_turn_summary(latest_user) if latest_user is not None else {}
        ),
        "latest_pending_user_turn": (
            _message_turn_summary(latest_pending_user)
            if latest_pending_user is not None
            else {}
        ),
        "tail": [_message_turn_summary(message) for message in messages[-8:]],
    }


def _tool_invocation_diagnostics() -> dict[str, Any]:
    try:
        counts = get_tracked_tool_invocation_counts()
    except Exception:
        counts = {}
    try:
        requires_advancing = read_only_skill_budget_requires_stateful_tools()
    except Exception:
        requires_advancing = False
    return {
        "counts": counts,
        "read_only_skill_budget_requires_advancing_tools": bool(requires_advancing),
    }


def _parse_state_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    parsed: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return parsed
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        parsed[key] = value
    return parsed


def _wallet_state_summary(path: Path) -> dict[str, Any]:
    try:
        root = path.expanduser().resolve()
    except Exception:
        root = path
    state_path = root / "state.env"
    state = _parse_state_env(state_path)
    return {
        "path": _safe_text(str(root), max_chars=1000),
        "exists": root.exists(),
        "is_dir": root.is_dir(),
        "state_env_exists": state_path.is_file(),
        "keystore_exists": (root / "keystore.json").is_file(),
        "password_file_exists": (root / "pw.txt").is_file(),
        "address": _safe_text(state.get("ADDRESS", ""), max_chars=200),
        "network": _safe_text(state.get("NETWORK_KEY") or state.get("NETWORK") or "", max_chars=200),
        "chain_id": _safe_text(state.get("CHAIN_ID", ""), max_chars=80),
    }


def _runtime_environment_summary() -> dict[str, Any]:
    keys = [
        "HOME",
        "SPOON_BOT_WORKSPACE_PATH",
        "SPOON_BOT_WALLET_PATH",
        "AGENT_WALLET_DIR",
        "WALLET_ADDRESS",
        "WALLET_NETWORK",
        "CHAIN_ID",
    ]
    env = {
        key: _safe_text(os.environ.get(key, ""), max_chars=1000)
        for key in keys
        if os.environ.get(key)
    }

    wallet_roots: dict[str, dict[str, Any]] = {}
    configured_wallet = os.environ.get("SPOON_BOT_WALLET_PATH") or os.environ.get("AGENT_WALLET_DIR")
    if configured_wallet:
        wallet_roots["configured"] = _wallet_state_summary(Path(configured_wallet))
    home = os.environ.get("HOME")
    if home:
        wallet_roots["home"] = _wallet_state_summary(Path(home) / ".agent-wallet")
    workspace = os.environ.get("SPOON_BOT_WORKSPACE_PATH")
    if workspace:
        wallet_roots["workspace"] = _wallet_state_summary(Path(workspace) / ".agent-wallet")

    return {
        "env": env,
        "wallet_roots": wallet_roots,
    }


def _extract_request_marker_text(text: str, marker: str = "[USER REQUEST]:") -> str:
    text = str(text or "")
    start = text.rfind(marker)
    if start < 0:
        return ""
    tail = text[start + len(marker):].lstrip()
    lines: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if lines and stripped.startswith("[") and stripped.endswith(":"):
            break
        if lines and stripped.startswith("[") and "]:" in stripped:
            break
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _extract_request_section_text(text: str, heading: str) -> str:
    text = str(text or "")
    start = text.rfind(heading)
    if start < 0:
        return ""
    tail = text[start + len(heading):].lstrip()
    lines: list[str] = []
    for line in tail.splitlines():
        stripped = line.strip()
        if lines and not stripped:
            break
        if lines and stripped.startswith("[") and stripped.endswith("]"):
            break
        if lines and stripped.endswith(":") and not any(char.isspace() for char in stripped):
            break
        if lines and stripped in {
            "Verified chronological evidence:",
            "Latest status evidence:",
            "Verified tool milestones:",
            "Verified executed tool commands:",
            "Current final draft:",
            "[CURRENT FINAL DRAFT]",
        }:
            break
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _extract_embedded_user_request(text: str) -> str:
    for marker in (
        "[USER REQUEST]:",
        "[ORIGINAL USER REQUEST]",
        "Newest user request:",
    ):
        if marker == "[USER REQUEST]:":
            extracted = _extract_request_marker_text(text, marker)
        else:
            extracted = _extract_request_section_text(text, marker)
        if extracted:
            return extracted
    return ""


def _extract_active_request(system_prompt: str) -> str:
    return _safe_text(
        _extract_embedded_user_request(str(system_prompt or "")),
        max_chars=2000,
    )


def _extract_active_request_from_messages(messages: list[Any]) -> str:
    for message in reversed(messages):
        if _message_role(message).strip().casefold() != "user":
            continue
        if _is_synthetic_user_message(message):
            continue
        content = _message_content_text(message).strip()
        if content:
            return _safe_text(content, max_chars=2000)

    for message in reversed(messages):
        if _message_role(message).strip().casefold() != "user":
            continue
        content = _message_content_text(message)
        marked = _extract_embedded_user_request(content)
        if marked:
            return _safe_text(marked, max_chars=2000)
    return ""


def _snapshot_dir(agent_loop: Any) -> Path:
    configured = os.getenv("SPOON_BOT_CONTEXT_SNAPSHOT_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    workspace = Path(getattr(agent_loop, "workspace", Path.cwd()))
    return workspace / ".spoon-bot" / "llm_context_snapshots"


def _build_snapshot_summary(
    event: dict[str, Any],
    *,
    messages: list[Any],
    tools: list[Any],
) -> dict[str, Any]:
    diagnostics_keys = [
        "role_counts",
        "user_count",
        "assistant_count",
        "tool_message_count",
        "internal_user_count",
        "synthetic_user_count",
        "real_user_count",
        "duplicate_user_content_count",
        "latest_provider_user_first_line",
        "latest_user_first_line",
        "latest_non_internal_user_first_line",
        "latest_real_user_first_line",
        "latest_internal_user_first_line",
    ]
    return {
        "ts": event.get("ts"),
        "monotonic": event.get("monotonic"),
        "label": event.get("label"),
        "provider": event.get("provider"),
        "model": event.get("model"),
        "base_url": event.get("base_url"),
        "session_key": event.get("session_key"),
        "user_id": event.get("user_id"),
        "request_id": event.get("request_id"),
        "trace_id": event.get("trace_id"),
        "request_context": event.get("request_context", {}),
        "workspace": event.get("workspace"),
        "request_kwargs": event.get("request_kwargs", {}),
        "active_user_request": event.get("active_user_request", ""),
        "active_user_request_source": event.get("active_user_request_source", ""),
        "active_user_request_from_messages": event.get(
            "active_user_request_from_messages",
            "",
        ),
        "context_health": event.get("context_health", {}),
        "provider_request_digest": event.get("provider_request_digest", ""),
        "system_prompt_sha256": event.get("system_prompt_sha256", ""),
        "message_fingerprints_tail": event.get("message_fingerprints", [])[-12:],
        "message_count": event.get("message_count", 0),
        "tool_count": event.get("tool_count", 0),
        "tool_names": [_tool_name(schema) for schema in tools if _tool_name(schema)],
        "runtime_memory": event.get("runtime_memory", {}),
        "runtime_wallet_roots": (
            (event.get("runtime_environment") or {}).get("wallet_roots", {})
        ),
        "tool_invocation_diagnostics": event.get(
            "tool_invocation_diagnostics",
            {},
        ),
        "session_turns": {
            "session_key": (event.get("session_turns") or {}).get("session_key", ""),
            "message_count": (event.get("session_turns") or {}).get("message_count", 0),
            "latest_user_turn": (event.get("session_turns") or {}).get(
                "latest_user_turn",
                {},
            ),
            "latest_pending_user_turn": (event.get("session_turns") or {}).get(
                "latest_pending_user_turn",
                {},
            ),
        },
        "diagnostics": {
            key: event.get(key)
            for key in diagnostics_keys
            if key in event
        },
        "tail_messages": [
            _serialize_message_summary(message)
            for message in messages[-8:]
        ],
    }


def _build_context_health(event: dict[str, Any]) -> dict[str, Any]:
    warnings: list[str] = []
    notes: list[str] = []

    request_context = event.get("request_context") or {}
    missing_request_fields = [
        field
        for field in ("user_id", "session_key", "request_id", "trace_id", "transport")
        if not request_context.get(field)
    ]
    if missing_request_fields:
        warnings.append("missing_request_metadata")

    if not event.get("active_user_request"):
        warnings.append("active_user_request_missing")
    if int(event.get("real_user_count") or 0) <= 0:
        warnings.append("no_real_user_message_in_provider_request")
    if int(event.get("duplicate_user_content_count") or 0) > 0:
        warnings.append("duplicate_user_content")
    if (
        int(event.get("synthetic_user_count") or 0) > 0
        and event.get("latest_provider_user_first_line")
        and event.get("latest_provider_user_first_line")
        != event.get("latest_real_user_first_line")
    ):
        notes.append("latest_provider_user_is_runtime_context")

    active_from_messages = _normalized_compare_text(
        event.get("active_user_request_from_messages")
    )
    active = _normalized_compare_text(event.get("active_user_request"))
    if active and active_from_messages and active != active_from_messages:
        notes.append("system_prompt_active_request_overrode_messages")

    session_turns = event.get("session_turns") or {}
    latest_pending = session_turns.get("latest_pending_user_turn") or {}
    pending_excerpt = _normalized_compare_text(latest_pending.get("content_excerpt"))
    if pending_excerpt and active:
        comparable = pending_excerpt[: min(len(pending_excerpt), len(active), 500)]
        if comparable and comparable not in active and comparable not in active_from_messages:
            warnings.append("active_request_differs_from_pending_session_turn")

    tool_diag = event.get("tool_invocation_diagnostics") or {}
    if tool_diag.get("read_only_skill_budget_requires_advancing_tools"):
        warnings.append("read_only_skill_budget_exhausted_needs_advancing_tool")

    return {
        "ok": not warnings,
        "warnings": warnings,
        "notes": notes,
        "missing_request_fields": missing_request_fields,
    }


def write_llm_context_snapshot(
    agent_loop: Any,
    *,
    label: str,
    args: list[Any],
    kwargs: dict[str, Any],
    response_note: str | None = None,
) -> Path | None:
    """Write the exact provider request context after runtime guard rewriting."""
    if not context_snapshot_enabled():
        return None

    messages_value = _extract_messages(args, kwargs)
    messages = list(messages_value) if isinstance(messages_value, (list, tuple)) else []
    tools = _extract_tools(args, kwargs)
    diagnostics = _message_diagnostics(messages)
    request_hints = get_request_execution_hints() or {}
    request_context = _request_context_summary(agent_loop)
    session_turns = _session_turn_summary(agent_loop)
    tool_invocation_diagnostics = _tool_invocation_diagnostics()

    agent = getattr(agent_loop, "_agent", None)
    system_prompt = getattr(agent, "system_prompt", "") if agent is not None else ""
    active_request_from_system = _extract_active_request(str(system_prompt or ""))
    active_request_from_messages = _extract_active_request_from_messages(messages)
    active_user_request = active_request_from_system or active_request_from_messages
    active_user_request_source = (
        "system_prompt"
        if active_request_from_system
        else ("messages" if active_request_from_messages else "")
    )
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "monotonic": time.monotonic(),
        "label": label,
        "provider": str(getattr(agent_loop, "provider", "") or ""),
        "model": str(getattr(agent_loop, "model", "") or ""),
        "base_url": _safe_text(str(getattr(agent_loop, "base_url", "") or ""), max_chars=1000),
        "session_key": str(
            request_context.get("session_key")
            or getattr(agent_loop, "session_key", "")
            or ""
        ),
        "user_id": str(
            request_context.get("user_id")
            or getattr(agent_loop, "user_id", "")
            or ""
        ),
        "request_id": request_context.get("request_id", ""),
        "trace_id": request_context.get("trace_id", ""),
        "request_context": request_context,
        "workspace": str(getattr(agent_loop, "workspace", "") or ""),
        "request_kwargs": _request_kwargs_summary(kwargs),
        "provider_context_injection": _safe_payload(
            getattr(agent_loop, "_last_provider_context_injection", {}) or {},
            max_chars=4000,
        ),
        "request_execution_hints": _safe_payload(request_hints, max_chars=12000),
        "tool_invocation_diagnostics": _safe_payload(
            tool_invocation_diagnostics,
            max_chars=8000,
        ),
        "session_turns": session_turns,
        "active_user_request": active_user_request,
        "active_user_request_source": active_user_request_source,
        "active_user_request_from_messages": active_request_from_messages,
        "runtime_environment": _runtime_environment_summary(),
        "runtime_memory": _runtime_memory_summary(agent_loop),
        "system_prompt_sha256": _stable_digest(system_prompt),
        "system_prompt": _safe_text(
            system_prompt,
        ),
        "next_step_prompt_sha256": _stable_digest(
            getattr(agent, "next_step_prompt", "") if agent is not None else ""
        ),
        "next_step_prompt": _safe_text(
            getattr(agent, "next_step_prompt", "") if agent is not None else "",
        ),
        "provider_tail_prompt": _safe_text(
            getattr(agent, "_spoon_bot_provider_tail_prompt", "")
            if agent is not None
            else "",
        ),
        "message_count": len(messages),
        **diagnostics,
        "message_fingerprints": _message_fingerprints(messages),
        "messages": [_serialize_message(message) for message in messages],
        "tool_count": len(tools),
        "tools_sha256": _stable_digest(_summarize_tools(tools)),
        "tools": _summarize_tools(tools),
    }
    event["provider_request_digest"] = _stable_digest(
        {
            "messages": [_serialize_message(message) for message in messages],
            "tools": event["tools"],
            "request_kwargs": event["request_kwargs"],
            "system_prompt_sha256": event["system_prompt_sha256"],
            "next_step_prompt_sha256": event["next_step_prompt_sha256"],
        }
    )
    event["context_health"] = _build_context_health(event)
    event["summary"] = _build_snapshot_summary(event, messages=messages, tools=tools)
    if response_note:
        event["response_note"] = _safe_text(response_note, max_chars=4000)

    target_dir = _snapshot_dir(agent_loop)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"llm_context_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str))
        handle.write("\n")
    summary_target = (
        target_dir / f"llm_context_summary_{datetime.now(timezone.utc):%Y%m%d}.jsonl"
    )
    with summary_target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event["summary"], ensure_ascii=False, default=str))
        handle.write("\n")
    return target
