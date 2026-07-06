"""State and persistence helpers for AgentLoop.

ponytail: physical split only; behavior stays on AgentLoop.
"""

from __future__ import annotations

import inspect
import json
import re
import shlex
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

try:
    from spoon_ai.schema import ToolCall as CoreToolCall, AgentState
except ImportError as e:
    logger.error(f"spoon-core SDK is required: {e}")
    raise ImportError(
        "spoon-bot requires spoon-core SDK. Install with: pip install spoon-ai-sdk"
    ) from e

from spoon_bot.agent.tools.execution_context import (
    bind_tool_owner,
    bind_tool_workspace,
    build_tool_owner_key,
)
from spoon_bot.skills.zip_install import InstalledSkillZip, install_skill_zip_archive
from spoon_bot.subagent.tools import SubagentTool
from spoon_bot.utils.retry import (
    DEFAULT_RETRY_CONFIG,
    RetryConfig,
    is_context_overflow_error,
    with_provider_retry,
)

if TYPE_CHECKING:
    pass

AgentLoop: Any = None

_ATTACHMENT_CONTEXT_HEADER = "Attached workspace files (source of truth for this request):"
_ATTACHMENT_ONLY_PLACEHOLDER = "The user attached files without extra text. Inspect the files and answer based on their contents."
_SANDBOX_WORKSPACE_ROOT = "/workspace"
_MISSING = object()
_TURN_STATE_PENDING = "pending"
_TURN_STATE_COMPLETED = "completed"
_TURN_STATE_INTERRUPTED = "interrupted"
_TURN_STATE_SUPERSEDED = "superseded"
_DEFAULT_NON_SHELL_ACTIVE_TOOL_TIMEOUT = 120.0
_DEFAULT_POST_TOOL_RESULT_SILENCE_TIMEOUT = 30.0
_DEFAULT_INTERNAL_RECOVERY_TIMEOUT = 180.0
_DEFAULT_PROVIDER_ASK_TIMEOUT = 0.0


def _workspace_root_path(workspace: Path | str | None) -> Path:
    """Resolve the workspace root used to validate persisted file references."""
    return Path(workspace or Path.home() / ".spoon-bot" / "workspace").expanduser().resolve()


def _resolve_workspace_file(path_str: str, workspace: Path | str | None) -> Path | None:
    """Resolve a file path and ensure it stays within the configured workspace."""
    candidate = str(path_str or "").strip()
    if not candidate:
        return None

    workspace_root = _workspace_root_path(workspace)
    sandbox_root = _SANDBOX_WORKSPACE_ROOT.rstrip("/")
    try:
        if candidate.startswith("/"):
            normalized = Path(candidate).as_posix()
            workspace_root_str = workspace_root.as_posix().rstrip("/")
            if normalized == sandbox_root or normalized.startswith(sandbox_root + "/"):
                relative = normalized[len(sandbox_root) :].lstrip("/")
                resolved = (workspace_root / relative).resolve(strict=True)
            elif normalized == workspace_root_str or normalized.startswith(
                workspace_root_str + "/"
            ):
                relative = normalized[len(workspace_root_str) :].lstrip("/")
                resolved = (workspace_root / relative).resolve(strict=True)
            else:
                resolved = Path(candidate).expanduser().resolve(strict=True)
        else:
            resolved = (workspace_root / candidate).resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None

    if resolved != workspace_root and workspace_root not in resolved.parents:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _normalize_media_list(raw: Any) -> list[str]:
    """Normalize stored media payloads to a list of non-empty file paths."""
    if not isinstance(raw, list):
        return []

    items: list[str] = []
    for item in raw:
        if isinstance(item, str):
            value = item.strip()
        else:
            value = str(item).strip() if item is not None else ""
        if value:
            items.append(value)
    return items


def _sanitize_media_list(raw: Any, workspace: Path | str | None) -> list[str]:
    """Keep only workspace-backed media paths from persisted session metadata."""
    return [
        path
        for path in _normalize_media_list(raw)
        if _resolve_workspace_file(path, workspace) is not None
    ]


def _normalize_attachment_refs(raw: Any) -> list[dict[str, Any]]:
    """Normalize stored attachment references from session metadata."""
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("workspace_path") or item.get("uri") or "").strip()
        if not uri:
            continue
        attachment = dict(item)
        attachment["uri"] = uri
        attachment.setdefault("workspace_path", uri)
        normalized.append(attachment)
    return normalized


def _sanitize_attachment_refs(raw: Any, workspace: Path | str | None) -> list[dict[str, Any]]:
    """Keep only workspace-backed attachment references from persisted metadata."""
    sanitized: list[dict[str, Any]] = []
    for item in _normalize_attachment_refs(raw):
        uri = str(item.get("workspace_path") or item.get("uri") or "").strip()
        if _resolve_workspace_file(uri, workspace) is None:
            continue
        sanitized.append(item)
    return sanitized


def _attachment_context_entries(
    attachments: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], str]]:
    """Return normalized attachment entries with resolved display paths."""
    normalized_items: list[tuple[dict[str, Any], str]] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("workspace_path") or item.get("uri") or "").strip()
        if uri:
            normalized_items.append((item, uri))
    return normalized_items


def _build_attachment_context_lines(attachments: list[dict[str, Any]]) -> list[str]:
    """Build the synthetic attachment context block appended to prompts."""
    normalized_items = _attachment_context_entries(attachments)
    if not normalized_items:
        return []

    lines = [_ATTACHMENT_CONTEXT_HEADER]
    for item, uri in normalized_items:
        name = str(item.get("name") or item.get("file_name") or "").strip()
        mime_type = str(item.get("mime_type") or item.get("file_type") or "").strip()
        size = item.get("size") if "size" in item else item.get("file_size")
        suffix = []
        if name:
            suffix.append(f"name: {name}")
        if mime_type:
            suffix.append(f"mime: {mime_type}")
        if isinstance(size, int) and size > 0:
            suffix.append(f"size: {size} bytes")
        line = f"- {uri}"
        if suffix:
            line += f" ({', '.join(suffix)})"
        lines.append(line)
    lines.append(
        "Use these attached workspace files as the primary source of truth for this request."
    )
    return lines


def _ensure_attachment_context(content: str, attachments: list[dict[str, Any]]) -> str:
    """Append attachment path context unless it is already present in content."""
    if not attachments:
        return content

    text = content if isinstance(content, str) else str(content)
    normalized_items = _attachment_context_entries(attachments)
    uris = [uri for _, uri in normalized_items]
    if not uris:
        return text
    if _ATTACHMENT_CONTEXT_HEADER in text and all(uri in text for uri in uris):
        return text

    lines = [text.strip()] if text.strip() else [_ATTACHMENT_ONLY_PLACEHOLDER]
    lines.extend(["", *_build_attachment_context_lines(attachments)])
    return "\n".join(lines)


def _strip_attachment_context(content: str, attachments: list[dict[str, Any]]) -> str:
    """Remove the exact synthetic attachment block so sessions keep user-authored text only."""
    if not attachments:
        return content

    text = content if isinstance(content, str) else str(content)
    if _ATTACHMENT_CONTEXT_HEADER not in text:
        return text

    context_lines = _build_attachment_context_lines(attachments)
    if not context_lines:
        return text

    delimiter = f"\n\n{context_lines[0]}\n"
    if delimiter not in text:
        return text

    prefix, suffix = text.split(delimiter, 1)
    expected_suffix = "\n".join(context_lines[1:])
    if suffix != expected_suffix:
        return text
    if prefix == _ATTACHMENT_ONLY_PLACEHOLDER:
        return ""
    return prefix


class LoopStateMixin:
    def _estimate_token_count(self) -> int:
        """Rough token estimate for current session messages (~4 chars per token)."""
        return sum(len(m.get("content", "")) for m in self._session.messages) // 4

    def _trim_context_if_needed(self) -> int:
        """No-op placeholder.

        History compaction is deliberately runtime-only now. Persisted session
        history remains the source of truth so search/recovery keeps working.
        """
        return 0

    def _warn_dropped_refs(self, *, kind: str, dropped: list[str]) -> None:
        """Log dropped persisted refs once per (session, ref) pair."""
        unique_refs = [ref for ref in dict.fromkeys(filter(None, dropped))]
        if not unique_refs:
            return

        cache_map = (
            self._warned_invalid_attachment_refs
            if kind == "attachment"
            else self._warned_invalid_media_refs
        )
        seen = cache_map.setdefault(self.session_key, set())

        new_refs = [ref for ref in unique_refs if ref not in seen]
        if new_refs:
            seen.update(new_refs)
            preview = ", ".join(new_refs[:5])
            suffix = f" (+{len(new_refs) - 5} more)" if len(new_refs) > 5 else ""
            logger.warning(
                f"Dropped invalid persisted {kind} refs outside workspace during "
                f"history sync (session={self.session_key}): {preview}{suffix}"
            )
        else:
            logger.debug(
                f"Dropped invalid persisted {kind} refs outside workspace during "
                f"history sync (session={self.session_key}): "
                f"{len(unique_refs)} already-known ref(s)"
            )

    @staticmethod
    def _message_token_count(message: str) -> int:
        """Approximate token count for lightweight history-scope decisions."""
        return len(re.findall(r"[0-9A-Za-z\u4e00-\u9fff]+", message or ""))

    @classmethod
    def _history_rehydrate_scope(
        cls,
        upcoming_message: str | None,
        *,
        history_messages: list[dict[str, Any]] | None = None,
    ) -> str:
        """Choose how much prior session history should enter the runtime context."""
        normalized = (upcoming_message or "").strip()
        if not normalized:
            return "full"

        # Runtime history is intentionally isolated by default. Persisted session
        # history remains searchable/observable, but the active agent loop should
        # not inherit unfinished tool chains or prior-task intent for a new user
        # turn. This mirrors a Claude Code-style active-request boundary without
        # deriving control flow from prompt text.
        return "minimal"

    @staticmethod
    def _filter_rehydratable_history(
        history_messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop interrupted/superseded turn fragments from runtime rehydration."""
        filtered: list[dict[str, Any]] = []
        skipping_aborted_turn = False

        for message in history_messages:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role") or "").strip().lower()
            if role == "user":
                turn_state = AgentLoop._turn_state_of_message(message)
                if turn_state in {_TURN_STATE_INTERRUPTED, _TURN_STATE_SUPERSEDED}:
                    skipping_aborted_turn = True
                    continue
                skipping_aborted_turn = False
                filtered.append(message)
                continue

            if skipping_aborted_turn and role in {"assistant", "tool"}:
                continue

            filtered.append(message)

        return filtered

    async def _sync_runtime_history_from_session(
        self,
        *,
        upcoming_message: str | None = None,
    ) -> int:
        """Sync persisted session history into spoon-core runtime memory."""
        if not self._agent:
            return 0

        memory = getattr(self._agent, "memory", None)
        if memory is None or not hasattr(memory, "clear"):
            return 0

        try:
            memory.clear()
        except Exception as exc:
            logger.warning(f"Failed to clear runtime memory before history sync: {exc}")
            return 0

        injected_count = 0
        history_messages = (
            self._session.get_messages()
            if hasattr(self._session, "get_messages")
            else self._session.get_history()
        )
        if not isinstance(history_messages, list):
            history_messages = []
        history_messages = AgentLoop._filter_rehydratable_history(history_messages)

        rehydrate_scope = AgentLoop._history_rehydrate_scope(
            upcoming_message,
            history_messages=history_messages,
        )
        if rehydrate_scope == "minimal":
            history_messages = []

        try:
            from spoon_ai.schema import Function as _CoreFunction  # noqa: F401
        except Exception:
            _CoreFunction = None  # type: ignore[assignment]

        def _rehydrate_tool_calls(raw: Any) -> list | None:
            if not raw or _CoreFunction is None:
                return None
            if not isinstance(raw, list):
                return None
            out: list = []
            for item in raw:
                if isinstance(item, CoreToolCall):
                    out.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                tc_id = item.get("id")
                if not tc_id:
                    continue
                fn = item.get("function") or {}
                if isinstance(fn, dict):
                    name = fn.get("name") or ""
                    arguments = fn.get("arguments") or ""
                else:
                    name = getattr(fn, "name", "") or ""
                    arguments = getattr(fn, "arguments", "") or ""
                if arguments is not None and not isinstance(arguments, str):
                    try:
                        arguments = json.dumps(arguments, ensure_ascii=False)
                    except Exception:
                        arguments = str(arguments)
                try:
                    out.append(
                        CoreToolCall(
                            id=str(tc_id),
                            type=str(item.get("type") or "function"),
                            function=_CoreFunction(name=str(name), arguments=str(arguments or "")),
                        )
                    )
                except Exception:
                    continue
            return out or None

        for msg in history_messages:
            role = str(msg.get("role", "")).strip().lower()
            if role not in {"user", "assistant", "tool"}:
                continue
            if role == "user" and AgentLoop._turn_state_of_message(msg) in {
                _TURN_STATE_INTERRUPTED,
                _TURN_STATE_SUPERSEDED,
            }:
                continue

            assistant_tool_calls = None
            if role == "assistant":
                assistant_tool_calls = _rehydrate_tool_calls(msg.get("tool_calls"))
                if assistant_tool_calls is None:
                    continue
            content = msg.get("content", "")

            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)

            raw_media = _normalize_media_list(msg.get("media"))
            media = _sanitize_media_list(raw_media, self.workspace)
            if raw_media and len(media) != len(raw_media):
                self._warn_dropped_refs(
                    kind="media",
                    dropped=[ref for ref in raw_media if ref not in media],
                )

            raw_attachments = _normalize_attachment_refs(msg.get("attachments"))
            attachments = _sanitize_attachment_refs(raw_attachments, self.workspace)
            if raw_attachments and len(attachments) != len(raw_attachments):
                kept = {
                    str(item.get("workspace_path") or item.get("uri") or "").strip()
                    for item in attachments
                }
                dropped = [
                    str(item.get("workspace_path") or item.get("uri") or "").strip()
                    for item in raw_attachments
                    if str(item.get("workspace_path") or item.get("uri") or "").strip() not in kept
                ]
                self._warn_dropped_refs(kind="attachment", dropped=dropped)

            content = self._build_runtime_message_content(
                role,
                content,
                media=media,
                attachments=attachments,
            )

            extra_kwargs: dict[str, Any] = {}
            if role == "tool":
                tc_id = msg.get("tool_call_id")
                if tc_id:
                    extra_kwargs["tool_call_id"] = str(tc_id)
                tool_name = msg.get("name") or msg.get("tool_name")
                if tool_name:
                    extra_kwargs["tool_name"] = str(tool_name)
            elif role == "assistant":
                if assistant_tool_calls:
                    extra_kwargs["tool_calls"] = assistant_tool_calls

            try:
                await self._agent.add_message(role, content, **extra_kwargs)
                injected_count += 1
            except Exception as exc:
                logger.warning(
                    f"Failed to inject session history message "
                    f"(role={role}, index={injected_count}): {exc}"
                )

        return injected_count

    async def _prepare_request_context(self, upcoming_message: str | None = None) -> None:
        """Prepare request context by injecting persisted history into runtime memory.

        Persisted history (``self._session.messages``) is the authoritative
        store and is never trimmed here — see :meth:`_trim_context_if_needed`.
        Runtime memory is rebuilt from the persisted transcript for the next
        provider request. If that rebuilt runtime context is already close to
        the configured ``context_window``, older runtime-only history is
        compacted before the provider call starts.
        """
        self._refresh_recent_turn_notice()
        self._refresh_recent_invoked_skill_contexts()
        trimmed_count = self._trim_context_if_needed()
        persisted_repaired = self._normalize_persisted_session_tool_context()
        session = getattr(self, "_session", None)
        if session is None:
            session_history = []
        else:
            session_history = (
                session.get_messages()
                if hasattr(session, "get_messages")
                else session.get_history()
            )
        if not isinstance(session_history, list):
            session_history = []
        session_history = AgentLoop._filter_rehydratable_history(session_history)
        rehydrate_scope = AgentLoop._history_rehydrate_scope(
            upcoming_message,
            history_messages=session_history,
        )
        injected_count = await self._sync_runtime_history_from_session(
            upcoming_message=upcoming_message
        )

        # Repair tool-call ordering after history injection - session storage
        # may not preserve tool_call_id metadata, producing orphaned tool
        # messages that providers (OpenAI, Gemini, etc.) reject.
        repaired = 0
        messages_ref: list | None = None
        if self._agent and hasattr(self._agent, "memory"):
            messages_ref = getattr(self._agent.memory, "messages", None)
            if isinstance(messages_ref, list):
                repaired = self._normalize_runtime_tool_context(messages_ref)

        compressed = 0
        runtime_tokens_before = self._estimate_runtime_tokens()
        runtime_tokens = runtime_tokens_before
        trigger_budget = self._runtime_compaction_trigger_budget()
        if isinstance(messages_ref, list) and messages_ref and runtime_tokens > trigger_budget:
            try:
                compressed = self._compress_runtime_context(
                    force=True,
                    budget_tokens=trigger_budget,
                )
            except Exception as exc:
                logger.debug(f"Proactive runtime compaction skipped: {exc}")
            if runtime_tokens > trigger_budget and compressed == 0:
                compressed = self._force_compress_runtime_context()
            runtime_tokens = self._estimate_runtime_tokens()
            if compressed:
                AgentLoop._queue_runtime_notice(
                    self,
                    kind="runtime_compaction",
                    stage="preflight",
                    compressed_actions=compressed,
                    estimated_tokens_before=runtime_tokens_before,
                    estimated_tokens_after=runtime_tokens,
                    trigger_budget=trigger_budget,
                )
        session_tokens = self._estimate_token_count()

        logger.info(
            f"Session context prepared: session={self.session_key}, "
            f"injected_messages={injected_count}, "
            f"runtime_tokens~{runtime_tokens}, session_tokens~{session_tokens}, "
            f"trimmed_messages={trimmed_count}, rehydrate_scope={rehydrate_scope}"
            + (f", repaired_session_tool_order={persisted_repaired}" if persisted_repaired else "")
            + (f", repaired_tool_order={repaired}" if repaired else "")
            + (f", compressed_actions={compressed}" if compressed else "")
        )

    def _runtime_memory_snapshot_index(self) -> int:
        """Return the current length of runtime memory."""
        try:
            return len(AgentLoop._get_runtime_memory_messages(self))
        except Exception:
            return 0

    def _normalize_persisted_session_tool_context(self) -> int:
        """Repair persisted session tool traces before injecting them into runtime memory."""
        messages = getattr(getattr(self, "_session", None), "messages", None)
        if not isinstance(messages, list) or not messages:
            return 0

        try:
            before = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            before = repr(messages)

        repaired = AgentLoop._normalize_runtime_tool_context(messages, finalized=True)

        try:
            after = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            after = repr(messages)

        if before != after:
            try:
                self.sessions.save(self._session)
            except Exception as exc:
                logger.debug(f"Failed to save normalized session tool context: {exc}")
            return max(repaired, 1)
        return repaired

    def _normalize_runtime_memory_before_run(self, label: str) -> int:
        """Repair runtime tool-call context immediately before provider execution."""
        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
            if not isinstance(messages, list) or not messages:
                return 0
            repaired = AgentLoop._normalize_runtime_tool_context(messages)
            if repaired:
                logger.info(
                    f"[{label}] Normalized {repaired} runtime tool-context "
                    "entry/entries before model run"
                )
            return repaired
        except Exception as exc:
            logger.debug(f"[{label}] Runtime tool-context normalization skipped: {exc}")
            return 0

    async def _ensure_runtime_user_tail_before_run(self, label: str) -> int:
        """Ensure provider-bound continuation requests end with a user turn."""
        if not AgentLoop._provider_requires_user_tail(self):
            return 0
        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
        except Exception as exc:
            logger.debug(f"[{label}] Runtime tail check skipped: {exc}")
            return 0

        if not isinstance(messages, list) or not messages:
            return 0

        last_role = str(AgentLoop._message_role_value(messages[-1]) or "").strip().lower()
        if last_role == "user":
            return 0

        agent = getattr(self, "_agent", None)
        add_message = getattr(agent, "add_message", None)
        if not callable(add_message):
            return 0

        prompt = getattr(agent, "next_step_prompt", None)
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = self.DEFAULT_NEXT_STEP_PROMPT

        await add_message("user", prompt.strip())
        logger.info(
            f"[{label}] Added continuation user turn before model run "
            f"(previous_tail_role={last_role or 'unknown'})"
        )
        return 1

    @staticmethod
    def _truncate_runtime_memory(self, start_index: int) -> None:
        """Remove runtime-only messages appended by an aborted turn."""
        if not isinstance(start_index, int) or start_index < 0:
            return
        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
        except Exception:
            return
        if isinstance(messages, list) and start_index < len(messages):
            del messages[start_index:]

    @staticmethod
    def _drain_agent_output_queue(self) -> None:
        """Discard queued stream chunks from an interrupted run."""
        agent = getattr(self, "_agent", None)
        output_queue = getattr(agent, "output_queue", None)
        if output_queue is None or not hasattr(output_queue, "empty"):
            return
        while True:
            try:
                if output_queue.empty():
                    break
                if hasattr(output_queue, "get_nowait"):
                    output_queue.get_nowait()
                else:
                    break
            except Exception:
                break

    def _capture_turn_tool_trace(self, start_index: int) -> list[dict[str, Any]]:
        """Capture tool-call artifacts added since ``start_index``."""
        if not isinstance(start_index, int) or start_index < 0:
            return []

        messages = AgentLoop._get_runtime_memory_messages(self)
        if not messages or start_index >= len(messages):
            return []

        captured: list[dict[str, Any]] = []
        for msg in messages[start_index:]:
            role = AgentLoop._stream_message_role(msg).lower()
            if role not in ("tool", "assistant"):
                continue

            content = AgentLoop._stream_message_attr(msg, "text_content", None)
            if not isinstance(content, str) or not content:
                content = AgentLoop._stream_message_attr(msg, "content", "") or ""
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)

            extras: dict[str, Any] = {}

            if role == "tool":
                tool_call_id = AgentLoop._stream_message_attr(
                    msg, "tool_call_id", None
                ) or AgentLoop._stream_message_attr(msg, "id", None)
                if tool_call_id:
                    extras["tool_call_id"] = str(tool_call_id)
                    extras.update(
                        AgentLoop._stream_tool_result_metadata_for_trace(
                            self,
                            tool_call_id,
                        )
                    )
                name = AgentLoop._stream_message_attr(msg, "name", None)
                if name:
                    extras["name"] = str(name)
            else:
                tool_calls = AgentLoop._stream_message_attr(msg, "tool_calls", None) or []
                serialized: list[dict[str, Any]] = []
                for tc in tool_calls:
                    tc_id = getattr(tc, "id", None) or (
                        tc.get("id") if isinstance(tc, dict) else None
                    )
                    tc_type = (
                        getattr(tc, "type", None)
                        or (tc.get("type") if isinstance(tc, dict) else None)
                        or "function"
                    )
                    fn = getattr(tc, "function", None) or (
                        tc.get("function") if isinstance(tc, dict) else None
                    )
                    if fn is not None:
                        tc_name = getattr(fn, "name", None) or (
                            fn.get("name") if isinstance(fn, dict) else None
                        )
                        tc_args = getattr(fn, "arguments", None) or (
                            fn.get("arguments") if isinstance(fn, dict) else None
                        )
                    else:
                        tc_name = getattr(tc, "name", None)
                        tc_args = getattr(tc, "arguments", None)

                    if tc_args is not None and not isinstance(tc_args, str):
                        try:
                            tc_args = json.dumps(tc_args, ensure_ascii=False)
                        except Exception:
                            tc_args = str(tc_args)

                    if tc_name or tc_id:
                        serialized.append(
                            {
                                "id": tc_id,
                                "type": tc_type,
                                "function": {
                                    "name": tc_name,
                                    "arguments": tc_args or "",
                                },
                            }
                        )
                if not serialized:
                    continue
                extras["tool_calls"] = serialized

            captured.append({"role": role, "content": content, "extras": extras})
        return captured

    def _persist_turn_tool_trace(self, start_index: int) -> int:
        """Append captured tool-trace messages to ``self._session``."""
        if not self._should_persist_tool_trace():
            return 0

        try:
            messages = AgentLoop._get_runtime_memory_messages(self)
            if isinstance(messages, list) and messages:
                AgentLoop._normalize_runtime_tool_context(messages)
            trace = self._capture_turn_tool_trace(start_index)
        except Exception as exc:
            logger.debug(f"Tool-trace capture failed: {exc}")
            return 0

        if not trace:
            return 0

        return AgentLoop._persist_tool_trace_entries(self, trace)

    @staticmethod
    def _tool_trace_entry_key(entry: dict[str, Any]) -> str:
        """Return a stable key for persisted tool trace de-duplication."""
        extras = entry.get("extras") if isinstance(entry.get("extras"), dict) else {}
        safe_extras = {
            key: value
            for key, value in extras.items()
            if key not in {"timestamp", "trace_key", "message_kind"}
        }
        payload = {
            "role": str(entry.get("role") or "").strip().lower(),
            "content": str(entry.get("content") or ""),
            "extras": safe_extras,
        }
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            return repr(payload)

    def _persisted_tool_trace_keys(self) -> set[str]:
        """Return trace keys already stored in the current session."""
        messages = getattr(getattr(self, "_session", None), "messages", None)
        if not isinstance(messages, list):
            return set()

        keys: set[str] = set()
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role not in {"assistant", "tool"}:
                continue
            if message.get("trace_key"):
                keys.add(str(message.get("trace_key")))
                continue
            extras = {
                key: value
                for key, value in message.items()
                if key not in {"role", "content", "timestamp"}
            }
            keys.add(
                AgentLoop._tool_trace_entry_key(
                    {
                        "role": role,
                        "content": message.get("content", ""),
                        "extras": extras,
                    }
                )
            )
        return keys

    def _persist_tool_trace_entries(self, trace: list[dict[str, Any]]) -> int:
        """Persist already-captured tool trace entries after generic pairing repair."""
        if not self._should_persist_tool_trace() or not trace:
            return 0

        messages: list[dict[str, Any]] = []
        for entry in trace:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").strip().lower()
            if role not in {"assistant", "tool"}:
                continue
            content = entry.get("content", "")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)
            extras = entry.get("extras") if isinstance(entry.get("extras"), dict) else {}
            messages.append({"role": role, "content": content, **extras})

        if not messages:
            return 0

        AgentLoop._normalize_runtime_tool_context(messages, finalized=True)
        existing_keys = AgentLoop._persisted_tool_trace_keys(self)

        persisted = 0
        for msg in messages:
            role = str(msg.get("role") or "").strip().lower()
            if role not in {"assistant", "tool"}:
                continue
            if role == "assistant" and not msg.get("tool_calls"):
                continue
            if role == "tool" and not msg.get("tool_call_id"):
                continue
            content = msg.get("content", "")
            extras = {
                key: value
                for key, value in msg.items()
                if key not in {"role", "content", "timestamp"}
            }
            entry = {"role": role, "content": content, "extras": extras}
            trace_key = AgentLoop._tool_trace_entry_key(entry)
            if trace_key in existing_keys:
                continue
            extras = {**extras, "trace_key": trace_key, "message_kind": "tool_trace"}
            try:
                self._session.add_message(role, content, **extras)
                existing_keys.add(trace_key)
                persisted += 1
            except Exception as exc:
                logger.debug(f"Tool-trace persist skipped a streamed message: {exc}")
        return persisted

    def _persist_turn_tool_trace_checkpoint(self, start_index: int | None = None) -> int:
        """Persist currently available tool evidence for in-flight recovery."""
        if not isinstance(start_index, int) or start_index < 0:
            return 0
        try:
            self._merge_turn_invoked_skills_from_runtime(start_index)
        except Exception as exc:
            logger.debug(f"Tool-trace checkpoint skill merge skipped: {exc}")
        try:
            persisted = self._persist_turn_tool_trace(start_index)
            if persisted:
                AgentLoop._persist_session_if_possible(self)
            return persisted
        except Exception as exc:
            logger.debug(f"Tool-trace checkpoint persist skipped: {exc}")
            return 0

    @staticmethod
    def _tool_call_name_and_arguments(tool_call: Any) -> tuple[str, Any]:
        """Extract a function/tool name and raw arguments from common tool-call shapes."""
        fn = getattr(tool_call, "function", None) or (
            tool_call.get("function") if isinstance(tool_call, dict) else None
        )
        if fn is not None:
            name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
            arguments = getattr(fn, "arguments", None) or (
                fn.get("arguments") if isinstance(fn, dict) else None
            )
            return str(name or ""), arguments

        name = getattr(tool_call, "name", None) or (
            tool_call.get("name") if isinstance(tool_call, dict) else None
        )
        arguments = getattr(tool_call, "arguments", None) or (
            tool_call.get("arguments") if isinstance(tool_call, dict) else None
        )
        return str(name or ""), arguments

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
        """Parse structured tool-call arguments without inspecting user prompt text."""
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str) and arguments.strip():
            try:
                parsed = json.loads(arguments)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    @staticmethod
    def _skill_name_from_workspace_path(path_value: Any) -> str | None:
        """Return the skill name for structured workspace skill paths."""
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        normalized = path_value.strip().replace("\\", "/")
        parts = [part for part in normalized.split("/") if part and part != "."]
        for index, part in enumerate(parts[:-1]):
            if part == "skills" and index + 1 < len(parts):
                name = parts[index + 1].strip()
                return name or None
        return None

    @staticmethod
    def _skill_names_from_argument_text(value: Any) -> list[str]:
        """Extract workspace skill names from structured tool argument text.

        This deliberately keys off the runtime skill path shape rather than
        product names, prompts, or command-specific routing rules.
        """
        if not isinstance(value, str) or not value.strip():
            return []

        try:
            tokens = shlex.split(value)
        except ValueError:
            tokens = value.split()

        names: list[str] = []
        for token in [value, *tokens]:
            skill_name = AgentLoop._skill_name_from_workspace_path(token)
            if skill_name and skill_name not in names:
                names.append(skill_name)
        return names

    @staticmethod
    def _iter_tool_argument_strings(value: Any) -> list[str]:
        """Return bounded string leaves from parsed tool arguments."""
        strings: list[str] = []
        stack: list[Any] = [value]
        while stack and len(strings) < 32:
            item = stack.pop()
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, (list, tuple)):
                stack.extend(item)
        return strings

    @staticmethod
    def _extract_skill_names_from_tool_call(tool_name: str, arguments: Any) -> list[str]:
        """Extract skill usage from explicit structured tool calls only."""
        name = str(tool_name or "").strip()
        parsed = AgentLoop._parse_tool_arguments(arguments)
        if not parsed:
            return []

        names: list[str] = []
        for key in ("path", "file_path"):
            skill_name = AgentLoop._skill_name_from_workspace_path(parsed.get(key))
            if skill_name and skill_name not in names:
                names.append(skill_name)

        for value in AgentLoop._iter_tool_argument_strings(parsed):
            for skill_name in AgentLoop._skill_names_from_argument_text(value):
                if skill_name and skill_name not in names:
                    names.append(skill_name)

        if name == "skill_marketplace":
            for key in ("skill_name", "name"):
                value = str(parsed.get(key) or "").strip()
                if value and value not in names:
                    names.append(value)
        return names

    def _discover_invoked_skill_contexts_from_runtime(
        self,
        start_index: int,
    ) -> list[dict[str, Any]]:
        """Infer skill usage from actual tool calls/results in the current turn."""
        if not isinstance(start_index, int) or start_index < 0:
            return []

        messages = AgentLoop._get_runtime_memory_messages(self)
        if not messages or start_index >= len(messages):
            return []

        discovered_by_name: dict[str, tuple[int, dict[str, Any]]] = {}
        order = 0

        def _add_name(name: str) -> None:
            nonlocal order
            skill_name = str(name or "").strip()
            if not skill_name:
                return
            context = self._resolve_skill_context_by_name(skill_name)
            if context is None:
                return
            order += 1
            context = {**context, "source": "tool_usage"}
            discovered_by_name[skill_name] = (order, context)

        for msg in messages[start_index:]:
            role = AgentLoop._stream_message_role(msg).lower()
            if role == "assistant":
                tool_calls = AgentLoop._stream_message_attr(msg, "tool_calls", None) or []
                for tool_call in tool_calls:
                    tool_name, arguments = AgentLoop._tool_call_name_and_arguments(tool_call)
                    for skill_name in AgentLoop._extract_skill_names_from_tool_call(
                        tool_name,
                        arguments,
                    ):
                        _add_name(skill_name)
                continue

        return [
            context
            for _order, context in sorted(
                discovered_by_name.values(),
                key=lambda item: item[0],
                reverse=True,
            )
        ]

    def _merge_latest_user_invoked_skills(self, skill_contexts: list[dict[str, Any]]) -> int:
        """Merge discovered skill metadata into the latest persisted user turn."""
        if not skill_contexts:
            return 0

        session = getattr(self, "_session", None)
        messages = getattr(session, "messages", None)
        if not isinstance(messages, list):
            return 0

        target: dict[str, Any] | None = None
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role == "user":
                state = AgentLoop._turn_state_of_message(message)
                if state not in {_TURN_STATE_INTERRUPTED, _TURN_STATE_SUPERSEDED}:
                    target = message
                break

        if target is None:
            return 0

        existing = AgentLoop._iter_message_invoked_skill_refs(target)
        merged: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for skill in [*existing, *skill_contexts]:
            skill_meta = AgentLoop._session_skill_metadata(skill)
            if not skill_meta:
                continue
            name = skill_meta["name"]
            if name in seen_names:
                continue
            seen_names.add(name)
            merged.append(skill_meta)

        if not merged:
            return 0

        if target.get("invoked_skills") == merged:
            return 0

        target["invoked_skills"] = merged
        return len(merged)

    def _merge_turn_invoked_skills_from_runtime(self, start_index: int) -> int:
        """Persist current-turn skill usage discovered from actual tool execution."""
        try:
            contexts = self._discover_invoked_skill_contexts_from_runtime(start_index)
            return self._merge_latest_user_invoked_skills(contexts)
        except Exception as exc:
            logger.debug(f"Runtime skill usage merge skipped: {exc}")
            return 0

    @staticmethod
    def _should_persist_tool_trace() -> bool:
        """Whether to persist tool traces alongside user/assistant turns."""
        import os as _os

        raw = _os.getenv("SPOON_BOT_PERSIST_TOOL_TRACE")
        if raw is None:
            return True
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _attachment_looks_like_skill_zip(item: dict[str, Any]) -> bool:
        name = str(item.get("name") or item.get("file_name") or "").strip().lower()
        mime_type = str(item.get("mime_type") or item.get("file_type") or "").strip().lower()
        uri = str(item.get("workspace_path") or item.get("uri") or "").strip().lower()
        return (
            name.endswith(".zip")
            or uri.endswith(".zip")
            or mime_type in {"application/zip", "application/x-zip-compressed"}
        )

    async def _install_skill_zip_attachments(
        self,
        attachments: list[dict[str, Any]] | None,
    ) -> list[InstalledSkillZip]:
        """Install attached skill zip archives before the LLM plans commands."""
        self._current_turn_skill_zip_installs = []
        self._current_turn_skill_zip_failures = []
        if not attachments:
            return []

        installed: list[InstalledSkillZip] = []
        for item in attachments:
            if not isinstance(item, dict) or not self._attachment_looks_like_skill_zip(item):
                continue
            uri = str(item.get("workspace_path") or item.get("uri") or "").strip()
            path = _resolve_workspace_file(uri, self.workspace)
            if path is None:
                failure = f"{uri or '<missing>'}: attachment is not available in workspace"
                self._current_turn_skill_zip_failures.append(failure)
                logger.warning(f"Skill zip attachment skipped: {failure}")
                continue
            try:
                result = install_skill_zip_archive(
                    path,
                    self.workspace,
                    name_hint=str(item.get("name") or path.name),
                    reinstall=True,
                )
            except Exception as exc:
                failure = f"{path.name}: {exc}"
                self._current_turn_skill_zip_failures.append(failure)
                logger.warning(f"Skill zip attachment install failed: {failure}")
                continue
            if result is None:
                continue
            installed.append(result)
            try:
                self.record_touched_paths(result.skill_md)
            except Exception:
                pass
            AgentLoop._queue_runtime_notice(
                self,
                kind="skill_zip_install",
                stage="prepare",
                name=result.name,
                path=self._workspace_relative_display_path(result.skill_md),
                reinstalled=result.reinstalled,
                file_count=result.file_count,
            )

        if installed:
            try:
                await self.reload_skills()
            except Exception as exc:
                failure = f"reload_skills failed after zip install: {exc}"
                self._current_turn_skill_zip_failures.append(failure)
                logger.warning(failure)

        self._current_turn_skill_zip_installs = installed
        return installed

    def _workspace_relative_display_path(self, path: Path) -> str:
        try:
            rel = path.resolve().relative_to(self.workspace.resolve())
            return rel.as_posix()
        except Exception:
            return Path(path).as_posix()

    def _current_turn_skill_zip_context(self) -> str:
        lines: list[str] = []
        for item in getattr(self, "_current_turn_skill_zip_installs", []):
            action = "reinstalled" if item.reinstalled else "installed"
            lines.append(
                f"- {action} `{item.name}` from `{item.source.name}` at "
                f"`{self._workspace_relative_display_path(item.skill_md)}`"
            )
        for failure in getattr(self, "_current_turn_skill_zip_failures", []):
            lines.append(f"- failed: {failure}")
        if not lines:
            return ""
        return "[ATTACHED SKILL ZIP STATUS]\n" + "\n".join(lines)

    def _add_current_turn_skill_zip_context(self, text: str) -> str:
        context = self._current_turn_skill_zip_context()
        if not context:
            return text
        return f"{context}\n{text}"

    def _build_runtime_message_content(
        self,
        role: str,
        content: str,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Build message content in the format expected by spoon-core runtime memory."""
        text_content = _ensure_attachment_context(content, attachments or [])
        if role == "user" and media:
            return self.context._build_user_content(text_content, media)
        return text_content

    def _build_current_turn_runtime_user_text(self, message: str) -> str:
        """Return the real user turn stored in runtime memory.

        Request-scoped facts are injected through the temporary system prompt.
        Keeping this message raw lets provider history retain an actual user
        turn instead of replacing it with another synthetic wrapper.
        """
        return str(message or "")

    @classmethod
    def _assistant_session_save_kwargs(cls, content: str) -> dict[str, Any]:
        """Persist assistant replies without prompt-derived dispatch metadata."""
        return {"message_kind": "assistant_reply"}

    @staticmethod
    def _session_skill_metadata(skill: dict[str, Any]) -> dict[str, Any] | None:
        """Return sanitized skill metadata safe to persist in session history."""
        if not isinstance(skill, dict) or not skill.get("name"):
            return None
        return {
            "name": str(skill.get("name")),
            "location": str(skill.get("location") or ""),
            "workspace_relative_path": str(skill.get("workspace_relative_path") or ""),
            "organized": bool(skill.get("organized", True)),
            "source": str(skill.get("source") or "skill_match"),
        }

    @staticmethod
    def _session_message_save_kwargs(
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        invoked_skill: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build optional persisted-session metadata for a user turn."""
        save_kwargs: dict[str, Any] = {}
        if media:
            save_kwargs["media"] = list(media)
        if attachments:
            save_kwargs["attachments"] = [
                dict(item) for item in attachments if isinstance(item, dict)
            ]
        skill_meta = AgentLoop._session_skill_metadata(invoked_skill or {})
        if skill_meta:
            save_kwargs["invoked_skills"] = [dict(skill_meta)]
        return save_kwargs

    @staticmethod
    def _turn_state_of_message(message: Any) -> str:
        """Normalize persisted turn state metadata for user messages."""
        if not isinstance(message, dict):
            return ""
        return str(message.get("turn_state") or "").strip().lower()

    @staticmethod
    def _update_session_user_turn_state(
        session: Any,
        state: str,
        *,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        """Update the latest unresolved user turn in persisted session history."""
        messages = getattr(session, "messages", None)
        if not isinstance(messages, list):
            return None

        terminal_states = {_TURN_STATE_COMPLETED, _TURN_STATE_INTERRUPTED, _TURN_STATE_SUPERSEDED}
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").strip().lower() != "user":
                continue

            current_state = AgentLoop._turn_state_of_message(message)
            if current_state in terminal_states:
                continue

            message["turn_state"] = state
            message["turn_state_updated_at"] = datetime.now().isoformat()
            if reason:
                message["turn_state_reason"] = reason
            elif "turn_state_reason" in message:
                message.pop("turn_state_reason", None)
            return message
        return None

    @staticmethod
    def _persist_session_if_possible(self) -> None:
        """Persist the current session when the manager is available."""
        try:
            self.sessions.save(self._session)
        except Exception as exc:
            logger.warning(f"Failed to save session: {exc}")

    @staticmethod
    def _mark_latest_user_turn_state(
        self,
        state: str,
        *,
        reason: str | None = None,
    ) -> bool:
        """Mark the latest unresolved user turn as completed/interrupted/superseded."""
        message = AgentLoop._update_session_user_turn_state(
            getattr(self, "_session", None),
            state,
            reason=reason,
        )
        if message is None:
            return False
        AgentLoop._persist_session_if_possible(self)
        return True

    def _refresh_recent_turn_notice(self) -> None:
        """Expose immediate prior interruption/supersede state to the next prompt."""
        notice: str | None = None
        session = getattr(self, "_session", None)
        history_messages = (
            session.get_messages()
            if session is not None and hasattr(session, "get_messages")
            else getattr(session, "messages", [])
        )
        if isinstance(history_messages, list):
            for message in reversed(history_messages):
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "").strip().lower()
                if role != "user":
                    break

                state = AgentLoop._turn_state_of_message(message)
                if state == _TURN_STATE_INTERRUPTED:
                    interrupted_request = self._truncate_request_for_prompt(
                        str(message.get("content") or "").strip()
                    )
                    notice = (
                        "The immediately previous user request was interrupted before completion.\n"
                        f"[INTERRUPTED PREVIOUS REQUEST]: {interrupted_request}\n"
                        "Resolve it against the newest user message as follows:\n"
                        "- If the newest user message is itself a standalone actionable request, "
                        "treat it as replacing the interrupted request.\n"
                        "- If the newest user message only adds constraints or details to the "
                        "interrupted request, continue the interrupted request with the new "
                        "constraints applied.\n"
                        "- Do not execute both as separate tasks unless the newest user message "
                        "explicitly asks for both."
                    )
                elif state == _TURN_STATE_SUPERSEDED:
                    notice = (
                        "A previous unfinished user request was superseded by a newer request "
                        "and is no longer pending. Execute only the newest user request unless "
                        "it explicitly asks to resume the earlier one."
                    )
                break

        self._recent_turn_notice = notice

    def _refresh_recent_invoked_skill_contexts(self) -> None:
        """Expose recent completed skill-backed turns as bounded continuity context."""
        self._recent_invoked_skill_contexts = self._find_recent_invoked_skill_contexts()

    def _recent_completed_assistant_contexts(
        self,
        *,
        max_turns: int = 6,
    ) -> list[dict[str, Any]]:
        """Return bounded prior assistant results without replaying old user tasks."""
        session = getattr(self, "_session", None)
        history_messages = (
            session.get_messages()
            if session is not None and hasattr(session, "get_messages")
            else getattr(session, "messages", [])
        )
        if not isinstance(history_messages, list):
            return []

        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for message in history_messages:
            if not isinstance(message, dict):
                continue

            role = str(message.get("role") or "").strip().lower()
            if role == "user":
                state = AgentLoop._turn_state_of_message(message)
                if state in {
                    "",
                    _TURN_STATE_PENDING,
                    _TURN_STATE_INTERRUPTED,
                    _TURN_STATE_SUPERSEDED,
                }:
                    current = None
                    continue

                skills = [
                    meta
                    for meta in (
                        AgentLoop._session_skill_metadata(item)
                        for item in AgentLoop._iter_message_invoked_skill_refs(message)
                    )
                    if meta
                ]
                current = {"assistant": "", "skills": skills}
                turns.append(current)
                continue

            if current is None or role != "assistant":
                continue
            message_kind = str(message.get("message_kind") or "")
            if message_kind and message_kind != "assistant_reply":
                continue
            if message.get("tool_calls"):
                continue
            if current.get("assistant"):
                continue

            content = message.get("content", "")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)
            if content.strip():
                current["assistant"] = content

        useful_turns = [
            turn for turn in turns if str(turn.get("assistant") or "").strip() or turn.get("skills")
        ]
        return list(reversed(useful_turns[-max_turns:]))

    @staticmethod
    def _iter_message_invoked_skill_refs(message: dict[str, Any]) -> list[dict[str, Any]]:
        """Return invoked skill references persisted on a session message."""
        invoked = message.get("invoked_skills")
        if isinstance(invoked, list):
            return [item for item in invoked if isinstance(item, dict)]
        return []

    def _find_recent_invoked_skill_contexts(
        self,
        *,
        max_user_turns: int = 12,
        max_skills: int = 4,
    ) -> list[dict[str, Any]]:
        """Return recent skill metadata without rehydrating old task history.

        The session transcript can be long and noisy.  For follow-up requests we
        keep a small newest-first set of skill identities, then resolve each
        against the current skill catalog so imported or stale session data
        cannot inject an arbitrary path/prompt.
        """
        session = getattr(self, "_session", None)
        history_messages = (
            session.get_messages()
            if session is not None and hasattr(session, "get_messages")
            else getattr(session, "messages", [])
        )
        if not isinstance(history_messages, list):
            return []

        contexts: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        user_turns_seen = 0
        for message in reversed(history_messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "").strip().lower()
            if role != "user":
                continue

            user_turns_seen += 1
            if user_turns_seen > max_user_turns:
                break

            state = AgentLoop._turn_state_of_message(message)
            if state in {_TURN_STATE_INTERRUPTED, _TURN_STATE_SUPERSEDED}:
                continue

            for skill in AgentLoop._iter_message_invoked_skill_refs(message):
                skill_name = str(skill.get("name") or "").strip()
                if not skill_name or skill_name in seen_names:
                    continue

                context = self._resolve_skill_context_by_name(skill_name)
                if context is None:
                    continue

                timestamp = str(message.get("timestamp") or "").strip()
                if timestamp:
                    context = {**context, "last_used_at": timestamp}
                contexts.append(context)
                seen_names.add(skill_name)
                if len(contexts) >= max_skills:
                    return contexts

        if len(contexts) < max_skills:
            seen_names = {
                str(context.get("name") or "").strip()
                for context in contexts
                if str(context.get("name") or "").strip()
            }
            fallback = self._find_recent_skill_contexts_from_tool_traces(
                history_messages,
                seen_names=seen_names,
                max_messages=max_user_turns * 24,
                max_skills=max_skills - len(contexts),
            )
            contexts.extend(fallback)

        return contexts

    def _find_recent_skill_contexts_from_tool_traces(
        self,
        history_messages: list[Any],
        *,
        seen_names: set[str],
        max_messages: int,
        max_skills: int,
    ) -> list[dict[str, Any]]:
        """Recover recent skill usage from persisted structured tool calls.

        Older sessions may not have ``invoked_skills`` on user messages.  The
        assistant tool-call arguments still contain real workspace paths such as
        ``skills/<name>/SKILL.md`` or ``skills/<name>/cli/...``; use those as
        generic evidence instead of prompt-specific routing.
        """
        if max_skills <= 0 or not isinstance(history_messages, list):
            return []

        contexts: list[dict[str, Any]] = []
        inspected = 0
        for message in reversed(history_messages):
            if inspected >= max_messages or len(contexts) >= max_skills:
                break
            if not isinstance(message, dict):
                continue
            inspected += 1
            role = str(message.get("role") or "").strip().lower()
            if role != "assistant":
                continue

            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                tool_name, arguments = AgentLoop._tool_call_name_and_arguments(tool_call)
                for skill_name in AgentLoop._extract_skill_names_from_tool_call(
                    tool_name,
                    arguments,
                ):
                    skill_name = str(skill_name or "").strip()
                    if not skill_name or skill_name in seen_names:
                        continue
                    context = self._resolve_skill_context_by_name(skill_name)
                    if context is None:
                        continue
                    context = {**context, "source": "tool_trace"}
                    contexts.append(context)
                    seen_names.add(skill_name)
                    if len(contexts) >= max_skills:
                        break
                if len(contexts) >= max_skills:
                    break

        return contexts

    def _persist_user_turn_to_session(
        self,
        message: str,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        invoked_skill: dict[str, Any] | None = None,
    ) -> None:
        """Persist the current user turn before the model run starts."""
        try:
            self._session.add_message(
                "user",
                _strip_attachment_context(message, attachments or []),
                turn_id=uuid.uuid4().hex,
                turn_state=_TURN_STATE_PENDING,
                turn_state_updated_at=datetime.now().isoformat(),
                **AgentLoop._session_message_save_kwargs(
                    media,
                    attachments,
                    invoked_skill=invoked_skill,
                ),
            )
            AgentLoop._persist_session_if_possible(self)
        except Exception as exc:
            logger.warning(f"Failed to persist user turn: {exc}")

    def set_subagent_context(
        self,
        *,
        session_key: str | None = None,
        channel: str | None = None,
        metadata: dict[str, Any] | None = None,
        reply_to: str | None = None,
    ) -> None:
        """Bind the spawn tool to the current requester session/channel."""
        resolved_session_key = session_key or getattr(self, "session_key", None)
        manager = getattr(self, "_subagent_manager", None)
        if manager is not None:
            manager.set_spawner_context(
                session_key=resolved_session_key,
                channel=channel,
                metadata=metadata,
                reply_to=reply_to,
            )

        registry = getattr(self, "tools", None)
        if registry is None:
            return
        spawn_tool = registry.get("spawn")
        if spawn_tool and isinstance(spawn_tool, SubagentTool):
            spawn_tool.set_spawner_context(
                session_key=resolved_session_key,
                channel=channel,
                metadata=metadata,
                reply_to=reply_to,
            )

    def _persist_turn(
        self,
        user_message: str,
        assistant_message: str,
        *,
        media: list[str] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> None:
        """Save a completed user/assistant turn to session storage."""
        try:
            save_kwargs: dict[str, Any] = {}
            if media:
                save_kwargs["media"] = list(media)
            if attachments:
                save_kwargs["attachments"] = [
                    dict(item) for item in attachments if isinstance(item, dict)
                ]
            self._session.add_message(
                "user",
                _strip_attachment_context(user_message, attachments or []),
                **save_kwargs,
            )
            self._session.add_message("assistant", assistant_message)
            self.sessions.save(self._session)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    def _persist_incomplete_turn_marker(
        self,
        *,
        label: str,
        reason: BaseException | str,
    ) -> None:
        """Close a persisted user turn when execution fails before an answer."""
        try:
            if not getattr(self, "_session", None):
                return
            reason_text = (
                type(reason).__name__
                if isinstance(reason, BaseException)
                else str(reason or "Incomplete")
            )
            messages = getattr(self._session, "messages", None)
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if (
                    isinstance(last, dict)
                    and last.get("role") == "assistant"
                    and last.get("incomplete") is True
                ):
                    return
            marker = (
                "[Previous request did not complete: "
                f"{reason_text}. Treat it as historical context only; do not "
                "continue or answer it unless a later user request explicitly "
                "asks to resume or refer to it.]"
            )
            self._session.add_message(
                "assistant",
                marker,
                incomplete=True,
                incomplete_reason=reason_text,
                incomplete_source=label,
            )
            self.sessions.save(self._session)
        except Exception as exc:
            logger.debug(f"Failed to persist incomplete turn marker: {exc}")

    @staticmethod
    def _turn_failure_state_reason(label: str, reason: BaseException | str) -> str:
        """Return a bounded turn_state_reason for failed provider/runtime turns."""
        safe_label = re.sub(r"[^0-9A-Za-z_-]+", "_", str(label or "runtime")).strip("_")
        if not safe_label:
            safe_label = "runtime"
        reason_name = type(reason).__name__ if isinstance(reason, BaseException) else "error"
        return f"{safe_label}_error:{reason_name}"

    def _persist_failed_turn_context(
        self,
        *,
        label: str,
        reason: BaseException | str,
        start_index: int | None = None,
    ) -> None:
        """Close and persist an errored turn so the next request keeps context.

        User turns are saved before model execution starts. If the model/provider
        fails before a final assistant answer, leaving the turn as ``pending``
        makes the next request look contextless and drops any tool evidence that
        only existed in runtime memory. Persist the runtime tool trace first,
        then close the user turn as interrupted and add a small marker.
        """
        if not getattr(self, "_session", None):
            return

        try:
            if isinstance(start_index, int) and start_index >= 0:
                try:
                    self._merge_turn_invoked_skills_from_runtime(start_index)
                except Exception as exc:
                    logger.debug(f"Failed-turn skill merge skipped: {exc}")
                try:
                    persisted = self._persist_turn_tool_trace(start_index)
                    if persisted:
                        AgentLoop._persist_session_if_possible(self)
                except Exception as exc:
                    logger.debug(f"Failed-turn tool trace persist skipped: {exc}")

            AgentLoop._mark_latest_user_turn_state(
                self,
                _TURN_STATE_INTERRUPTED,
                reason=AgentLoop._turn_failure_state_reason(label, reason),
            )
            AgentLoop._persist_incomplete_turn_marker(
                self,
                label=label,
                reason=reason,
            )
        except Exception as exc:
            logger.debug(f"Failed-turn persistence skipped: {exc}")

    def _persist_cancelled_turn_context(self, *, start_index: int | None = None) -> None:
        """Persist available tool trace before marking a cancelled turn interrupted."""
        if not getattr(self, "_session", None):
            return

        try:
            if isinstance(start_index, int) and start_index >= 0:
                try:
                    self._merge_turn_invoked_skills_from_runtime(start_index)
                except Exception as exc:
                    logger.debug(f"Cancelled-turn skill merge skipped: {exc}")
                try:
                    persisted = self._persist_turn_tool_trace(start_index)
                    if persisted:
                        AgentLoop._persist_session_if_possible(self)
                except Exception as exc:
                    logger.debug(f"Cancelled-turn tool trace persist skipped: {exc}")
            AgentLoop._mark_latest_user_turn_state(
                self,
                _TURN_STATE_INTERRUPTED,
                reason="task_cancelled",
            )
        except Exception as exc:
            logger.debug(f"Cancelled-turn persistence skipped: {exc}")

    @staticmethod
    def _persist_interrupted_assistant_reply(
        self,
        content: str,
        *,
        reason: str = "task_cancelled",
    ) -> bool:
        """Persist user-visible assistant text from an interrupted stream."""
        if not getattr(self, "_session", None):
            return False

        content = str(content or "")
        if not content.strip():
            return False

        try:
            messages = getattr(self._session, "messages", None)
            if isinstance(messages, list) and messages:
                last = messages[-1]
                if (
                    isinstance(last, dict)
                    and last.get("role") == "assistant"
                    and last.get("incomplete") is True
                    and str(last.get("content") or "") == content
                ):
                    return False

            self._session.add_message(
                "assistant",
                content,
                incomplete=True,
                incomplete_reason=reason,
                incomplete_source="stream",
                **AgentLoop._assistant_session_save_kwargs(content),
            )
            AgentLoop._persist_session_if_possible(self)
            return True
        except Exception as exc:
            logger.debug(f"Interrupted assistant reply persistence skipped: {exc}")
            return False

    def _current_tool_owner_key(self, session_key: str | None = None) -> str:
        """Resolve a user-scoped ownership key for background tool jobs."""
        return build_tool_owner_key(
            getattr(self, "user_id", None),
            session_key if session_key is not None else getattr(self, "session_key", None),
        )

    async def _run_agent_with_retry(
        self,
        label: str = "agent",
        pre_retry_cleanup: Callable[[], Any] | None = None,
        **run_kwargs: Any,
    ) -> Any:
        """Run ``self._agent.run()`` wrapped with provider-level retry.

        Centralises the retry pattern used by both ``process()`` and
        ``process_with_thinking()`` so the logic isn't duplicated.

        Args:
            label: Descriptive label used in log messages (e.g. "stream").
            pre_retry_cleanup: Optional sync callable invoked before each retry
                (e.g. to drain the output queue for streaming).
            **run_kwargs: Forwarded to ``self._agent.run()``.
        """
        retry_config = getattr(self, "_retry_config", None)
        if not isinstance(retry_config, RetryConfig):
            retry_config = DEFAULT_RETRY_CONFIG

        async def _do_run() -> Any:
            with (
                bind_tool_owner(self._current_tool_owner_key()),
                bind_tool_workspace(str(getattr(self, "workspace", "") or "")),
            ):
                request = run_kwargs.get("request")
                if not (isinstance(request, str) and request.strip()):
                    await AgentLoop._ensure_runtime_user_tail_before_run(self, label)
                return await self._agent.run(**run_kwargs)

        def _on_retry(attempt: int, exc: Exception, delay: float) -> None:
            logger.warning(
                f"[{label}] Provider transient error (attempt {attempt + 1}/"
                f"{retry_config.max_retries + 1}), "
                f"retrying in {delay:.1f}s: {type(exc).__name__}: {exc}"
            )
            if pre_retry_cleanup:
                try:
                    pre_retry_cleanup()
                except Exception:
                    pass

        return await with_provider_retry(
            _do_run,
            config=retry_config,
            on_retry=_on_retry,
        )

    @staticmethod
    def _resolve_retry_runner(self) -> Callable[..., Awaitable[Any]]:
        """Pick the configured retry runner, falling back for MagicMock-based tests."""
        retry_runner = getattr(self, "_run_agent_with_retry", None)
        self_type_module = getattr(type(self), "__module__", "")
        if not self_type_module.startswith("unittest.mock") and isinstance(self, AgentLoop):
            return retry_runner

        if callable(retry_runner):
            runner_module = getattr(type(retry_runner), "__module__", "")
            if not runner_module.startswith("unittest.mock"):
                return retry_runner
            if getattr(retry_runner, "side_effect", None) is not None:
                return retry_runner
            if getattr(retry_runner, "_mock_wraps", None) is not None:
                return retry_runner

        async def _fallback(**kwargs: Any) -> Any:
            return await AgentLoop._run_agent_with_retry(self, **kwargs)

        return _fallback

    def _reset_agent_state_for_retry(self) -> None:
        """Reset transient runtime state before retrying the same turn."""
        if hasattr(self._agent, "state") and self._agent.state != AgentState.IDLE:
            self._agent.state = AgentState.IDLE
            self._agent.current_step = 0
        if hasattr(self._agent, "_shutdown_event") and self._agent._shutdown_event.is_set():
            self._agent._shutdown_event.clear()

    def _prepare_agent_for_new_turn(self) -> None:
        """Clear transient execution state so the newest prompt owns the next run."""
        if hasattr(self._agent, "state") and self._agent.state != AgentState.IDLE:
            logger.warning(
                f"Agent {getattr(self._agent, 'name', 'runtime')} was in "
                f"{self._agent.state} state before a new turn; resetting to IDLE"
            )
            self._agent.state = AgentState.IDLE

        if hasattr(self._agent, "current_step"):
            try:
                self._agent.current_step = 0
            except Exception:
                pass

        if hasattr(self._agent, "_shutdown_event") and self._agent._shutdown_event.is_set():
            logger.info("Clearing previous shutdown signal before processing a new turn")
            self._agent._shutdown_event.clear()

        if hasattr(self._agent, "tool_calls"):
            try:
                tool_calls = getattr(self._agent, "tool_calls")
                if hasattr(tool_calls, "clear"):
                    tool_calls.clear()
                elif tool_calls:
                    self._agent.tool_calls = []
            except Exception:
                pass

    def _compress_runtime_context_for_overflow_retry(self) -> int:
        """Compress runtime context only after an explicit overflow signal."""
        compressed = 0
        try:
            compressed = self._compress_runtime_context(
                force=True,
                budget_tokens=self._runtime_compaction_trigger_budget(),
            )
        except Exception as exc:
            logger.debug(f"Soft runtime compaction failed during overflow recovery: {exc}")
        if compressed == 0:
            compressed = self._force_compress_runtime_context()
        return compressed

    async def _run_agent_with_context_overflow_recovery(
        self,
        *,
        label: str,
        retry_runner: Callable[..., Awaitable[Any]],
        pre_overflow_retry_cleanup: Callable[[], Any] | None = None,
        **run_kwargs: Any,
    ) -> Any:
        """Retry once after deliberate runtime compaction on context overflow."""
        try:
            result = await retry_runner(label=label, **run_kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            if not is_context_overflow_error(exc):
                raise

            logger.warning(
                f"[{label}] Context overflow detected: {type(exc).__name__}: {exc}. "
                "Compacting older runtime history and retrying once."
            )
            if pre_overflow_retry_cleanup:
                try:
                    pre_overflow_retry_cleanup()
                except Exception:
                    pass

            compressed = self._compress_runtime_context_for_overflow_retry()
            if compressed <= 0:
                raise

            AgentLoop._queue_runtime_notice(
                self,
                kind="runtime_compaction",
                stage="overflow_retry",
                compressed_actions=compressed,
                estimated_tokens_before=getattr(exc, "estimated_tokens", None),
                trigger_budget=getattr(exc, "max_tokens", None),
                error_type=type(exc).__name__,
            )

            self._reset_agent_state_for_retry()
            result = await retry_runner(label=f"{label}:overflow_retry", **run_kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
