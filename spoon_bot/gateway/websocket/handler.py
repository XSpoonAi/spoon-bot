"""WebSocket endpoint and message handler.

This module implements the WebSocket endpoint for real-time agent communication,
including streaming responses, user confirmation flow, and event notifications.

Based on: docs/plans/2025-02-05-spoon-bot-api-design.md
"""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable, Coroutine

from fastapi import WebSocket, WebSocketDisconnect, Query
from loguru import logger

from spoon_bot.gateway.app import (
    cancel_ws_session_chat_task,
    clear_ws_session_chat_task,
    get_agent,
    get_connection_manager,
    get_config,
    get_agent_execution_lock,
    get_ws_session_chat_lock,
    get_ws_session_chat_task_id,
    get_session_execution_lock,
    has_active_ws_session_chat_task,
    is_auth_required,
    register_ws_session_chat_task,
)
from spoon_bot.gateway.auth.jwt import verify_token
from spoon_bot.gateway.auth.api_key import verify_api_key
from spoon_bot.gateway.errors import TimeoutCode, build_timeout_error_detail
from spoon_bot.gateway.observability.budget import BudgetExhaustedError, check_budget
from spoon_bot.gateway.observability.tracing import TimerSpan, build_timing_payload, new_trace_id
from spoon_bot.gateway.websocket.protocol import (
    ClientMethod,
    ServerEvent,
    WSRequest,
    WSResponse,
    WSError,
    WSEvent,
    WSStreamChunk,
    parse_message,
)
from spoon_bot.gateway.websocket.workspace_fs import SANDBOX_WORKSPACE_ROOT, WorkspaceFSService
from spoon_bot.gateway.websocket.workspace_terminal import WorkspaceTerminalService
from spoon_bot.gateway.websocket.workspace_watch import WorkspaceWatchService


# ---------------------------------------------------------------------------
# Methods that are safe for concurrent (task-based) dispatch.
# Path-sensitive fs/workspace ops rely on WorkspaceFSService locking to
# serialize conflicting requests while allowing unrelated paths to proceed.
# ---------------------------------------------------------------------------

_CONCURRENT_METHODS: frozenset[str] = frozenset({
    ClientMethod.FS_LIST.value,
    ClientMethod.FS_STAT.value,
    ClientMethod.FS_READ.value,
    ClientMethod.FS_WRITE.value,
    ClientMethod.FS_MKDIR.value,
    ClientMethod.FS_RENAME.value,
    ClientMethod.FS_REMOVE.value,
    ClientMethod.FS_WATCH.value,
    ClientMethod.FS_UNWATCH.value,
    "workspace.tree",
})
_CONCURRENT_REQUEST_LIMIT = 16
_ATTACHMENT_CONTEXT_HEADER = "Attached workspace files (source of truth for this request):"


def _workspace_root() -> Path:
    """Return the resolved workspace root used for attachment/media validation."""
    agent = get_agent()
    workspace = Path(getattr(agent, "workspace", Path.home() / ".spoon-bot" / "workspace"))
    return workspace.expanduser().resolve()


def _resolve_workspace_file(path_str: str) -> Path | None:
    """Resolve a file path and ensure it stays within the configured workspace."""
    candidate = str(path_str or "").strip()
    if not candidate:
        return None

    workspace = _workspace_root()
    sandbox_root = SANDBOX_WORKSPACE_ROOT.rstrip("/")
    try:
        if candidate.startswith("/"):
            normalized = Path(candidate).as_posix()
            workspace_root_str = workspace.as_posix().rstrip("/")
            if normalized == sandbox_root or normalized.startswith(sandbox_root + "/"):
                relative = normalized[len(sandbox_root):].lstrip("/")
                resolved = (workspace / relative).resolve(strict=True)
            elif normalized == workspace_root_str or normalized.startswith(workspace_root_str + "/"):
                relative = normalized[len(workspace_root_str):].lstrip("/")
                resolved = (workspace / relative).resolve(strict=True)
            else:
                resolved = Path(candidate).expanduser().resolve(strict=True)
        else:
            resolved = (workspace / candidate).resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None

    if resolved != workspace and workspace not in resolved.parents:
        return None
    if not resolved.is_file():
        return None
    return resolved


def _string_list_from_any(raw: Any) -> list[str]:
    """Best-effort normalize media payloads to a list of non-empty strings."""
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


def _normalize_attachment_refs(raw: Any) -> list[dict[str, Any]]:
    """Normalize attachment metadata from websocket params."""
    if not isinstance(raw, list):
        return []

    attachments: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        uri = str(item.get("workspace_path") or item.get("uri") or "").strip()
        if not uri:
            continue
        attachment = dict(item)
        attachment["uri"] = uri
        attachment.setdefault("workspace_path", uri)
        attachments.append(attachment)
    return attachments


def _validate_media_paths(media: list[str]) -> list[str]:
    """Reject media paths that are missing or outside the sandbox workspace."""
    invalid = [path for path in media if _resolve_workspace_file(path) is None]
    if invalid:
        raise ValueError(f"Invalid media path(s): {', '.join(invalid)}")
    return media


def _validate_attachment_paths(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject attachment paths that are missing or outside the sandbox workspace."""
    missing = [
        str(item.get("workspace_path") or item.get("uri") or "").strip()
        for item in attachments
        if _resolve_workspace_file(str(item.get("workspace_path") or item.get("uri") or "").strip()) is None
    ]
    if missing:
        raise ValueError(f"Invalid attachment path(s): {', '.join(missing)}")
    return attachments


def _derive_media_from_attachments(attachments: list[dict[str, Any]]) -> list[str]:
    """Best-effort derive image media paths from attachment metadata."""
    items: list[str] = []
    for attachment in attachments:
        path = str(attachment.get("workspace_path") or attachment.get("uri") or "").strip()
        mime_type = str(attachment.get("mime_type") or attachment.get("file_type") or "").strip().lower()
        if not path:
            continue
        if mime_type.startswith("image/") or Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
            items.append(path)
    return items


def _merge_attachment_context(message: str, attachments: list[dict[str, Any]]) -> str:
    """Ensure request text contains workspace attachment references."""
    if not attachments:
        return message

    normalized: list[tuple[dict[str, Any], str]] = []
    for item in attachments:
        path = str(item.get("workspace_path") or item.get("uri") or "").strip()
        if path:
            normalized.append((item, path))
    if not normalized:
        return message

    text = message.strip()
    if _ATTACHMENT_CONTEXT_HEADER in text and all(path in text for _, path in normalized):
        return text

    lines = [text] if text else [
        "The user attached files without extra text. Inspect the files and answer based on their contents."
    ]
    lines.extend(["", _ATTACHMENT_CONTEXT_HEADER])
    for item, path in normalized:
        parts: list[str] = []
        name = str(item.get("name") or item.get("file_name") or "").strip()
        mime_type = str(item.get("mime_type") or item.get("file_type") or "").strip()
        size = item.get("size") if "size" in item else item.get("file_size")
        if name:
            parts.append(f"name: {name}")
        if mime_type:
            parts.append(f"mime: {mime_type}")
        if isinstance(size, int) and size > 0:
            parts.append(f"size: {size} bytes")
        line = f"- {path}"
        if parts:
            line += f" ({', '.join(parts)})"
        lines.append(line)
    lines.append("Use these attached workspace files as the primary source of truth for this request.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter for WS auth attempts (#19)
# ---------------------------------------------------------------------------

class _AuthRateLimiter:
    """Track failed auth attempts per IP and enforce cooldown."""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 60):
        self._max = max_attempts
        self._window = window_seconds
        # ip -> list of timestamps of failed attempts
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def record_failure(self, ip: str) -> None:
        now = time.monotonic()
        self._attempts[ip].append(now)
        # Prune old entries
        cutoff = now - self._window
        self._attempts[ip] = [t for t in self._attempts[ip] if t > cutoff]

    def is_blocked(self, ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        attempts = [t for t in self._attempts.get(ip, []) if t > cutoff]
        self._attempts[ip] = attempts
        return len(attempts) >= self._max

    def clear(self, ip: str) -> None:
        self._attempts.pop(ip, None)


_auth_limiter = _AuthRateLimiter()


def _runtime_sandbox_id() -> str:
    for key in ("CYPHER_SANDBOX_ID", "SANDBOX_ID", "MODAL_SANDBOX_ID"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return "runtime"


# ---------------------------------------------------------------------------
# Helper: switch agent session (mirrors agent.py _switch_session)
# ---------------------------------------------------------------------------

def _switch_agent_session(agent, session_key: str | None) -> None:
    """Switch the agent's active session if supported (#11)."""
    if not session_key or session_key == "default":
        return
    if hasattr(agent, "sessions") and hasattr(agent, "_session"):
        try:
            agent._session = agent.sessions.get_or_create(session_key)
            agent.session_key = session_key
        except Exception:
            pass  # Gracefully degrade


def _get_agent_response_source(agent) -> dict[str, Any]:
    """Read machine-readable response source metadata from the agent."""
    getter = getattr(agent, "get_last_response_source", None)
    if callable(getter) and not inspect.iscoroutinefunction(getter):
        try:
            raw = getter()
            if inspect.isawaitable(raw):
                return {
                    "type": "agent",
                    "is_subagent": False,
                    "subagent_id": None,
                    "subagent_name": None,
                }
            if isinstance(raw, dict):
                return dict(raw)
        except Exception:
            pass
    return {
        "type": "agent",
        "is_subagent": False,
        "subagent_id": None,
        "subagent_name": None,
    }


async def websocket_endpoint(
    websocket: WebSocket,
    token: str | None = Query(default=None),
    api_key: str | None = Query(default=None),
) -> None:
    """
    WebSocket endpoint for real-time agent communication.

    Authenticate via query parameter:
    - ?token=<jwt_access_token>
    - ?api_key=<api_key>
    """
    config = get_config()
    manager = get_connection_manager()

    # Determine client IP for rate limiting
    client_ip = "unknown"
    if websocket.client:
        client_ip = websocket.client.host

    # Authenticate
    user_id = None
    session_key = "default"

    # If auth is not required, allow anonymous connections
    if not is_auth_required():
        user_id = "anonymous"
    else:
        # Rate-limit check (#19)
        if _auth_limiter.is_blocked(client_ip):
            # Accept then close so the client sees WS close code
            await websocket.accept()
            await websocket.close(code=4029, reason="Too many failed auth attempts. Try again later.")
            return

        if token:
            token_data = verify_token(token, config.jwt.secret_key, config.jwt.algorithm, expected_type="access")
            if token_data:
                user_id = token_data.user_id
                session_key = token_data.session_key

        if not user_id and api_key:
            api_key_data = verify_api_key(api_key, config)
            if api_key_data:
                user_id = api_key_data.user_id

    if not user_id:
        # Accept first so the client can observe the WS close code (#17)
        await websocket.accept()
        _auth_limiter.record_failure(client_ip)
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Successful auth — clear any rate-limit history for this IP
    if is_auth_required():
        _auth_limiter.clear(client_ip)

    # Connect
    conn_id = await manager.connect(websocket, user_id, session_key)
    handler = WebSocketHandler(conn_id, session_key, user_id=user_id)

    try:
        # Send connection success event
        await manager.send_message(
            conn_id,
            WSEvent(
                event="connection.established",
                data={"connection_id": conn_id, "session_key": session_key},
            ),
        )

        # Message loop
        while True:
            try:
                data = await websocket.receive_json()
                # Every inbound frame counts as liveness evidence, even when
                # we don't end up sending a response. Without this, a mostly
                # client-driven conversation could look idle to the ping loop
                # and get evicted during a long streaming turn.
                manager.touch(conn_id)
                message = parse_message(data)

                if message.type.value == "ping":
                    await manager.send_message(
                        conn_id,
                        {"type": "pong", "timestamp": data.get("timestamp")},
                    )
                    continue

                if isinstance(message, WSRequest):
                    # Run chat requests as background tasks so the message
                    # loop stays free to process cancel / status requests.
                    if message.method in ("agent.chat", ClientMethod.CHAT_SEND.value):
                        _req = message  # capture for closure
                        session_key = handler._resolve_effective_session_key(_req.params)
                        registry_user_id = handler._resolve_registry_user_id()
                        session_chat_lock = get_ws_session_chat_lock(
                            session_key,
                            user_id=registry_user_id,
                        )

                        async with session_chat_lock:
                            # New chat for the same session should replace any stale/disconnected run.
                            had_active_task = has_active_ws_session_chat_task(
                                session_key,
                                user_id=registry_user_id,
                            )
                            cancelled = await cancel_ws_session_chat_task(
                                session_key,
                                user_id=registry_user_id,
                            )
                            if had_active_task and not cancelled and has_active_ws_session_chat_task(
                                session_key,
                                user_id=registry_user_id,
                            ):
                                await manager.send_message(
                                    conn_id,
                                    WSError(
                                        id=_req.id,
                                        code="TASK_BUSY",
                                        message=(
                                            "A previous chat task for this session is still shutting down. "
                                            "Send chat.cancel and retry shortly."
                                        ),
                                    ),
                                )
                                continue

                            async def _run_chat(
                                req: WSRequest = _req,
                                bound_session_key: str = session_key,
                                bound_user_id: str = registry_user_id,
                            ) -> None:
                                try:
                                    result = await handler._handle_chat(req.params)
                                    await manager.send_message(
                                        conn_id, WSResponse(id=req.id, result=result),
                                    )
                                except Exception as exc:
                                    logger.error(f"Chat error: {exc}")
                                    await manager.send_message(
                                        conn_id,
                                        WSError(id=req.id, code="HANDLER_ERROR", message=str(exc)),
                                    )
                                finally:
                                    handler._current_task = None
                                    clear_ws_session_chat_task(
                                        bound_session_key,
                                        user_id=bound_user_id,
                                        task=asyncio.current_task(),
                                    )

                            chat_task = asyncio.create_task(_run_chat())
                            handler._current_task = chat_task
                            register_ws_session_chat_task(
                                session_key,
                                chat_task,
                                user_id=registry_user_id,
                                cancel_cb=handler._request_current_task_cancel,
                                task_id_cb=handler._current_task_id_for_cancel,
                            )

                    elif message.method in _CONCURRENT_METHODS:
                        task = asyncio.create_task(
                            handler._dispatch_concurrent_request(manager, conn_id, message),
                        )
                        handler._concurrent_tasks.add(task)
                        task.add_done_callback(handler._concurrent_tasks.discard)

                    else:
                        response = await handler.handle_request(message)
                        await manager.send_message(conn_id, response)

            except ValueError as e:
                await manager.send_message(
                    conn_id,
                    WSError(
                        code="INVALID_MESSAGE",
                        message=f"Invalid message format: {e}",
                    ),
                )

    except WebSocketDisconnect:
        logger.info(f"WebSocket {conn_id} disconnected by client")
    except Exception as e:
        logger.error(f"WebSocket error for {conn_id}: {e}")
    finally:
        await handler._cleanup_resources()
        await manager.disconnect(conn_id)


class WebSocketHandler:
    """
    Handles WebSocket message processing.

    Uses the global AgentLoop from the gateway app for all operations.
    Supports confirmation flow for dangerous tool calls.
    """

    def __init__(
        self,
        connection_id: str,
        session_id: str | None = None,
        *,
        user_id: str | None = None,
    ):
        self.connection_id = connection_id
        self.session_id = session_id or connection_id
        self._default_session_key = session_id or "default"
        self._user_id = user_id.strip() if isinstance(user_id, str) and user_id.strip() else "anonymous"
        self._cancel_requested = False
        self._current_task_id: str | None = None
        self._current_task: asyncio.Task | None = None
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._chat_lock = asyncio.Lock()
        self._concurrent_request_slots = asyncio.Semaphore(_CONCURRENT_REQUEST_LIMIT)
        self._concurrent_tasks: set[asyncio.Task] = set()
        agent = get_agent()
        workspace = Path(getattr(agent, "workspace", Path.home() / ".spoon-bot" / "workspace"))
        yolo_mode = bool(getattr(agent, "yolo_mode", False))
        self._sandbox_id = _runtime_sandbox_id()
        self._workspace_watch_service = WorkspaceWatchService(
            workspace_root=workspace,
            emit_change=self._emit_workspace_change,
        )
        self._workspace_fs_service = WorkspaceFSService(workspace_root=workspace, yolo_mode=yolo_mode)
        self._workspace_terminal_service = WorkspaceTerminalService(
            workspace_root=workspace,
            emit_stdout=self._emit_terminal_stdout,
            emit_closed=self._emit_terminal_closed,
            sandbox_id=self._sandbox_id,
        )

        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            # Chat methods
            "agent.chat": self._handle_chat,
            ClientMethod.CHAT_SEND.value: self._handle_chat,  # "chat.send" -> same handler
            "agent.cancel": self._handle_cancel,
            ClientMethod.CHAT_CANCEL.value: self._handle_cancel,
            # Confirmation
            ClientMethod.CONFIRM_RESPOND.value: self._handle_confirm_respond,
            # Status
            "agent.status": self._handle_status,
            # Session methods
            "session.switch": self._handle_session_switch,
            ClientMethod.SESSION_LIST.value: self._handle_session_list,
            "session.clear": self._handle_session_clear,
            ClientMethod.SESSION_EXPORT.value: self._handle_session_export,
            ClientMethod.SESSION_IMPORT.value: self._handle_session_import,
            ClientMethod.SESSION_SEARCH.value: self._handle_session_search,
            # Subscription
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
            # Workspace
            "workspace.tree": self._handle_workspace_tree,
            ClientMethod.FS_LIST.value: self._handle_fs_list,
            ClientMethod.FS_STAT.value: self._handle_fs_stat,
            ClientMethod.FS_READ.value: self._handle_fs_read,
            ClientMethod.FS_WRITE.value: self._handle_fs_write,
            ClientMethod.FS_MKDIR.value: self._handle_fs_mkdir,
            ClientMethod.FS_RENAME.value: self._handle_fs_rename,
            ClientMethod.FS_REMOVE.value: self._handle_fs_remove,
            ClientMethod.FS_WATCH.value: self._handle_fs_watch,
            ClientMethod.FS_UNWATCH.value: self._handle_fs_unwatch,
            ClientMethod.TERM_OPEN.value: self._handle_term_open,
            ClientMethod.TERM_INPUT.value: self._handle_term_input,
            ClientMethod.TERM_RESIZE.value: self._handle_term_resize,
            ClientMethod.TERM_CLOSE.value: self._handle_term_close,
            # Audio streaming
            "audio.stream.start": self._handle_audio_stream_start,
            "audio.stream.end": self._handle_audio_stream_end,
        }

        self._audio_stream_manager = None  # Lazy-init

    async def handle_request(self, request: WSRequest) -> WSResponse | WSError:
        """Route and handle a request."""
        handler = self._handlers.get(request.method)

        if not handler:
            return WSError(
                id=request.id,
                code="UNKNOWN_METHOD",
                message=f"Unknown method: {request.method}",
            )

        try:
            # Params are already validated as dict in parse_message / WSRequest.from_dict
            result = await handler(request.params)
            return WSResponse(id=request.id, result=result)
        except ValueError as e:
            # Business-logic validation errors → INVALID_PARAMS (#14)
            return WSError(
                id=request.id,
                code="INVALID_PARAMS",
                message=str(e),
            )
        except Exception as e:
            logger.error(f"Error handling {request.method}: {e}")
            return WSError(
                id=request.id,
                code="HANDLER_ERROR",
                message="Internal handler error",  # Don't leak internals (#14)
            )

    async def _dispatch_concurrent_request(
        self,
        manager,
        connection_id: str,
        request: WSRequest,
    ) -> None:
        """Run a concurrent-safe request with a bounded per-connection slot pool."""
        async with self._concurrent_request_slots:
            try:
                response = await self.handle_request(request)
                await manager.send_message(connection_id, response)
            except Exception as exc:
                logger.error(f"Concurrent RPC error ({request.method}): {exc}")
                await manager.send_message(
                    connection_id,
                    WSError(id=request.id, code="HANDLER_ERROR", message=str(exc)),
                )

    # ========== Chat ==========

    def _resolve_effective_session_key(self, params: dict[str, Any]) -> str:
        """Resolve the session key for a request using connection binding fallback."""
        bound_session_key = self._default_session_key
        try:
            manager = get_connection_manager()
            conn = manager.get_connection(self.connection_id)
        except RuntimeError:
            conn = None

        if conn:
            bound_session_key = conn.session_key

        session_key = params.get("session_key", bound_session_key)
        if not isinstance(session_key, str) or not session_key.strip():
            session_key = bound_session_key
        return session_key.strip()

    def _resolve_registry_user_id(self) -> str:
        """Resolve the user identity key for WS task ownership."""
        user_id = self._user_id
        try:
            manager = get_connection_manager()
            conn = manager.get_connection(self.connection_id)
        except RuntimeError:
            conn = None
        conn_user_id = getattr(conn, "user_id", None) if conn else None
        if isinstance(conn_user_id, str) and conn_user_id.strip():
            user_id = conn_user_id.strip()
        return user_id

    def _request_current_task_cancel(self) -> None:
        """Signal cancellation for the current task owned by this handler."""
        self._cancel_requested = True
        task = self._current_task
        if task is not None and not task.done():
            task.cancel()

    def _current_task_id_for_cancel(self) -> str | None:
        return self._current_task_id

    async def _handle_chat(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle chat.send / agent.chat — serialized to prevent state races."""
        async with self._chat_lock:
            return await self._execute_chat(params)

    async def _execute_chat(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute chat request — uses the global AgentLoop."""
        agent = get_agent()
        manager = get_connection_manager()
        config = get_config()
        conn = manager.get_connection(self.connection_id)
        if not conn:
            logger.info(
                f"WS connection {self.connection_id} is no longer live; "
                "continuing chat task without WebSocket delivery"
            )

        attachments = _validate_attachment_paths(_normalize_attachment_refs(params.get("attachments")))
        media = _string_list_from_any(params.get("media"))
        media = _validate_media_paths(list(dict.fromkeys(media + _derive_media_from_attachments(attachments))))
        message = _merge_attachment_context(str(params.get("message", "")), attachments)
        if not message:
            raise ValueError("Message or attachments are required")

        session_key = self._resolve_effective_session_key(params)
        self._default_session_key = session_key

        stream = params.get("stream", False)
        thinking = params.get("thinking", False)
        reasoning_effort = params.get("reasoning_effort")
        task_id = f"task_{uuid4().hex[:8]}"
        trace_id = new_trace_id()
        span = TimerSpan("ws_chat")
        self._current_task_id = task_id
        self._cancel_requested = False
        had_error = False

        session_lock = get_session_execution_lock(session_key)
        agent_lock = get_agent_execution_lock()

        try:
            async with session_lock:
                async with agent_lock:
                    # Switch agent session to the requested session_key (#11)
                    _switch_agent_session(agent, session_key)
                    setattr(agent, "user_id", self._resolve_registry_user_id())

                    # Emit thinking event
                    await manager.send_message(
                        self.connection_id,
                        WSEvent(event="agent.thinking", data={
                            "task_id": task_id,
                            "status": "processing",
                            "trace_id": trace_id,
                        }),
                    )
                    check_budget("request", config.budget.request_timeout_ms, span.elapsed_ms)

                    if stream:
                        # Streaming mode: emit chunks via WSEvent
                        full_content = ""
                        async for chunk_data in agent.stream(
                            message=message,
                            media=media,
                            attachments=attachments,
                            thinking=thinking,
                            reasoning_effort=reasoning_effort,
                        ):
                            check_budget("stream", config.budget.stream_timeout_ms, span.elapsed_ms)
                            if self._cancel_requested:
                                break

                            chunk_type = chunk_data.get("type", "content")
                            # Support both "delta" and "content" keys (#10)
                            delta = chunk_data.get("delta", "") or chunk_data.get("content", "")
                            metadata = chunk_data.get("metadata", {})
                            source = chunk_data.get("source") or _get_agent_response_source(agent)

                            # Detect error chunks and propagate as agent.error event
                            if chunk_type == "error":
                                had_error = True
                                await manager.send_message(
                                    self.connection_id,
                                    WSEvent(event=ServerEvent.AGENT_ERROR.value, data={
                                        "task_id": task_id,
                                        "trace_id": trace_id,
                                        "timing": build_timing_payload(span),
                                        "error": {
                                            "code": metadata.get("error_code", "AGENT_RUNTIME_ERROR"),
                                            "message": delta,
                                        },
                                        "source": source,
                                    }),
                                )
                                continue

                            if chunk_type == "done":
                                done_content = (
                                    metadata.get("content", "")
                                    if isinstance(metadata, dict)
                                    else ""
                                )
                                if not full_content and isinstance(done_content, str) and done_content:
                                    full_content = done_content
                                    await manager.send_message(
                                        self.connection_id,
                                        WSEvent(event=ServerEvent.AGENT_STREAM_CHUNK.value, data={
                                            "task_id": task_id,
                                            "type": "content",
                                            "delta": done_content,
                                            "metadata": {"fallback": "done_metadata_content"},
                                            "trace_id": trace_id,
                                            "source": source,
                                        }),
                                    )

                                # Send stream done event
                                await manager.send_message(
                                    self.connection_id,
                                    WSEvent(event=ServerEvent.AGENT_STREAM_DONE.value, data={
                                        "task_id": task_id,
                                        "content": full_content,
                                        "trace_id": trace_id,
                                        "timing": build_timing_payload(span),
                                        "source": source,
                                    }),
                                )
                            else:
                                if chunk_type == "content" and delta:
                                    full_content += delta

                                # Send stream chunk event (even for non-content types like tool_call)
                                if delta or chunk_type != "content":
                                    await manager.send_message(
                                        self.connection_id,
                                        WSEvent(event=ServerEvent.AGENT_STREAM_CHUNK.value, data={
                                            "task_id": task_id,
                                            "type": chunk_type,
                                            "delta": delta,
                                            "metadata": metadata,
                                            "trace_id": trace_id,
                                            "source": source,
                                        }),
                                    )

                        response = full_content
                    else:
                        # Non-streaming mode
                        if thinking:
                            response, thinking_content = await agent.process_with_thinking(
                                message=message,
                                media=media,
                                attachments=attachments,
                                reasoning_effort=reasoning_effort,
                            )
                        else:
                            response = await agent.process(
                                message=message,
                                media=media,
                                attachments=attachments,
                                reasoning_effort=reasoning_effort,
                            )
                            thinking_content = None
                        check_budget("request", config.budget.request_timeout_ms, span.elapsed_ms)

            response_source = _get_agent_response_source(agent)
            # Emit complete event
            span.stop()
            timing = build_timing_payload(span)
            complete_data: dict[str, Any] = {
                "task_id": task_id,
                "status": "done",
                "response": response if isinstance(response, str) else str(response),
                "trace_id": trace_id,
                "timing": timing,
                "source": response_source,
            }
            if not stream and thinking and thinking_content:
                complete_data["thinking_content"] = thinking_content

            await manager.send_message(
                self.connection_id,
                WSEvent(event="agent.complete", data=complete_data),
            )

            result: dict[str, Any] = {
                "success": not had_error,
                "task_id": task_id,
                "content": response,
                "session_key": session_key,
                "trace_id": trace_id,
                "timing": timing,
                "source": response_source,
            }
            if not stream and thinking and thinking_content:
                result["thinking_content"] = thinking_content

            return result
        except BudgetExhaustedError as exc:
            span.stop()
            await manager.send_message(
                self.connection_id,
                WSEvent(
                    event=ServerEvent.AGENT_ERROR.value,
                    data={
                        "task_id": task_id,
                        "trace_id": trace_id,
                        "timing": build_timing_payload(span),
                        "error": {
                            "code": TimeoutCode.TIMEOUT_TOTAL.value,
                            "message": str(exc),
                            "budget_type": exc.budget_type,
                            "elapsed_ms": exc.elapsed_ms,
                            "limit_ms": exc.limit_ms,
                        },
                        "source": _get_agent_response_source(agent),
                    },
                ),
            )
            raise
        except asyncio.TimeoutError:
            span.stop()
            timeout_detail = build_timeout_error_detail(
                TimeoutCode.TIMEOUT_UPSTREAM,
                elapsed_ms=span.elapsed_ms,
                limit_ms=config.budget.request_timeout_ms,
            )
            await manager.send_message(
                self.connection_id,
                WSEvent(
                    event=ServerEvent.AGENT_ERROR.value,
                    data={
                        "task_id": task_id,
                        "trace_id": trace_id,
                        "timing": build_timing_payload(span),
                        "error": timeout_detail.model_dump(),
                        "source": _get_agent_response_source(agent),
                    },
                ),
            )
            raise
        finally:
            self._current_task_id = None

    async def _handle_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle cancel request (#13) — cancels both stream and non-stream."""
        self._cancel_requested = True
        target_session_key = self._resolve_effective_session_key(params)
        registry_user_id = self._resolve_registry_user_id()
        session_chat_lock = get_ws_session_chat_lock(
            target_session_key,
            user_id=registry_user_id,
        )
        async with session_chat_lock:
            task_id = (
                get_ws_session_chat_task_id(target_session_key, user_id=registry_user_id)
                or self._current_task_id
            )

            # Prefer session-level cancellation so reconnects can stop stale tasks.
            cancelled = await cancel_ws_session_chat_task(
                target_session_key,
                user_id=registry_user_id,
            )
        if not cancelled:
            # Compatibility fallback for direct/unit invocations.
            cancelled = await self._cancel_current_task_for_cleanup()

        return {"cancelled": True, "task_id": task_id, "task_interrupted": cancelled}

    async def _cancel_current_task_for_cleanup(self, timeout: float = 2.0) -> bool:
        """Cancel and await the current background chat task."""
        task = self._current_task
        if task is None:
            return False

        if task.done():
            self._current_task = None
            return False

        self._request_current_task_cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            logger.warning(
                f"Timed out waiting for WS task cleanup: conn={self.connection_id}, "
                f"task_id={self._current_task_id}"
            )
        finally:
            self._current_task = None
            clear_ws_session_chat_task(
                self._default_session_key,
                user_id=self._resolve_registry_user_id(),
                task=task,
            )

        return True

    async def _cleanup_resources(self) -> None:
        # Cancel in-flight concurrent fs/workspace tasks
        for task in list(self._concurrent_tasks):
            if not task.done():
                task.cancel()
        if self._concurrent_tasks:
            await asyncio.gather(*self._concurrent_tasks, return_exceptions=True)
        self._concurrent_tasks.clear()
        await self._workspace_terminal_service.shutdown()
        await self._workspace_watch_service.close()

    async def _emit_workspace_change(self, payload: dict[str, Any]) -> None:
        manager = get_connection_manager()
        await manager.send_message(
            self.connection_id,
            WSEvent(
                event=ServerEvent.SANDBOX_FILE_CHANGED.value,
                data=payload,
            ),
        )

    async def _emit_terminal_stdout(self, payload: dict[str, Any]) -> None:
        manager = get_connection_manager()
        await manager.send_message(
            self.connection_id,
            WSEvent(
                event=ServerEvent.SANDBOX_STDOUT.value,
                data=payload,
            ),
        )

    async def _emit_terminal_closed(self, payload: dict[str, Any]) -> None:
        manager = get_connection_manager()
        await manager.send_message(
            self.connection_id,
            WSEvent(
                event=ServerEvent.TERM_CLOSED.value,
                data=payload,
            ),
        )

    # ========== Confirmation ==========

    async def _handle_confirm_respond(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle confirm.respond — approve/deny a pending confirmation."""
        request_id = params.get("request_id")
        approved = params.get("approved", False)
        reason = params.get("reason")

        if not request_id:
            raise ValueError("request_id is required")

        future = self._pending_confirms.pop(request_id, None)
        if future is None or future.done():
            return {"success": False, "error": "Request not found or expired"}

        future.set_result(approved)

        manager = get_connection_manager()
        await manager.send_message(
            self.connection_id,
            WSEvent(event="confirm.response", data={
                "request_id": request_id,
                "approved": approved,
                "reason": reason,
            }),
        )

        return {"success": True, "request_id": request_id, "approved": approved}

    async def request_confirmation(
        self,
        action: str,
        description: str,
        tool_name: str,
        arguments: dict[str, Any],
        risk_level: str = "medium",
        timeout_seconds: int = 300,
    ) -> bool:
        """Request user confirmation for a dangerous operation."""
        request_id = f"cfm_{uuid4().hex[:8]}"
        loop = asyncio.get_event_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending_confirms[request_id] = future

        manager = get_connection_manager()
        await manager.send_message(
            self.connection_id,
            WSEvent(event="confirm.request", data={
                "request_id": request_id,
                "action": action,
                "description": description,
                "tool_name": tool_name,
                "risk_level": risk_level,
                "timeout_seconds": timeout_seconds,
            }),
        )

        try:
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            self._pending_confirms.pop(request_id, None)
            await manager.send_message(
                self.connection_id,
                WSEvent(event="confirm.timeout", data={"request_id": request_id}),
            )
            return False

    # ========== Status ==========

    async def _handle_status(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle agent.status — returns global agent info."""
        agent = get_agent()
        tool_count = 0
        skill_count = 0
        try:
            if hasattr(agent, 'tools'):
                if hasattr(agent.tools, 'list_tools'):
                    tool_count = len(agent.tools.list_tools())
                elif hasattr(agent.tools, '__len__'):
                    tool_count = len(agent.tools)
            if hasattr(agent, 'skills') and agent.skills:
                skill_count = len(agent.skills)
        except Exception:
            pass

        return {
            "status": "ready",
            "model": getattr(agent, 'model', 'unknown'),
            "provider": getattr(agent, 'provider', 'unknown'),
            "tools": tool_count,
            "skills": skill_count,
            "pending_confirmations": len(self._pending_confirms),
            "current_task_id": self._current_task_id,
        }

    # ========== Sessions ==========

    async def _handle_session_switch(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle session switch (#15) — validates session_key is a non-empty string."""
        manager = get_connection_manager()
        conn = manager.get_connection(self.connection_id)
        if not conn:
            raise ValueError("Connection not found")

        new_session = params.get("session_key", "default")

        # Validate session_key type (#15)
        if not isinstance(new_session, str):
            raise ValueError(
                f"session_key must be a non-empty string, got {type(new_session).__name__}"
            )
        new_session = new_session.strip()
        if not new_session:
            raise ValueError("session_key must be a non-empty string")

        conn.session_key = new_session
        self._default_session_key = new_session
        return {"session_key": new_session, "switched": True}

    async def _handle_session_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle session list."""
        agent = get_agent()
        try:
            sessions = agent.sessions.list_sessions()
            return {
                "sessions": [
                    {
                        "key": s.session_key if hasattr(s, 'session_key') else str(s),
                        "message_count": len(s.messages) if hasattr(s, 'messages') else 0,
                    }
                    for s in sessions
                ]
            }
        except Exception:
            return {"sessions": []}

    async def _handle_session_search(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ``session.search``.

        Runs a persisted-history search across one or all sessions so the
        client / model can recover earlier tool calls, tool results, or
        message bodies that have fallen out of the runtime context due to
        compaction.

        Params:
          q (str, required): search query.
          session_key (str, optional): restrict to a single session.
          regex (bool, default False): interpret ``q`` as a regex.
          case_sensitive (bool, default False).
          roles (list[str], optional): restrict to these message roles
              (e.g. ``["tool", "assistant"]``).
          include_extras (bool, default True): also match serialized
              ``tool_call_id`` / ``tool_calls`` / ``attachments`` metadata.
          limit (int, default 50, max 500).
          offset (int, default 0).
          max_content_length (int, default 2000): truncate returned
              ``content`` to at most N characters.
        """
        import re as _re

        query = params.get("q") or params.get("query")
        if not isinstance(query, str) or not query:
            return {"error": "INVALID_QUERY", "message": "q is required"}

        # Wait for any pending chat task so the freshest turn is persisted
        # before we search.
        if self._current_task and not self._current_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._current_task), timeout=30.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass

        agent = get_agent()
        sessions_manager = getattr(agent, "sessions", None)
        if sessions_manager is None:
            return {"error": "INTERNAL_ERROR", "message": "Session manager unavailable"}

        session_key = params.get("session_key")
        if session_key is not None and not isinstance(session_key, str):
            return {"error": "INVALID_QUERY", "message": "session_key must be a string"}

        roles_raw = params.get("roles")
        roles: list[str] | None
        if roles_raw is None:
            roles = None
        elif isinstance(roles_raw, (list, tuple)):
            roles = [str(r) for r in roles_raw if r]
            if not roles:
                roles = None
        else:
            return {"error": "INVALID_QUERY", "message": "roles must be a list"}

        def _as_bool(value: Any, default: bool) -> bool:
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        def _as_int(value: Any, default: int, *, low: int, high: int) -> int:
            if value is None:
                return default
            try:
                n = int(value)
            except (TypeError, ValueError):
                return default
            return max(low, min(high, n))

        regex = _as_bool(params.get("regex"), False)
        case_sensitive = _as_bool(params.get("case_sensitive"), False)
        include_extras = _as_bool(params.get("include_extras"), True)
        limit = _as_int(params.get("limit"), 50, low=1, high=500)
        offset = _as_int(params.get("offset"), 0, low=0, high=1_000_000)
        max_content_length_raw = params.get("max_content_length", 2000)
        if max_content_length_raw is None:
            max_content_length: int | None = None
        else:
            max_content_length = _as_int(
                max_content_length_raw, 2000, low=1, high=100_000
            )

        try:
            hits = sessions_manager.search_messages(
                query,
                session_key=session_key,
                regex=regex,
                case_sensitive=case_sensitive,
                roles=roles,
                include_extras=include_extras,
                limit=limit,
                offset=offset,
                max_content_length=max_content_length,
            )
        except _re.error as exc:
            return {"error": "INVALID_REGEX", "message": str(exc)}
        except ValueError as exc:
            return {"error": "INVALID_QUERY", "message": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"session.search failed: {exc}")
            return {"error": "INTERNAL_ERROR", "message": "Search failed"}

        return {
            "query": query,
            "total": len(hits),
            "limit": limit,
            "offset": offset,
            "hits": [
                {
                    "session_key": h.session_key,
                    "seq": h.seq,
                    "role": h.role,
                    "content": h.content,
                    "timestamp": h.timestamp,
                    "matched_in": h.matched_in,
                    "snippet": h.snippet,
                    "extras": dict(h.extras) if h.extras else {},
                }
                for h in hits
            ],
        }

    async def _handle_session_clear(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle session clear."""
        agent = get_agent()
        session_key = params.get("session_key", "default")

        session = agent.sessions.get(session_key)
        if session:
            count = len(session.messages)
            session.messages.clear()
            agent.sessions.save(session)
            return {"cleared": True, "messages_removed": count}

        return {"cleared": False, "messages_removed": 0}

    async def _handle_session_export(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle session.export."""
        # Wait for any pending chat task to finish so messages are persisted
        if self._current_task and not self._current_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._current_task), timeout=30.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass

        agent = get_agent()
        session_key = params.get("session_key", "default")

        # Prefer agent's current in-memory session (most up-to-date)
        session = None
        if (
            hasattr(agent, '_session')
            and hasattr(agent._session, 'session_key')
            and agent._session.session_key == session_key
        ):
            session = agent._session
        else:
            session = agent.sessions.get(session_key)

        if session and hasattr(session, 'messages'):
            return {
                "success": True,
                "state": {
                    "version": "1.0",
                    "session_key": session_key,
                    "messages": [
                        dict(m)
                        for m in (session.messages if isinstance(session.messages, list) else [])
                        if isinstance(m, dict)
                    ],
                },
            }
        return {"success": True, "state": {"version": "1.0", "session_key": session_key, "messages": []}}

    async def _handle_session_import(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle session.import (#12) — actually persist imported messages."""
        state = params.get("state")
        if not state:
            raise ValueError("state is required")

        if not isinstance(state, dict):
            raise ValueError("state must be a JSON object")

        session_key = state.get("session_key", "default")
        messages = state.get("messages", [])

        if not isinstance(messages, list):
            raise ValueError("state.messages must be an array")

        agent = get_agent()

        try:
            session = agent.sessions.get_or_create(session_key)
            imported_messages: list[tuple[str, str, dict[str, Any]]] = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    extras = {
                        key: value
                        for key, value in msg.items()
                        if key not in {"role", "content", "timestamp"}
                    }
                    if "media" in extras:
                        extras["media"] = _validate_media_paths(_string_list_from_any(extras.get("media")))
                    if "attachments" in extras:
                        extras["attachments"] = _validate_attachment_paths(
                            _normalize_attachment_refs(extras.get("attachments"))
                        )
                    if role and (content or extras.get("attachments") or extras.get("media")):
                        imported_messages.append((role, content, extras))

            # Replace messages only after the full import payload validates.
            session.messages.clear()
            for role, content, extras in imported_messages:
                session.add_message(role, content, **extras)
            agent.sessions.save(session)

            return {
                "success": True,
                "restored": {
                    "session_key": session_key,
                    "messages_imported": len(session.messages),
                },
            }
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Session import failed: {e}")
            return {
                "success": False,
                "error": f"Import failed: {e}",
            }

    # ========== Subscriptions ==========

    async def _handle_subscribe(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle event subscription (#16) — validates events is list[str]."""
        manager = get_connection_manager()
        events = params.get("events", [])

        # Validate events is a list of strings (#16)
        if not isinstance(events, list):
            raise ValueError(
                f"events must be a list of strings, got {type(events).__name__}"
            )
        # Ensure all items are strings
        events = [str(e) for e in events if isinstance(e, str)]

        manager.subscribe(self.connection_id, events)
        return {"subscribed": events}

    async def _handle_unsubscribe(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle event unsubscription (#16) — validates events is list[str]."""
        manager = get_connection_manager()
        events = params.get("events", [])

        # Validate events is a list of strings (#16)
        if not isinstance(events, list):
            raise ValueError(
                f"events must be a list of strings, got {type(events).__name__}"
            )
        events = [str(e) for e in events if isinstance(e, str)]

        manager.unsubscribe(self.connection_id, events)
        return {"unsubscribed": events}

    # ========== Workspace ==========

    async def _handle_workspace_tree(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return workspace directory tree."""
        from spoon_bot.gateway.api.v1.workspace import _build_tree, TreeNode
        from pathlib import Path

        agent = get_agent()
        workspace = Path(getattr(agent, "workspace", Path.home() / ".spoon-bot" / "workspace"))

        sub_path = params.get("path", "")
        depth = min(max(int(params.get("depth", 3)), 1), 10)
        include_hidden = bool(params.get("include_hidden", False))

        target = (workspace / sub_path).resolve()
        if not str(target).startswith(str(workspace.resolve())):
            raise ValueError("Path is outside workspace boundary")
        if not target.exists():
            raise ValueError(f"Path not found: {sub_path}")

        nodes = _build_tree(target, max_depth=depth, include_hidden=include_hidden)
        return {"tree": [n.model_dump() for n in nodes]}

    async def _handle_fs_list(self, params: dict[str, Any]) -> dict[str, Any]:
        path = str(params.get("path", "")).strip() or "/workspace"
        cursor = str(params["cursor"]).strip() if params.get("cursor") else None
        limit = int(params.get("limit", 200))
        return await self._workspace_fs_service.list(path, cursor=cursor, limit=limit)

    async def _handle_fs_stat(self, params: dict[str, Any]) -> dict[str, Any]:
        path = str(params.get("path", "")).strip()
        if not path:
            raise ValueError("path is required")
        return await self._workspace_fs_service.stat(path)

    async def _handle_fs_read(self, params: dict[str, Any]) -> dict[str, Any]:
        path = str(params.get("path", "")).strip()
        if not path:
            raise ValueError("path is required")
        encoding = str(params.get("encoding", "utf-8"))
        offset = int(params.get("offset", 0))
        limit = int(params.get("limit", 262144))
        return await self._workspace_fs_service.read(
            path,
            encoding=encoding,
            offset=offset,
            limit=limit,
        )

    async def _handle_fs_write(self, params: dict[str, Any]) -> dict[str, Any]:
        path = str(params.get("path", "")).strip()
        if not path:
            raise ValueError("path is required")
        if "content" not in params:
            raise ValueError("content is required")
        return await self._workspace_fs_service.write(
            path,
            content=str(params.get("content", "")),
            encoding=str(params.get("encoding", "utf-8")),
            create=bool(params.get("create", True)),
            truncate=bool(params.get("truncate", True)),
        )

    async def _handle_fs_mkdir(self, params: dict[str, Any]) -> dict[str, Any]:
        path = str(params.get("path", "")).strip()
        if not path:
            raise ValueError("path is required")
        return await self._workspace_fs_service.mkdir(
            path,
            recursive=bool(params.get("recursive", False)),
        )

    async def _handle_fs_rename(self, params: dict[str, Any]) -> dict[str, Any]:
        from_path = str(params.get("from_path", "")).strip()
        to_path = str(params.get("to_path", "")).strip()
        if not from_path:
            raise ValueError("from_path is required")
        if not to_path:
            raise ValueError("to_path is required")
        return await self._workspace_fs_service.rename(
            from_path,
            to_path,
            overwrite=bool(params.get("overwrite", False)),
        )

    async def _handle_fs_remove(self, params: dict[str, Any]) -> dict[str, Any]:
        path = str(params.get("path", "")).strip()
        if not path:
            raise ValueError("path is required")
        return await self._workspace_fs_service.remove(
            path,
            recursive=bool(params.get("recursive", False)),
        )

    async def _handle_fs_watch(self, params: dict[str, Any]) -> dict[str, Any]:
        watch_path = str(params.get("path", "")).strip()
        if not watch_path:
            raise ValueError("path is required")
        recursive = bool(params.get("recursive", False))
        watch_id = await self._workspace_watch_service.add_watch(watch_path, recursive=recursive)
        return {"watch_id": watch_id}

    async def _handle_fs_unwatch(self, params: dict[str, Any]) -> dict[str, Any]:
        watch_id = str(params.get("watch_id", "")).strip()
        if not watch_id:
            raise ValueError("watch_id is required")
        removed = await self._workspace_watch_service.remove_watch(watch_id)
        return {"watch_id": watch_id, "removed": removed}

    async def _handle_term_open(self, params: dict[str, Any]) -> dict[str, Any]:
        env = params.get("env")
        if env is not None and not isinstance(env, dict):
            raise ValueError("env must be an object")
        return await self._workspace_terminal_service.open(
            cwd=str(params["cwd"]).strip() if params.get("cwd") else None,
            shell=str(params["shell"]).strip() if params.get("shell") else None,
            cols=int(params.get("cols", 120)),
            rows=int(params.get("rows", 32)),
            env=env if isinstance(env, dict) else None,
        )

    async def _handle_term_input(self, params: dict[str, Any]) -> dict[str, Any]:
        term_id = str(params.get("term_id", "")).strip()
        if not term_id:
            raise ValueError("term_id is required")
        if "input" not in params:
            raise ValueError("input is required")
        return await self._workspace_terminal_service.input(term_id, str(params.get("input", "")))

    async def _handle_term_resize(self, params: dict[str, Any]) -> dict[str, Any]:
        term_id = str(params.get("term_id", "")).strip()
        if not term_id:
            raise ValueError("term_id is required")
        if "cols" not in params or "rows" not in params:
            raise ValueError("cols and rows are required")
        return await self._workspace_terminal_service.resize(
            term_id,
            cols=int(params.get("cols", 0)),
            rows=int(params.get("rows", 0)),
        )

    async def _handle_term_close(self, params: dict[str, Any]) -> dict[str, Any]:
        term_id = str(params.get("term_id", "")).strip()
        if not term_id:
            raise ValueError("term_id is required")
        return await self._workspace_terminal_service.close(term_id)

    # ========== Audio Streaming ==========

    def _get_audio_stream_manager(self):
        """Lazy-init the audio stream manager."""
        if self._audio_stream_manager is None:
            try:
                from spoon_bot.services.audio.factory import create_transcriber
                from spoon_bot.services.audio.streaming import AudioStreamManager

                config = get_config()
                transcriber = create_transcriber(
                    provider=config.audio.stt_provider,
                    model=config.audio.stt_model,
                )
                self._audio_stream_manager = AudioStreamManager(
                    transcriber=transcriber,
                    max_stream_duration_seconds=config.audio.max_audio_duration_seconds,
                    max_audio_size_bytes=config.audio.max_audio_size_mb * 1024 * 1024,
                )
            except Exception as e:
                logger.error(f"Failed to initialize audio stream manager: {e}")
                raise ValueError(f"Audio streaming unavailable: {e}")
        return self._audio_stream_manager

    async def _handle_audio_stream_start(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle audio.stream.start — begin buffering audio chunks."""
        manager = self._get_audio_stream_manager()

        audio_format = params.get("format", "wav")
        language = params.get("language")
        sample_rate = params.get("sample_rate", 16000)
        channels = params.get("channels", 1)
        text_message = params.get("message", "")

        if not isinstance(audio_format, str):
            raise ValueError("format must be a string")

        session = manager.start_session(
            connection_id=self.connection_id,
            audio_format=audio_format,
            language=language,
            sample_rate=sample_rate,
            channels=channels,
            text_message=text_message,
        )

        return {
            "streaming": True,
            "session_id": session.session_id,
            "format": audio_format,
        }

    async def handle_audio_binary(self, data: bytes) -> None:
        """Handle incoming binary audio chunk data."""
        if self._audio_stream_manager is None:
            return  # No active stream — ignore
        try:
            self._audio_stream_manager.add_chunk(self.connection_id, data)
        except ValueError as e:
            logger.warning(f"Audio chunk rejected: {e}")
            manager = get_connection_manager()
            await manager.send_to(
                self.connection_id,
                WSEvent(
                    event="audio.stream.error",
                    data={"error": str(e)},
                ).to_dict(),
            )

    async def _handle_audio_stream_end(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle audio.stream.end — transcribe buffered audio."""
        if self._audio_stream_manager is None:
            raise ValueError("No audio stream active")

        try:
            result = await self._audio_stream_manager.end_session(self.connection_id)

            # Optionally process through agent
            process = params.get("process", True)
            if process and result.text:
                agent = get_agent()
                text_message = params.get("message", "")
                combined = f"{text_message}\n\n[Voice input]: {result.text}" if text_message else result.text
                response = await agent.process(message=combined)
            else:
                response = None

            return {
                "transcription": {
                    "text": result.text,
                    "language": result.language,
                    "duration_seconds": result.duration_seconds,
                    "provider": result.provider,
                },
                "response": response,
            }
        except Exception as e:
            logger.error(f"Audio stream end failed: {e}")
            self._audio_stream_manager.cancel_session(self.connection_id)
            raise ValueError(f"Audio transcription failed: {e}")
