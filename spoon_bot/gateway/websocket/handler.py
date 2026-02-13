"""WebSocket endpoint and message handler.

This module implements the WebSocket endpoint for real-time agent communication,
including streaming responses, user confirmation flow, and event notifications.

Based on: docs/plans/2025-02-05-spoon-bot-api-design.md
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from uuid import uuid4
from typing import Any, Callable, Coroutine

from fastapi import WebSocket, WebSocketDisconnect, Query
from loguru import logger

from spoon_bot.gateway.app import get_agent, get_connection_manager, get_config, is_auth_required
from spoon_bot.gateway.auth.jwt import verify_token
from spoon_bot.gateway.auth.api_key import verify_api_key
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
    handler = WebSocketHandler(conn_id)

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

                        async def _run_chat(req: WSRequest = _req) -> None:
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

                        handler._current_task = asyncio.create_task(_run_chat())
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
        await manager.disconnect(conn_id)


class WebSocketHandler:
    """
    Handles WebSocket message processing.

    Uses the global AgentLoop from the gateway app for all operations.
    Supports confirmation flow for dangerous tool calls.
    """

    def __init__(self, connection_id: str, session_id: str | None = None):
        self.connection_id = connection_id
        self.session_id = session_id or connection_id
        self._cancel_requested = False
        self._current_task_id: str | None = None
        self._current_task: asyncio.Task | None = None
        self._pending_confirms: dict[str, asyncio.Future] = {}

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
            # Subscription
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
        }

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

    # ========== Chat ==========

    async def _handle_chat(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle chat.send / agent.chat — uses the global AgentLoop."""
        agent = get_agent()
        manager = get_connection_manager()
        conn = manager.get_connection(self.connection_id)

        if not conn:
            raise ValueError("Connection not found")

        message = params.get("message", "")
        if not message:
            raise ValueError("Message is required")

        session_key = params.get("session_key", conn.session_key)
        # Validate session_key type
        if not isinstance(session_key, str) or not session_key.strip():
            session_key = conn.session_key

        stream = params.get("stream", False)
        thinking = params.get("thinking", False)
        task_id = f"task_{uuid4().hex[:8]}"
        self._current_task_id = task_id
        self._cancel_requested = False

        # Switch agent session to the requested session_key (#11)
        _switch_agent_session(agent, session_key)

        # Emit thinking event
        await manager.send_message(
            self.connection_id,
            WSEvent(event="agent.thinking", data={"task_id": task_id, "status": "processing"}),
        )

        if stream:
            # Streaming mode: emit chunks via WSEvent
            full_content = ""
            async for chunk_data in agent.stream(message=message, thinking=thinking):
                if self._cancel_requested:
                    break

                chunk_type = chunk_data.get("type", "content")
                # Support both "delta" and "content" keys (#10)
                delta = chunk_data.get("delta", "") or chunk_data.get("content", "")
                metadata = chunk_data.get("metadata", {})

                if chunk_type == "done":
                    # Send stream done event
                    await manager.send_message(
                        self.connection_id,
                        WSEvent(event=ServerEvent.AGENT_STREAM_DONE.value, data={
                            "task_id": task_id,
                            "content": full_content,
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
                            }),
                        )

            response = full_content
        else:
            # Non-streaming mode
            if thinking:
                response, thinking_content = await agent.process_with_thinking(message=message)
            else:
                response = await agent.process(message=message)
                thinking_content = None

        # Emit complete event
        complete_data: dict[str, Any] = {
            "task_id": task_id,
            "status": "done",
            "response": response[:200] if isinstance(response, str) else str(response)[:200],
        }
        if not stream and thinking and thinking_content:
            complete_data["thinking_content"] = thinking_content

        await manager.send_message(
            self.connection_id,
            WSEvent(event="agent.complete", data=complete_data),
        )

        self._current_task_id = None
        result: dict[str, Any] = {
            "success": True,
            "task_id": task_id,
            "content": response,
            "session_key": session_key,
        }
        if not stream and thinking and thinking_content:
            result["thinking_content"] = thinking_content

        return result

    async def _handle_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle cancel request (#13) — cancels both stream and non-stream."""
        self._cancel_requested = True
        task_id = self._current_task_id

        # Also cancel the asyncio task if running (#13)
        cancelled = False
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            cancelled = True

        return {"cancelled": True, "task_id": task_id, "task_interrupted": cancelled}

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
        agent = get_agent()
        session_key = params.get("session_key", "default")
        session = agent.sessions.get(session_key)
        if session and hasattr(session, 'messages'):
            return {
                "success": True,
                "state": {
                    "version": "1.0",
                    "session_key": session_key,
                    "messages": [
                        {"role": m.get("role", ""), "content": m.get("content", "")}
                        for m in (session.messages if isinstance(session.messages, list) else [])
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
            # Replace messages with imported ones
            session.messages.clear()
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role and content:
                        session.add_message(role, content)
            agent.sessions.save(session)

            return {
                "success": True,
                "restored": {
                    "session_key": session_key,
                    "messages_imported": len(session.messages),
                },
            }
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
