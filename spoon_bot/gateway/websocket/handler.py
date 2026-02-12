"""WebSocket endpoint and message handler.

This module implements the WebSocket endpoint for real-time agent communication,
including streaming responses, user confirmation flow, and event notifications.

Based on: docs/plans/2025-02-05-spoon-bot-api-design.md
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4
from typing import Any, Callable, Coroutine

from fastapi import WebSocket, WebSocketDisconnect, Query
from loguru import logger

from spoon_bot.gateway.app import get_agent, get_connection_manager, get_config, is_auth_required
from spoon_bot.gateway.auth.jwt import verify_token
from spoon_bot.gateway.auth.api_key import verify_api_key
from spoon_bot.gateway.errors import TimeoutCode, GatewayErrorCode
from spoon_bot.gateway.observability.tracing import new_trace_id, TimerSpan, build_timing_payload
from spoon_bot.gateway.observability.budget import BudgetExhaustedError, check_budget
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

    # Authenticate
    user_id = None
    session_key = "default"

    # If auth is not required, allow anonymous connections
    if not is_auth_required():
        user_id = "anonymous"
    else:
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
        await websocket.close(code=4001, reason="Authentication failed")
        return

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

        # Message loop — handles both JSON text frames and binary audio frames
        while True:
            try:
                raw = await websocket.receive()

                # --- Binary frame: audio data ---
                if "bytes" in raw and raw["bytes"]:
                    try:
                        handler._handle_audio_binary(raw["bytes"])
                    except Exception as exc:
                        await manager.send_message(
                            conn_id,
                            WSEvent(
                                event=ServerEvent.AUDIO_ERROR.value,
                                data={"error": str(exc)},
                            ),
                        )
                    continue

                # --- Text frame: JSON message ---
                if "text" in raw and raw["text"]:
                    data = json.loads(raw["text"])
                else:
                    continue

                message = parse_message(data)

                if message.type.value == "ping":
                    await manager.send_message(
                        conn_id,
                        {"type": "pong", "timestamp": data.get("timestamp")},
                    )
                    continue

                if isinstance(message, WSRequest):
                    # Run chat and audio requests as background tasks so the
                    # message loop stays free for cancel / status requests.
                    _is_chat = message.method in (
                        "agent.chat",
                        ClientMethod.CHAT_SEND.value,
                    )
                    _is_audio = message.method in (
                        ClientMethod.AUDIO_SEND.value,
                        ClientMethod.AUDIO_STREAM_START.value,
                        ClientMethod.AUDIO_STREAM_END.value,
                    )

                    if _is_chat or _is_audio:
                        _req = message  # capture for closure

                        async def _run_handler(req: WSRequest = _req) -> None:
                            try:
                                if req.method == ClientMethod.AUDIO_SEND.value:
                                    result = await handler._handle_audio_send(req.params)
                                elif req.method == ClientMethod.AUDIO_STREAM_START.value:
                                    result = await handler._handle_audio_stream_start(req.params)
                                elif req.method == ClientMethod.AUDIO_STREAM_END.value:
                                    result = await handler._handle_audio_stream_end(req.params)
                                else:
                                    result = await handler._handle_chat(req.params)
                                await manager.send_message(
                                    conn_id, WSResponse(id=req.id, result=result),
                                )
                            except Exception as exc:
                                logger.error(f"Handler error for {req.method}: {exc}")
                                await manager.send_message(
                                    conn_id,
                                    WSError(id=req.id, code="HANDLER_ERROR", message=str(exc)),
                                )

                        handler._current_task = asyncio.create_task(_run_handler())
                    else:
                        response = await handler.handle_request(message)
                        await manager.send_message(conn_id, response)

            except json.JSONDecodeError as e:
                await manager.send_message(
                    conn_id,
                    WSError(
                        code="INVALID_JSON",
                        message=f"Invalid JSON: {e}",
                    ),
                )

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
        # Propagate cancellation to any running task
        if handler._current_task is not None and not handler._current_task.done():
            handler._current_task.cancel()
            handler._cancel_requested = True
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
        self._pending_confirms: dict[str, asyncio.Future] = {}
        self._current_task: asyncio.Task | None = None

        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            # Chat methods
            "agent.chat": self._handle_chat,
            ClientMethod.CHAT_SEND.value: self._handle_chat,  # "chat.send" -> same handler
            "agent.cancel": self._handle_cancel,
            ClientMethod.CHAT_CANCEL.value: self._handle_cancel,
            # Audio methods
            ClientMethod.AUDIO_SEND.value: self._handle_audio_send,
            ClientMethod.AUDIO_STREAM_START.value: self._handle_audio_stream_start,
            ClientMethod.AUDIO_STREAM_END.value: self._handle_audio_stream_end,
            # Confirmation
            ClientMethod.CONFIRM_RESPOND.value: self._handle_confirm_respond,
            # Status
            "agent.status": self._handle_status,
            # Tool management
            "tools.list": self._handle_tools_list,
            "tools.activate": self._handle_tools_activate,
            "tools.deactivate": self._handle_tools_deactivate,
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
            result = await handler(request.params)
            return WSResponse(id=request.id, result=result)
        except Exception as e:
            logger.error(f"Error handling {request.method}: {e}")
            return WSError(
                id=request.id,
                code="HANDLER_ERROR",
                message=str(e),
            )

    # ========== Chat ==========

    async def _handle_chat(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle chat.send / agent.chat -- uses the global AgentLoop.

        Adds tracing (trace_id), timing (TimerSpan), budget checks, and
        standardized timeout/budget error codes to all emitted events.
        """
        agent = get_agent()
        manager = get_connection_manager()
        config = get_config()
        conn = manager.get_connection(self.connection_id)

        if not conn:
            raise ValueError("Connection not found")

        message = params.get("message", "")
        if not message:
            raise ValueError("Message is required")

        session_key = params.get("session_key", conn.session_key)
        stream = params.get("stream", False)
        thinking = params.get("thinking", False)
        task_id = f"task_{uuid4().hex[:8]}"
        self._current_task_id = task_id
        self._cancel_requested = False

        # --- Tracing and timing ---
        trace_id = new_trace_id()
        span = TimerSpan("ws_chat")

        # Track the current task for cancellation propagation
        self._current_task = asyncio.current_task()

        try:
            # Emit thinking event
            await manager.send_message(
                self.connection_id,
                WSEvent(event="agent.thinking", data={
                    "task_id": task_id,
                    "status": "processing",
                    "trace_id": trace_id,
                }),
            )

            if stream:
                # Streaming mode: emit chunks via WSEvent
                full_content = ""
                async for chunk_data in agent.stream(message=message, thinking=thinking):
                    if self._cancel_requested:
                        break

                    # Check stream budget
                    check_budget(
                        "stream",
                        config.budget.stream_timeout_ms,
                        span.elapsed_ms,
                    )

                    chunk_type = chunk_data.get("type", "content")
                    delta = chunk_data.get("delta", "")
                    metadata = chunk_data.get("metadata", {})

                    if chunk_type == "done":
                        # Send stream done event
                        await manager.send_message(
                            self.connection_id,
                            WSEvent(event=ServerEvent.AGENT_STREAM_DONE.value, data={
                                "task_id": task_id,
                                "content": full_content,
                                "trace_id": trace_id,
                            }),
                        )
                    else:
                        if chunk_type == "content":
                            full_content += delta

                        # Send stream chunk event
                        await manager.send_message(
                            self.connection_id,
                            WSEvent(event=ServerEvent.AGENT_STREAM_CHUNK.value, data={
                                "task_id": task_id,
                                "type": chunk_type,
                                "delta": delta,
                                "metadata": metadata,
                                "trace_id": trace_id,
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

            # Stop the timer and build timing payload
            span.stop()
            timing = build_timing_payload(span)

            # Check overall request budget after completion
            check_budget(
                "request",
                config.budget.request_timeout_ms,
                span.elapsed_ms,
            )

            # Emit complete event with trace_id and timing
            complete_data: dict[str, Any] = {
                "task_id": task_id,
                "status": "done",
                "response": response[:200] if isinstance(response, str) else str(response)[:200],
                "trace_id": trace_id,
                "timing": timing,
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
                "trace_id": trace_id,
                "timing": timing,
            }
            if not stream and thinking and thinking_content:
                result["thinking_content"] = thinking_content

            return result

        except asyncio.TimeoutError:
            span.stop()
            timing = build_timing_payload(span)
            error_code = TimeoutCode.TIMEOUT_UPSTREAM.value
            error_data: dict[str, Any] = {
                "task_id": task_id,
                "trace_id": trace_id,
                "timing": timing,
                "error": {
                    "code": error_code,
                    "message": "Upstream service timed out",
                    "elapsed_ms": span.elapsed_ms,
                    "limit_ms": config.budget.request_timeout_ms,
                },
            }
            await manager.send_message(
                self.connection_id,
                WSEvent(event=ServerEvent.AGENT_ERROR.value, data=error_data),
            )
            self._current_task_id = None
            raise

        except BudgetExhaustedError as exc:
            span.stop()
            timing = build_timing_payload(span)
            # Map budget_type to the appropriate timeout code
            code_map = {
                "request": TimeoutCode.TIMEOUT_TOTAL.value,
                "stream": TimeoutCode.TIMEOUT_TOTAL.value,
                "tool": TimeoutCode.TIMEOUT_TOOL.value,
            }
            error_code = code_map.get(exc.budget_type, GatewayErrorCode.BUDGET_EXHAUSTED.value)
            error_data = {
                "task_id": task_id,
                "trace_id": trace_id,
                "timing": timing,
                "error": {
                    "code": error_code,
                    "message": str(exc),
                    "budget_type": exc.budget_type,
                    "elapsed_ms": exc.elapsed_ms,
                    "limit_ms": exc.limit_ms,
                },
            }
            await manager.send_message(
                self.connection_id,
                WSEvent(event=ServerEvent.AGENT_ERROR.value, data=error_data),
            )
            self._current_task_id = None
            raise

        except asyncio.CancelledError:
            span.stop()
            timing = build_timing_payload(span)
            error_data = {
                "task_id": task_id,
                "trace_id": trace_id,
                "timing": timing,
                "error": {
                    "code": GatewayErrorCode.CANCELLED.value,
                    "message": "Task was cancelled",
                },
            }
            await manager.send_message(
                self.connection_id,
                WSEvent(event=ServerEvent.AGENT_ERROR.value, data=error_data),
            )
            self._current_task_id = None
            raise

        finally:
            self._current_task = None

    async def _handle_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle cancel request.

        Sets the cancel flag and also cancels the underlying asyncio task
        so that long-running agent/tool operations are interrupted.
        """
        self._cancel_requested = True
        task_id = self._current_task_id
        # Cancel the running asyncio task if one exists
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
        return {"cancelled": True, "task_id": task_id}

    # ========== Audio ==========

    def _handle_audio_binary(self, data: bytes) -> None:
        """Handle a binary WebSocket frame containing audio chunk data.

        Binary frames are used during audio streaming sessions (after
        audio.stream.start). The data is appended to the active stream buffer.
        """
        try:
            from spoon_bot.services.audio.streaming import AudioStreamManager
        except ImportError:
            raise ValueError("Audio streaming service not available")

        stream_mgr = self._get_stream_manager()
        if not stream_mgr.has_active_session(self.connection_id):
            raise ValueError("No active audio stream. Send audio.stream.start first.")
        stream_mgr.add_chunk(self.connection_id, data)

    def _get_stream_manager(self):
        """Get or create the AudioStreamManager singleton."""
        if not hasattr(self, "_stream_manager") or self._stream_manager is None:
            from spoon_bot.services.audio.factory import create_transcriber
            from spoon_bot.services.audio.streaming import AudioStreamManager

            config = get_config()
            transcriber = create_transcriber(config.audio.stt_provider)
            self._stream_manager = AudioStreamManager(
                transcriber=transcriber,
                max_audio_size_bytes=config.audio.max_audio_size_mb * 1024 * 1024,
                max_stream_duration_seconds=config.audio.max_audio_duration_seconds,
            )
        return self._stream_manager

    async def _handle_audio_send(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle audio.send — send a complete audio message (base64) for processing.

        Params:
            data: Base64-encoded audio data
            format: Audio format (wav, mp3, etc.)
            message: Optional accompanying text
            language: Optional language hint
            stream: Whether to stream the response
        """
        audio_data = params.get("data", "")
        audio_format = params.get("format", "wav")
        text_message = params.get("message", "")
        language = params.get("language")
        do_stream = params.get("stream", False)

        if not audio_data:
            raise ValueError("Audio data is required")

        manager = get_connection_manager()

        # Emit received event
        await manager.send_message(
            self.connection_id,
            WSEvent(event=ServerEvent.AUDIO_RECEIVED.value, data={"format": audio_format}),
        )

        # Process through audio pipeline
        from spoon_bot.gateway.api.v1.agent import _process_audio_input

        effective_message, transcription_info = await _process_audio_input(
            audio_data=audio_data,
            audio_format=audio_format,
            message=text_message,
            language=language,
        )

        # Emit transcription result
        if transcription_info:
            await manager.send_message(
                self.connection_id,
                WSEvent(
                    event=ServerEvent.AUDIO_TRANSCRIPTION.value,
                    data={
                        "text": transcription_info.text,
                        "language": transcription_info.language,
                        "duration_seconds": transcription_info.duration_seconds,
                    },
                ),
            )

        if not effective_message:
            raise ValueError("No speech detected in audio")

        # Route to chat handler with the transcribed/processed message
        chat_params = {
            "message": effective_message,
            "stream": do_stream,
        }
        return await self._handle_chat(chat_params)

    async def _handle_audio_stream_start(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle audio.stream.start — begin a streaming audio session.

        After this, the client sends binary frames with audio chunks,
        then sends audio.stream.end to finalize.
        """
        audio_format = params.get("format", "wav")
        language = params.get("language")
        sample_rate = params.get("sample_rate", 16000)
        channels = params.get("channels", 1)
        text_message = params.get("message", "")

        stream_mgr = self._get_stream_manager()
        stream_mgr.start_session(
            connection_id=self.connection_id,
            audio_format=audio_format,
            language=language,
            sample_rate=sample_rate,
            channels=channels,
            text_message=text_message,
        )

        return {
            "success": True,
            "session_id": self.connection_id,
            "message": "Audio stream started. Send binary frames, then audio.stream.end.",
        }

    async def _handle_audio_stream_end(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle audio.stream.end — finalize a streaming audio session.

        Transcribes the accumulated audio and sends the result through the chat pipeline.
        """
        do_stream = params.get("stream", False)
        manager = get_connection_manager()
        stream_mgr = self._get_stream_manager()

        # Transcribe accumulated audio
        result = await stream_mgr.end_session(self.connection_id)

        # Emit transcription
        await manager.send_message(
            self.connection_id,
            WSEvent(
                event=ServerEvent.AUDIO_TRANSCRIPTION.value,
                data={
                    "text": result.text,
                    "language": result.language,
                    "duration_seconds": result.duration_seconds,
                },
            ),
        )

        if result.is_empty:
            return {"success": True, "transcription": "", "message": "No speech detected"}

        # Get the original text message from the session (if any)
        text_message = params.get("message", "")
        if text_message:
            effective_message = f"{text_message}\n\n[Voice input]: {result.text}"
        else:
            effective_message = result.text

        # Route to chat handler
        chat_params = {
            "message": effective_message,
            "stream": do_stream,
        }
        return await self._handle_chat(chat_params)

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

    # ========== Tool management ==========

    async def _handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools.list — list all tools with active/inactive status."""
        agent = get_agent()
        tools = agent.get_available_tools()
        active = [t for t in tools if t["active"]]
        inactive = [t for t in tools if not t["active"]]
        return {
            "active": [t["name"] for t in active],
            "inactive": [t["name"] for t in inactive],
            "total": len(tools),
        }

    async def _handle_tools_activate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools.activate — dynamically load a tool into the agent."""
        agent = get_agent()
        name = params.get("name")
        if not name:
            raise ValueError("Tool name is required")
        success = agent.add_tool(name)
        return {"name": name, "activated": success}

    async def _handle_tools_deactivate(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle tools.deactivate — remove a tool from the agent."""
        agent = get_agent()
        name = params.get("name")
        if not name:
            raise ValueError("Tool name is required")
        success = agent.remove_tool(name)
        return {"name": name, "deactivated": success}

    # ========== Sessions ==========

    async def _handle_session_switch(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle session switch."""
        manager = get_connection_manager()
        conn = manager.get_connection(self.connection_id)
        if not conn:
            raise ValueError("Connection not found")

        new_session = params.get("session_key", "default")
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
        """Handle session.import."""
        state = params.get("state")
        if not state:
            raise ValueError("state is required")
        return {"success": True, "restored": {"session_id": self.session_id}}

    # ========== Subscriptions ==========

    async def _handle_subscribe(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle event subscription."""
        manager = get_connection_manager()
        events = params.get("events", [])
        manager.subscribe(self.connection_id, events)
        return {"subscribed": events}

    async def _handle_unsubscribe(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle event unsubscription."""
        manager = get_connection_manager()
        events = params.get("events", [])
        manager.unsubscribe(self.connection_id, events)
        return {"unsubscribed": events}
