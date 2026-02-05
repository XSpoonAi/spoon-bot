"""WebSocket endpoint and message handler.

This module implements the WebSocket endpoint for real-time agent communication,
including streaming responses, user confirmation flow, and event notifications.

Based on: docs/plans/2025-02-05-spoon-bot-api-design.md
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine

from fastapi import WebSocket, WebSocketDisconnect, Query
from loguru import logger

from spoon_bot.gateway.app import get_agent, get_connection_manager, get_config
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
from spoon_bot.gateway.websocket.agent import (
    WSDialogueAgent,
    get_agent_manager,
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

    if token:
        token_data = verify_token(token, config.jwt.secret_key, expected_type="access")
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
    """Handles WebSocket message processing."""

    def __init__(self, connection_id: str, session_id: str | None = None):
        self.connection_id = connection_id
        self.session_id = session_id or connection_id
        self._dialogue_agent: WSDialogueAgent | None = None
        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            # Legacy methods (for backward compatibility)
            "agent.chat": self._handle_chat,
            "agent.cancel": self._handle_cancel,
            "agent.status": self._handle_status,
            # New methods based on API design
            ClientMethod.CHAT_SEND.value: self._handle_chat_send,
            ClientMethod.CHAT_CANCEL.value: self._handle_chat_cancel,
            ClientMethod.CONFIRM_RESPOND.value: self._handle_confirm_respond,
            ClientMethod.AGENT_STATUS.value: self._handle_agent_status,
            # Session methods
            "session.switch": self._handle_session_switch,
            "session.list": self._handle_session_list,
            "session.clear": self._handle_session_clear,
            ClientMethod.SESSION_EXPORT.value: self._handle_session_export,
            ClientMethod.SESSION_IMPORT.value: self._handle_session_import,
            # Subscription methods
            "subscribe": self._handle_subscribe,
            "unsubscribe": self._handle_unsubscribe,
        }

    async def handle_request(self, request: WSRequest) -> WSResponse | WSError:
        """
        Route and handle a request.

        Args:
            request: Incoming WebSocket request.

        Returns:
            Response or error.
        """
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

    async def _handle_chat(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle chat request."""
        agent = get_agent()
        manager = get_connection_manager()
        conn = manager.get_connection(self.connection_id)

        if not conn:
            raise ValueError("Connection not found")

        message = params.get("message", "")
        session_key = params.get("session_key", conn.session_key)
        stream = params.get("stream", False)

        # Emit thinking event
        await manager.send_message(
            self.connection_id,
            WSEvent(event="agent.thinking", data={"status": "processing"}),
        )

        # Process with agent
        response = await agent.process(
            message=message,
            session_key=session_key,
        )

        # Emit complete event
        await manager.send_message(
            self.connection_id,
            WSEvent(event="agent.complete", data={"status": "done"}),
        )

        return {"content": response, "session_key": session_key}

    async def _handle_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle cancel request."""
        # TODO: Implement cancellation
        return {"cancelled": True}

    async def _handle_status(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle status request."""
        agent = get_agent()

        return {
            "status": "ready",
            "tools": len(agent.tools.list_tools()),
            "skills": len(agent.skills.list()),
        }

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
        sessions = agent.sessions.list_sessions()

        return {
            "sessions": [
                {"key": s.session_key, "message_count": len(s.messages)}
                for s in sessions
            ]
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

    # ========== New methods based on API design ==========

    async def _get_dialogue_agent(self) -> WSDialogueAgent:
        """Get or create the dialogue agent for this session."""
        if self._dialogue_agent is None:
            manager = get_connection_manager()

            # Create event callback that sends events via WebSocket
            async def event_callback(event: str, data: dict[str, Any]) -> None:
                await manager.send_message(
                    self.connection_id,
                    WSEvent(event=event, data=data),
                )

            agent_manager = get_agent_manager()
            try:
                self._dialogue_agent = await agent_manager.create_agent(
                    session_id=self.session_id,
                    event_callback=event_callback,
                )
            except ValueError:
                # Agent already exists, get it
                self._dialogue_agent = agent_manager.get_agent(self.session_id)

        return self._dialogue_agent

    async def _handle_chat_send(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Handle chat.send request - send a message and stream response.

        Params:
            message: str - User message
            stream: bool - Whether to stream (default: True)
            context: dict - Additional context (optional)
        """
        agent = await self._get_dialogue_agent()
        message = params.get("message", "")
        stream = params.get("stream", True)
        context = params.get("context")

        if not message:
            raise ValueError("Message is required")

        result = await agent.chat(message, stream=stream, context=context)
        return result

    async def _handle_chat_cancel(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle chat.cancel request - cancel current execution."""
        agent = await self._get_dialogue_agent()
        return await agent.cancel()

    async def _handle_confirm_respond(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Handle confirm.respond request - respond to a confirmation request.

        Params:
            request_id: str - The confirmation request ID
            approved: bool - Whether to approve or deny
            reason: str - Optional reason for the decision
        """
        agent = await self._get_dialogue_agent()

        request_id = params.get("request_id")
        approved = params.get("approved", False)
        reason = params.get("reason")

        if not request_id:
            raise ValueError("request_id is required")

        return await agent.respond_confirmation(request_id, approved, reason)

    async def _handle_agent_status(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle agent.status request - get agent status."""
        agent = await self._get_dialogue_agent()
        return await agent.get_status()

    async def _handle_session_export(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Handle session.export request - export session state.

        Params:
            layers: list[str] - Layers to export (core, runtime, workspace, metadata)
            format: str - Export format (json, msgpack, protobuf)
        """
        agent = await self._get_dialogue_agent()
        state = await agent.export_state()

        # Filter by requested layers if specified
        layers = params.get("layers")
        if layers:
            filtered_state = {"version": state["version"], "exported_at": state["exported_at"]}
            for layer in layers:
                if layer in state:
                    filtered_state[layer] = state[layer]
            state = filtered_state

        return {"success": True, "state": state}

    async def _handle_session_import(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Handle session.import request - import session state.

        Params:
            state: dict - State to import
            options: dict - Import options
        """
        agent = await self._get_dialogue_agent()

        state = params.get("state")
        if not state:
            raise ValueError("state is required")

        await agent.import_state(state)

        return {
            "success": True,
            "restored": {
                "session_id": self.session_id,
            }
        }
