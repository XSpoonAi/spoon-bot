"""WebSocket connection manager."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import WebSocket
from loguru import logger

from spoon_bot.gateway.websocket.protocol import WSEvent, WSMessage


@dataclass
class Connection:
    """WebSocket connection info."""

    id: str
    websocket: WebSocket
    user_id: str
    session_key: str
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    subscriptions: set[str] = field(default_factory=set)

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


class ConnectionManager:
    """
    Manages WebSocket connections.

    Handles:
    - Connection lifecycle (connect/disconnect)
    - Message routing to specific users
    - Event broadcasting to subscribers
    - Connection health monitoring
    """

    def __init__(self):
        self._connections: dict[str, Connection] = {}
        self._user_connections: dict[str, set[str]] = {}  # user_id -> connection_ids
        self._running = False
        self._ping_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the connection manager."""
        self._running = True
        self._ping_task = asyncio.create_task(self._ping_loop())
        logger.info("ConnectionManager started")

    async def stop(self) -> None:
        """Stop the connection manager."""
        self._running = False
        if self._ping_task:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for conn in list(self._connections.values()):
            try:
                await conn.websocket.close()
            except Exception:
                pass

        self._connections.clear()
        self._user_connections.clear()
        logger.info("ConnectionManager stopped")

    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        session_key: str = "default",
    ) -> str:
        """
        Accept a WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance.
            user_id: User identifier.
            session_key: Agent session key.

        Returns:
            Connection ID.
        """
        await websocket.accept()

        conn_id = str(uuid4())
        conn = Connection(
            id=conn_id,
            websocket=websocket,
            user_id=user_id,
            session_key=session_key,
        )

        self._connections[conn_id] = conn

        if user_id not in self._user_connections:
            self._user_connections[user_id] = set()
        self._user_connections[user_id].add(conn_id)

        logger.info(f"WebSocket connected: {conn_id} (user: {user_id})")
        return conn_id

    async def disconnect(self, connection_id: str) -> None:
        """
        Close and cleanup a connection.

        Args:
            connection_id: Connection to disconnect.
        """
        conn = self._connections.pop(connection_id, None)
        if not conn:
            return

        # Remove from user connections
        if conn.user_id in self._user_connections:
            self._user_connections[conn.user_id].discard(connection_id)
            if not self._user_connections[conn.user_id]:
                del self._user_connections[conn.user_id]

        try:
            await conn.websocket.close()
        except Exception:
            pass

        logger.info(f"WebSocket disconnected: {connection_id}")

    def get_connection(self, connection_id: str) -> Connection | None:
        """Get a connection by ID."""
        return self._connections.get(connection_id)

    async def send_message(
        self,
        connection_id: str,
        message: WSMessage | dict[str, Any],
    ) -> bool:
        """
        Send a message to a specific connection.

        Args:
            connection_id: Target connection.
            message: Message to send.

        Returns:
            True if sent successfully.
        """
        conn = self._connections.get(connection_id)
        if not conn:
            return False

        try:
            data = message.to_dict() if isinstance(message, WSMessage) else message
            await conn.websocket.send_json(data)
            conn.update_activity()
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            await self.disconnect(connection_id)
            return False

    async def send_to_user(
        self,
        user_id: str,
        message: WSMessage | dict[str, Any],
    ) -> int:
        """
        Send a message to all connections of a user.

        Args:
            user_id: Target user.
            message: Message to send.

        Returns:
            Number of connections message was sent to.
        """
        conn_ids = self._user_connections.get(user_id, set())
        sent = 0
        for conn_id in list(conn_ids):
            if await self.send_message(conn_id, message):
                sent += 1
        return sent

    async def broadcast_event(
        self,
        event: str,
        data: dict[str, Any],
        filter_subscribed: bool = True,
    ) -> int:
        """
        Broadcast an event to connections.

        Args:
            event: Event name.
            data: Event data.
            filter_subscribed: Only send to subscribed connections.

        Returns:
            Number of connections event was sent to.
        """
        message = WSEvent(event=event, data=data)
        sent = 0

        for conn_id, conn in list(self._connections.items()):
            if filter_subscribed and event not in conn.subscriptions:
                continue

            if await self.send_message(conn_id, message):
                sent += 1

        return sent

    def subscribe(self, connection_id: str, events: list[str]) -> None:
        """
        Subscribe a connection to events.

        Args:
            connection_id: Connection to subscribe.
            events: Events to subscribe to.
        """
        conn = self._connections.get(connection_id)
        if conn:
            conn.subscriptions.update(events)
            logger.debug(f"Connection {connection_id} subscribed to: {events}")

    def unsubscribe(self, connection_id: str, events: list[str]) -> None:
        """
        Unsubscribe a connection from events.

        Args:
            connection_id: Connection to unsubscribe.
            events: Events to unsubscribe from.
        """
        conn = self._connections.get(connection_id)
        if conn:
            conn.subscriptions.difference_update(events)
            logger.debug(f"Connection {connection_id} unsubscribed from: {events}")

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connections alive."""
        while self._running:
            await asyncio.sleep(30)  # Ping every 30 seconds

            for conn_id in list(self._connections.keys()):
                try:
                    await self.send_message(
                        conn_id,
                        {"type": "ping", "timestamp": datetime.utcnow().isoformat()},
                    )
                except Exception:
                    pass

    @property
    def connection_count(self) -> int:
        """Get total number of connections."""
        return len(self._connections)

    @property
    def user_count(self) -> int:
        """Get total number of connected users."""
        return len(self._user_connections)
