"""WebSocket module for real-time communication.

This module provides WebSocket-based dialogue transmission for spoon-bot,
including:
- Real-time streaming of agent responses
- User confirmation flow for dangerous operations
- Event-based communication
- Session state management

Based on: docs/plans/2025-02-05-spoon-bot-api-design.md
"""

from spoon_bot.gateway.websocket.manager import ConnectionManager
from spoon_bot.gateway.websocket.protocol import (
    MessageType,
    ClientMethod,
    ServerEvent,
    WSMessage,
    WSRequest,
    WSResponse,
    WSEvent,
    WSError,
    WSStreamChunk,
)
from spoon_bot.gateway.websocket.agent import (
    AgentState,
    ToolPermission,
    ConfirmRequest,
    AgentMetrics,
    WSDialogueAgent,
    WSDialogueAgentManager,
    get_agent_manager,
    init_agent_manager,
)

__all__ = [
    # Connection management
    "ConnectionManager",
    # Protocol types
    "MessageType",
    "ClientMethod",
    "ServerEvent",
    "WSMessage",
    "WSRequest",
    "WSResponse",
    "WSEvent",
    "WSError",
    "WSStreamChunk",
    # Dialogue agent
    "AgentState",
    "ToolPermission",
    "ConfirmRequest",
    "AgentMetrics",
    "WSDialogueAgent",
    "WSDialogueAgentManager",
    "get_agent_manager",
    "init_agent_manager",
]
