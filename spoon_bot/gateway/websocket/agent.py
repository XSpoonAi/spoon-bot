"""
WebSocket Agent for dialogue transmission.

This module implements the WebSocket-based dialogue agent that handles:
- Real-time streaming of agent responses
- User confirmation flow for dangerous operations
- Event-based communication (agent.thinking, agent.streaming, etc.)
- Session state management
- Tool call notifications

Based on the API design: docs/plans/2025-02-05-spoon-bot-api-design.md
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine
from uuid import uuid4

from loguru import logger

from spoon_bot.core import SpoonBot, SpoonBotConfig


class AgentState(str, Enum):
    """Agent execution state."""

    IDLE = "idle"
    THINKING = "thinking"
    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    WAITING_CONFIRM = "waiting_confirm"
    COMPLETE = "complete"
    ERROR = "error"


class ToolPermission(str, Enum):
    """Tool permission levels."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass
class ConfirmRequest:
    """Pending confirmation request."""

    request_id: str
    action: str
    description: str
    tool_name: str
    arguments: dict[str, Any]
    risk_level: str  # low | medium | high
    timeout_seconds: int = 300
    fallback_action: str = "deny"  # allow | deny
    created_at: datetime = field(default_factory=datetime.utcnow)
    future: asyncio.Future | None = None

    def is_expired(self) -> bool:
        """Check if request has expired."""
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.timeout_seconds


@dataclass
class ToolCallInfo:
    """Tool call information for notifications."""

    tool: str
    arguments: dict[str, Any]
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    result: Any = None
    success: bool = True
    error: str | None = None

    @property
    def duration_ms(self) -> int:
        """Get duration in milliseconds."""
        if self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0


@dataclass
class AgentMetrics:
    """Agent execution metrics."""

    task_id: str
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: datetime | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    files_changed: list[dict[str, str]] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def duration_ms(self) -> int:
        """Get total duration in milliseconds."""
        end = self.end_time or datetime.utcnow()
        return int((end - self.start_time).total_seconds() * 1000)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "duration_ms": self.duration_ms,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "tool_calls": [
                {"tool": tc.tool, "count": 1} for tc in self.tool_calls
            ],
            "files_changed": self.files_changed,
        }


class WSDialogueAgent:
    """
    WebSocket Dialogue Agent for real-time conversation.

    Handles:
    - Streaming agent responses via WebSocket
    - User confirmation requests for dangerous operations
    - Event emission for UI updates
    - Session and state management
    """

    def __init__(
        self,
        session_id: str,
        bot: SpoonBot | None = None,
        config: SpoonBotConfig | None = None,
        event_callback: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
    ):
        """
        Initialize dialogue agent.

        Args:
            session_id: Unique session identifier.
            bot: Pre-initialized SpoonBot instance.
            config: Bot configuration (used if bot is None).
            event_callback: Async callback for event emission.
        """
        self.session_id = session_id
        self._bot = bot
        self._config = config or SpoonBotConfig.from_env()
        self._event_callback = event_callback

        self._state = AgentState.IDLE
        self._current_task_id: str | None = None
        self._metrics: AgentMetrics | None = None
        self._pending_confirms: dict[str, ConfirmRequest] = {}
        self._tool_permissions: dict[str, ToolPermission] = {}
        self._cancel_requested = False
        self._initialized = False

        # Default tool permissions (can be configured)
        self._default_permissions = {
            "shell_execute": ToolPermission.CONFIRM,
            "file_delete": ToolPermission.CONFIRM,
            "file_write": ToolPermission.ALLOW,
            "file_read": ToolPermission.ALLOW,
            "web_request": ToolPermission.ALLOW,
            "mcp.*": ToolPermission.ALLOW,
        }

    async def initialize(self) -> None:
        """Initialize the agent."""
        if self._initialized:
            return

        if not self._bot:
            self._bot = SpoonBot(self._config)

        await self._bot.initialize()
        self._initialized = True

        logger.info(f"WSDialogueAgent initialized: session={self.session_id}")

    async def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        """Emit an event to the WebSocket client."""
        if self._event_callback:
            try:
                await self._event_callback(event, data)
            except Exception as e:
                logger.error(f"Failed to emit event {event}: {e}")

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state

    @property
    def is_busy(self) -> bool:
        """Check if agent is busy."""
        return self._state not in (AgentState.IDLE, AgentState.COMPLETE, AgentState.ERROR)

    async def chat(
        self,
        message: str,
        stream: bool = True,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process a chat message.

        Args:
            message: User message.
            stream: Whether to stream responses.
            context: Additional context.

        Returns:
            Response dictionary with content, usage, etc.
        """
        if not self._initialized:
            await self.initialize()

        if self.is_busy:
            raise RuntimeError("Agent is busy processing another request")

        self._cancel_requested = False
        self._current_task_id = f"task_{uuid4().hex[:8]}"
        self._metrics = AgentMetrics(task_id=self._current_task_id)

        try:
            # Emit thinking event
            self._state = AgentState.THINKING
            await self._emit_event("agent.thinking", {
                "task_id": self._current_task_id,
                "status": "processing",
            })

            accumulated_response = ""

            if stream:
                # Stream the response
                self._state = AgentState.STREAMING
                async for chunk in self._bot.stream(message):
                    if self._cancel_requested:
                        break

                    accumulated_response += chunk
                    await self._emit_event("agent.streaming", {
                        "task_id": self._current_task_id,
                        "delta": chunk,
                        "accumulated": accumulated_response,
                    })
            else:
                # Get full response
                accumulated_response = await self._bot.chat(message)

            # Complete
            self._state = AgentState.COMPLETE
            self._metrics.end_time = datetime.utcnow()

            result = {
                "success": True,
                "task_id": self._current_task_id,
                "response": accumulated_response,
                "metrics": self._metrics.to_dict(),
            }

            await self._emit_event("agent.complete", result)
            return result

        except Exception as e:
            self._state = AgentState.ERROR
            error_data = {
                "task_id": self._current_task_id,
                "code": "EXECUTION_ERROR",
                "message": str(e),
            }
            await self._emit_event("agent.error", error_data)
            raise

        finally:
            self._state = AgentState.IDLE

    async def cancel(self) -> dict[str, Any]:
        """Cancel current execution."""
        if not self.is_busy:
            return {"cancelled": False, "reason": "No active task"}

        self._cancel_requested = True
        await self._emit_event("agent.cancelled", {
            "task_id": self._current_task_id,
        })

        return {"cancelled": True, "task_id": self._current_task_id}

    async def request_confirmation(
        self,
        action: str,
        description: str,
        tool_name: str,
        arguments: dict[str, Any],
        risk_level: str = "medium",
        timeout_seconds: int = 300,
    ) -> bool:
        """
        Request user confirmation for an action.

        Args:
            action: Action type (e.g., "shell_execute").
            description: Human-readable description.
            tool_name: Tool name.
            arguments: Tool arguments.
            risk_level: Risk level (low/medium/high).
            timeout_seconds: Timeout for user response.

        Returns:
            True if approved, False if denied.
        """
        request_id = f"cfm_{uuid4().hex[:8]}"
        loop = asyncio.get_event_loop()
        future: asyncio.Future[bool] = loop.create_future()

        request = ConfirmRequest(
            request_id=request_id,
            action=action,
            description=description,
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk_level,
            timeout_seconds=timeout_seconds,
            future=future,
        )

        self._pending_confirms[request_id] = request
        self._state = AgentState.WAITING_CONFIRM

        # Emit confirmation request event
        await self._emit_event("confirm.request", {
            "request_id": request_id,
            "action": action,
            "description": description,
            "tool_name": tool_name,
            "risk_level": risk_level,
            "timeout_seconds": timeout_seconds,
            "fallback_action": request.fallback_action,
        })

        try:
            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError:
            # Handle timeout
            await self._emit_event("confirm.timeout", {
                "request_id": request_id,
                "fallback_action": request.fallback_action,
            })
            return request.fallback_action == "allow"
        finally:
            self._pending_confirms.pop(request_id, None)
            if self._state == AgentState.WAITING_CONFIRM:
                self._state = AgentState.STREAMING

    async def respond_confirmation(
        self,
        request_id: str,
        approved: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        """
        Respond to a confirmation request.

        Args:
            request_id: Confirmation request ID.
            approved: Whether approved or denied.
            reason: Optional reason for the decision.

        Returns:
            Response status.
        """
        request = self._pending_confirms.get(request_id)
        if not request:
            return {
                "success": False,
                "error": "Request not found or expired",
            }

        if request.is_expired():
            self._pending_confirms.pop(request_id, None)
            return {
                "success": False,
                "error": "Request has expired",
            }

        # Resolve the future
        if request.future and not request.future.done():
            request.future.set_result(approved)

        await self._emit_event("confirm.response", {
            "request_id": request_id,
            "approved": approved,
            "reason": reason,
        })

        return {
            "success": True,
            "request_id": request_id,
            "approved": approved,
        }

    def set_tool_permissions(self, permissions: dict[str, str]) -> None:
        """
        Set tool permissions.

        Args:
            permissions: Dict of tool_name -> permission (allow/confirm/deny).
        """
        for tool, perm in permissions.items():
            try:
                self._tool_permissions[tool] = ToolPermission(perm)
            except ValueError:
                logger.warning(f"Invalid permission '{perm}' for tool '{tool}'")

    def get_tool_permission(self, tool_name: str) -> ToolPermission:
        """Get permission for a tool."""
        # Check exact match first
        if tool_name in self._tool_permissions:
            return self._tool_permissions[tool_name]

        # Check wildcard patterns
        for pattern, perm in self._tool_permissions.items():
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if tool_name.startswith(prefix):
                    return perm

        # Check defaults
        if tool_name in self._default_permissions:
            return self._default_permissions[tool_name]

        for pattern, perm in self._default_permissions.items():
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if tool_name.startswith(prefix):
                    return perm

        return ToolPermission.ALLOW

    async def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        status = {
            "session_id": self.session_id,
            "state": self._state.value,
            "initialized": self._initialized,
            "current_task_id": self._current_task_id,
            "pending_confirmations": len(self._pending_confirms),
        }

        if self._bot:
            status.update(self._bot.get_status())

        if self._metrics:
            status["metrics"] = self._metrics.to_dict()

        return status

    async def export_state(self) -> dict[str, Any]:
        """
        Export agent state for persistence.

        Returns:
            State dictionary that can be imported later.
        """
        return {
            "version": "1.0",
            "exported_at": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "core": {
                "config": {
                    "model": self._config.model,
                    "provider": self._config.provider,
                    "max_steps": self._config.max_steps,
                    "system_prompt": self._config.system_prompt,
                },
            },
            "runtime": {
                "tool_permissions": {
                    k: v.value for k, v in self._tool_permissions.items()
                },
                "mcp_servers": dict(self._config.mcp_servers),
                "skills_enabled": self._config.enable_skills,
            },
            "metadata": {
                "state": self._state.value,
                "initialized": self._initialized,
            },
        }

    async def import_state(self, state: dict[str, Any]) -> None:
        """
        Import agent state from a previous export.

        Args:
            state: State dictionary from export_state().
        """
        if state.get("version") != "1.0":
            raise ValueError(f"Unsupported state version: {state.get('version')}")

        # Restore configuration
        core = state.get("core", {})
        config = core.get("config", {})

        self._config = SpoonBotConfig(
            model=config.get("model", self._config.model),
            provider=config.get("provider", self._config.provider),
            max_steps=config.get("max_steps", self._config.max_steps),
            system_prompt=config.get("system_prompt"),
        )

        # Restore runtime state
        runtime = state.get("runtime", {})

        permissions = runtime.get("tool_permissions", {})
        self.set_tool_permissions(permissions)

        mcp_servers = runtime.get("mcp_servers", {})
        self._config.mcp_servers = mcp_servers

        self._config.enable_skills = runtime.get("skills_enabled", True)

        # Reinitialize
        self._initialized = False
        self._bot = None
        await self.initialize()

        logger.info(f"WSDialogueAgent state imported: session={self.session_id}")

    async def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        if self.is_busy:
            await self.cancel()

        # Cancel pending confirmations
        for request_id, request in list(self._pending_confirms.items()):
            if request.future and not request.future.done():
                request.future.set_result(False)

        self._pending_confirms.clear()
        self._state = AgentState.IDLE
        self._initialized = False

        logger.info(f"WSDialogueAgent shutdown: session={self.session_id}")


class WSDialogueAgentManager:
    """
    Manager for multiple WebSocket dialogue agents.

    Handles:
    - Agent lifecycle (create, get, destroy)
    - Session management
    - Resource cleanup
    """

    def __init__(self, default_config: SpoonBotConfig | None = None):
        """
        Initialize agent manager.

        Args:
            default_config: Default configuration for new agents.
        """
        self._agents: dict[str, WSDialogueAgent] = {}
        self._default_config = default_config or SpoonBotConfig.from_env()
        self._lock = asyncio.Lock()

    async def create_agent(
        self,
        session_id: str,
        config: SpoonBotConfig | None = None,
        event_callback: Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
    ) -> WSDialogueAgent:
        """
        Create a new dialogue agent.

        Args:
            session_id: Unique session identifier.
            config: Agent configuration.
            event_callback: Event callback function.

        Returns:
            New WSDialogueAgent instance.
        """
        async with self._lock:
            if session_id in self._agents:
                raise ValueError(f"Agent already exists for session: {session_id}")

            agent = WSDialogueAgent(
                session_id=session_id,
                config=config or self._default_config,
                event_callback=event_callback,
            )

            await agent.initialize()
            self._agents[session_id] = agent

            logger.info(f"Created agent for session: {session_id}")
            return agent

    def get_agent(self, session_id: str) -> WSDialogueAgent | None:
        """Get an existing agent by session ID."""
        return self._agents.get(session_id)

    async def destroy_agent(self, session_id: str) -> bool:
        """
        Destroy an agent and cleanup resources.

        Args:
            session_id: Session identifier.

        Returns:
            True if agent was destroyed.
        """
        async with self._lock:
            agent = self._agents.pop(session_id, None)
            if agent:
                await agent.shutdown()
                logger.info(f"Destroyed agent for session: {session_id}")
                return True
            return False

    async def shutdown_all(self) -> None:
        """Shutdown all agents."""
        async with self._lock:
            for session_id in list(self._agents.keys()):
                agent = self._agents.pop(session_id, None)
                if agent:
                    await agent.shutdown()

            logger.info("All dialogue agents shutdown")

    @property
    def active_sessions(self) -> list[str]:
        """Get list of active session IDs."""
        return list(self._agents.keys())

    @property
    def agent_count(self) -> int:
        """Get number of active agents."""
        return len(self._agents)


# Global agent manager instance
_agent_manager: WSDialogueAgentManager | None = None


def get_agent_manager() -> WSDialogueAgentManager:
    """Get the global agent manager instance."""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = WSDialogueAgentManager()
    return _agent_manager


async def init_agent_manager(config: SpoonBotConfig | None = None) -> WSDialogueAgentManager:
    """Initialize the global agent manager."""
    global _agent_manager
    _agent_manager = WSDialogueAgentManager(default_config=config)
    return _agent_manager
