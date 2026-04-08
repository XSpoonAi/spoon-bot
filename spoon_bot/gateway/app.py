"""FastAPI application factory for the gateway."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from spoon_bot.gateway.config import GatewayConfig
from spoon_bot.gateway.websocket.manager import ConnectionManager
from spoon_bot.gateway.core_integration import (
    SpoonCoreAgent,
    SpoonCoreIdentity,
    SpoonCorePayments,
    is_spoon_core_available,
)
from spoon_bot.agent.tools.web import close_shared_http_client

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop
    from spoon_bot.channels.manager import ChannelManager


# Global state (set during lifespan)
_agent: SpoonCoreAgent | AgentLoop | None = None
_connection_manager: ConnectionManager | None = None
_channel_manager: ChannelManager | None = None
_config: GatewayConfig | None = None
_identity: SpoonCoreIdentity | None = None
_payments: SpoonCorePayments | None = None
_auth_required: bool = True  # Can be set to False via GATEWAY_AUTH_REQUIRED=false
_agent_execution_lock: asyncio.Lock | None = None
_session_execution_locks: dict[str, asyncio.Lock] = {}
_ws_session_chat_tasks: dict[str, "_WSSessionTaskHandle"] = {}
_ws_session_chat_locks: dict[str, asyncio.Lock] = {}


@dataclass
class _WSSessionTaskHandle:
    task: asyncio.Task
    cancel_cb: Callable[[], None] | None = None
    task_id_cb: Callable[[], str | None] | None = None


def is_auth_required() -> bool:
    """Check if authentication is required."""
    return _auth_required


def get_agent() -> SpoonCoreAgent | AgentLoop:
    """Get the agent instance."""
    if _agent is None:
        raise RuntimeError("Agent not initialized. Call set_agent() first.")
    return _agent


def get_connection_manager() -> ConnectionManager:
    """Get the WebSocket connection manager."""
    if _connection_manager is None:
        raise RuntimeError("Connection manager not initialized.")
    return _connection_manager


def get_config() -> GatewayConfig:
    """Get the gateway configuration."""
    if _config is None:
        raise RuntimeError("Gateway config not initialized.")
    return _config


def get_agent_execution_lock() -> asyncio.Lock:
    """Get the shared lock guarding global agent execution."""
    global _agent_execution_lock
    if _agent_execution_lock is None:
        _agent_execution_lock = asyncio.Lock()
    return _agent_execution_lock


def _normalize_session_key(session_key: str | None) -> str:
    return session_key.strip() if isinstance(session_key, str) and session_key.strip() else "default"


def _normalize_user_id(user_id: str | None) -> str:
    return user_id.strip() if isinstance(user_id, str) and user_id.strip() else "anonymous"


def _ws_session_chat_registry_key(session_key: str | None, user_id: str | None) -> str:
    return f"user:{_normalize_user_id(user_id)}|session:{_normalize_session_key(session_key)}"


def get_session_execution_lock(session_key: str) -> asyncio.Lock:
    """Get/create a lock for a specific session key."""
    key = _normalize_session_key(session_key)
    lock = _session_execution_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _session_execution_locks[key] = lock
    return lock


def register_ws_session_chat_task(
    session_key: str,
    task: asyncio.Task,
    *,
    user_id: str | None = None,
    cancel_cb: Callable[[], None] | None = None,
    task_id_cb: Callable[[], str | None] | None = None,
) -> None:
    """Register the active websocket chat task for a session."""
    _ws_session_chat_tasks[_ws_session_chat_registry_key(session_key, user_id)] = _WSSessionTaskHandle(
        task=task,
        cancel_cb=cancel_cb,
        task_id_cb=task_id_cb,
    )


def get_ws_session_chat_lock(session_key: str, user_id: str | None = None) -> asyncio.Lock:
    """Get/create the serialized replacement lock for a WS user+session key."""
    key = _ws_session_chat_registry_key(session_key, user_id)
    lock = _ws_session_chat_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _ws_session_chat_locks[key] = lock
    return lock


def get_ws_session_chat_task_id(session_key: str, user_id: str | None = None) -> str | None:
    """Get the current websocket chat task id for a session, if available."""
    handle = _ws_session_chat_tasks.get(_ws_session_chat_registry_key(session_key, user_id))
    if handle is None or handle.task_id_cb is None:
        return None
    try:
        return handle.task_id_cb()
    except Exception:
        return None


def has_active_ws_session_chat_task(session_key: str, user_id: str | None = None) -> bool:
    """Whether a session currently has a running websocket chat task."""
    handle = _ws_session_chat_tasks.get(_ws_session_chat_registry_key(session_key, user_id))
    return bool(handle is not None and not handle.task.done())


def clear_ws_session_chat_task(
    session_key: str,
    *,
    user_id: str | None = None,
    task: asyncio.Task | None = None,
) -> bool:
    """Clear the active websocket chat task for a session."""
    key = _ws_session_chat_registry_key(session_key, user_id)
    handle = _ws_session_chat_tasks.get(key)
    if handle is None:
        return False
    if task is not None and handle.task is not task:
        return False
    _ws_session_chat_tasks.pop(key, None)
    return True


async def cancel_ws_session_chat_task(
    session_key: str,
    timeout: float = 2.0,
    *,
    user_id: str | None = None,
) -> bool:
    """Cancel and await an active websocket chat task for a session."""
    key = _ws_session_chat_registry_key(session_key, user_id)
    handle = _ws_session_chat_tasks.get(key)
    if handle is None:
        return False

    task = handle.task
    if task.done():
        clear_ws_session_chat_task(session_key, user_id=user_id, task=task)
        return True

    if handle.cancel_cb is not None:
        try:
            handle.cancel_cb()
        except Exception:
            pass
    task.cancel()
    timed_out = False
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        timed_out = True
        logger.warning(f"Timed out waiting for ws chat task cleanup: session={key}")
    if task.done():
        clear_ws_session_chat_task(session_key, user_id=user_id, task=task)
        return True
    return False if timed_out else task.done()


def set_agent(agent: SpoonCoreAgent | AgentLoop) -> None:
    """Set the agent instance."""
    global _agent
    _agent = agent
    logger.info(f"Agent set for gateway (type: {type(agent).__name__})")


def get_identity() -> SpoonCoreIdentity | None:
    """Get the identity service."""
    return _identity


def get_payments() -> SpoonCorePayments | None:
    """Get the payment service."""
    return _payments


async def create_spoon_core_agent(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    mcp_servers: dict[str, dict] | None = None,
    enable_skills: bool = True,
) -> SpoonCoreAgent:
    """
    Create and initialize a SpoonCoreAgent.

    This is the recommended way to create an agent that maximally
    reuses spoon-core SDK components.

    Args:
        model: Model name to use.
        provider: LLM provider.
        mcp_servers: MCP server configurations.
        enable_skills: Whether to enable skill system.

    Returns:
        Initialized SpoonCoreAgent.
    """
    agent = SpoonCoreAgent(
        model=model,
        provider=provider,
        mcp_servers=mcp_servers,
        enable_skills=enable_skills,
    )
    await agent.initialize()
    return agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global _connection_manager, _identity, _payments

    # Startup
    logger.info("Starting spoon-bot gateway...")
    logger.info(f"spoon-core SDK available: {is_spoon_core_available()}")

    # Initialize connection manager
    _connection_manager = ConnectionManager()
    await _connection_manager.start()

    # Initialize spoon-core services if available
    _identity = SpoonCoreIdentity()
    _payments = SpoonCorePayments()

    if _identity.available:
        logger.info("ERC8004 identity service enabled")
    if _payments.available:
        logger.info("X402 payment service enabled")

    yield

    # Shutdown
    logger.info("Shutting down spoon-bot gateway...")
    if _connection_manager:
        await _connection_manager.stop()
    await close_shared_http_client()


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        config: Gateway configuration. If None, loads from environment.

    Returns:
        Configured FastAPI application.
    """
    global _config, _agent_execution_lock, _session_execution_locks, _ws_session_chat_tasks, _ws_session_chat_locks

    _config = config or GatewayConfig.from_env()
    _agent_execution_lock = None
    _session_execution_locks = {}
    _ws_session_chat_tasks = {}
    _ws_session_chat_locks = {}

    app = FastAPI(
        title="spoon-bot Gateway",
        description="WebSocket and REST API for spoon-bot agent control",
        version="1.0.0",
        docs_url=_config.docs_url,
        redoc_url=_config.redoc_url,
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_config.cors.allow_origins,
        allow_credentials=_config.cors.allow_credentials,
        allow_methods=_config.cors.allow_methods,
        allow_headers=_config.cors.allow_headers,
        max_age=_config.cors.max_age,
    )

    # Register routes
    _register_routes(app, _config)

    return app


def _register_routes(app: FastAPI, config: GatewayConfig) -> None:
    """Register all routes."""
    from spoon_bot.gateway.api.health import router as health_router
    from spoon_bot.gateway.api.v1.router import router as v1_router
    from spoon_bot.gateway.websocket.handler import websocket_endpoint

    # Health endpoints (no prefix)
    app.include_router(health_router)

    # V1 API routes
    app.include_router(v1_router, prefix=config.api_prefix)

    # WebSocket endpoint
    app.add_api_websocket_route(
        f"{config.api_prefix}/ws",
        websocket_endpoint,
        name="websocket",
    )

    logger.info(f"Registered routes with prefix: {config.api_prefix}")
