"""FastAPI application factory for the gateway."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from spoon_bot.agent.loop import AgentLoop


# Global state (set during lifespan)
_agent: SpoonCoreAgent | AgentLoop | None = None
_connection_manager: ConnectionManager | None = None
_config: GatewayConfig | None = None
_identity: SpoonCoreIdentity | None = None
_payments: SpoonCorePayments | None = None
_auth_required: bool = True  # Can be set to False via GATEWAY_AUTH_REQUIRED=false


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


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        config: Gateway configuration. If None, loads from environment.

    Returns:
        Configured FastAPI application.
    """
    global _config

    _config = config or GatewayConfig.from_env()

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
