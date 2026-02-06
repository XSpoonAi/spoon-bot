"""
Standalone gateway server entry point for Docker / uvicorn deployment.

This module provides a create_app factory that:
1. Loads configuration from environment variables
2. Creates and initializes the agent during app lifespan
3. Returns a fully configured FastAPI application

Usage with uvicorn:
    uvicorn spoon_bot.gateway.server:create_app --factory --host 0.0.0.0 --port 8080

Environment variables:
    See .env.example for all available configuration options.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

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
from spoon_bot.gateway import app as app_module


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Application lifespan handler with auto agent initialization."""

    logger.info("Starting spoon-bot gateway (Docker mode)...")
    logger.info(f"spoon-core SDK available: {is_spoon_core_available()}")

    # Initialize connection manager
    connection_manager = ConnectionManager()
    await connection_manager.start()
    app_module._connection_manager = connection_manager

    # Auto-create agent from environment variables
    provider = os.environ.get("SPOON_BOT_DEFAULT_PROVIDER", "anthropic")
    model = os.environ.get("SPOON_BOT_DEFAULT_MODEL", "")
    base_url = os.environ.get("BASE_URL", "")

    # Determine default model per provider if not specified
    if not model:
        default_models = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "deepseek": "deepseek-chat",
            "gemini": "gemini-2.0-flash",
            "openrouter": "anthropic/claude-sonnet-4",
        }
        model = default_models.get(provider, "claude-sonnet-4-20250514")

    # Log base URL if custom proxy is set
    anthropic_base = os.environ.get("ANTHROPIC_BASE_URL", "")
    if anthropic_base:
        logger.info(f"Custom Anthropic base URL: {anthropic_base}")

    logger.info(f"Initializing agent: provider={provider}, model={model}")

    try:
        agent = SpoonCoreAgent(
            model=model,
            provider=provider,
            enable_skills=True,
        )
        await agent.initialize()
        app_module._agent = agent
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        logger.warning("Gateway will start without agent - set valid API keys to enable")

    # Initialize spoon-core services
    identity = SpoonCoreIdentity()
    payments = SpoonCorePayments()
    app_module._identity = identity
    app_module._payments = payments

    if identity.available:
        logger.info("ERC8004 identity service enabled")
    if payments.available:
        logger.info("X402 payment service enabled")

    yield

    # Shutdown
    logger.info("Shutting down spoon-bot gateway...")
    if connection_manager:
        await connection_manager.stop()


def create_app() -> FastAPI:
    """
    Create the FastAPI application with auto agent initialization.

    This factory function is designed for uvicorn --factory usage in Docker.
    It reads all configuration from environment variables.

    Returns:
        Configured FastAPI application with agent auto-initialization.
    """
    config = GatewayConfig.from_env()
    app_module._config = config

    app = FastAPI(
        title="spoon-bot Gateway",
        description="WebSocket and REST API for spoon-bot agent control",
        version="1.0.0",
        docs_url=config.docs_url,
        redoc_url=config.redoc_url,
        lifespan=_lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allow_origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allow_methods,
        allow_headers=config.cors.allow_headers,
        max_age=config.cors.max_age,
    )

    # Register routes
    from spoon_bot.gateway.api.health import router as health_router
    from spoon_bot.gateway.api.v1.router import router as v1_router
    from spoon_bot.gateway.websocket.handler import websocket_endpoint

    app.include_router(health_router)
    app.include_router(v1_router, prefix=config.api_prefix)
    app.add_api_websocket_route(
        f"{config.api_prefix}/ws",
        websocket_endpoint,
        name="websocket",
    )

    logger.info(
        f"Gateway configured: host={config.host}, port={config.port}, "
        f"debug={config.debug}"
    )

    return app
