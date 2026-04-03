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
    GATEWAY_AUTH_REQUIRED: Set to "false" to disable authentication (default: true)
    GATEWAY_API_KEY: Set a gateway API key for docker access
    OPENAI_API_KEY / ANTHROPIC_API_KEY: LLM provider API keys
    OPENAI_BASE_URL / ANTHROPIC_BASE_URL / BASE_URL: Custom LLM base URLs
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
    SpoonCoreIdentity,
    SpoonCorePayments,
    is_spoon_core_available,
)
from spoon_bot.gateway import app as app_module
from spoon_bot.agent.tools.web import close_shared_http_client


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Application lifespan handler with auto agent initialization."""
    from dotenv import load_dotenv

    load_dotenv(override=False)

    logger.info("Starting spoon-bot gateway (Docker mode)...")
    logger.info(f"spoon-core SDK available: {is_spoon_core_available()}")

    # Initialize connection manager
    connection_manager = ConnectionManager()
    await connection_manager.start()
    app_module._connection_manager = connection_manager

    # Auto-create agent.
    # All config resolution (YAML > env vars) is handled by load_agent_config().
    from spoon_bot.channels.config import load_agent_config

    try:
        agent_cfg = load_agent_config()
    except Exception as _cfg_err:
        logger.warning(f"Could not load agent config: {_cfg_err}")
        agent_cfg = {}

    provider = agent_cfg.get("provider")
    model = agent_cfg.get("model")
    if not provider or not model:
        logger.error(
            "Missing required config: model and provider must be set "
            "in config.yaml or via SPOON_BOT_DEFAULT_MODEL / SPOON_BOT_DEFAULT_PROVIDER env vars"
        )

    base_url = agent_cfg.get("base_url")
    if base_url:
        logger.info(f"Custom base URL: {base_url}")

    logger.info(f"Initializing agent: provider={provider}, model={model}")

    try:
        from spoon_bot.agent.loop import create_agent

        enable_skills = agent_cfg.get("enable_skills", True)
        logger.info(f"Skills enabled: {enable_skills}")

        # Session persistence config from env
        session_store_backend = os.environ.get("SESSION_STORE_BACKEND", "file")
        session_store_dsn = os.environ.get("SESSION_STORE_DSN")
        session_store_db_path = os.environ.get("SESSION_STORE_DB_PATH")

        # Context window override (optional)
        _ctx_env = os.environ.get("CONTEXT_WINDOW")
        context_window = int(_ctx_env) if _ctx_env else None

        # YOLO mode: operate directly in user's path without sandbox
        yolo_mode = (
            agent_cfg.get("yolo_mode")
            or os.environ.get("SPOON_BOT_YOLO_MODE", "").lower() in ("1", "true", "yes")
        )
        if yolo_mode:
            logger.info("YOLO mode enabled — agent will work directly in user path")

        create_kwargs: dict = dict(
            model=model,
            provider=provider,
            api_key=agent_cfg.get("api_key"),
            base_url=base_url or None,
            workspace=agent_cfg.get("workspace", "/data/workspace"),
            enable_skills=enable_skills,
            auto_commit=False,  # No git auto-commit in Docker gateway mode
            session_store_backend=session_store_backend,
            session_store_dsn=session_store_dsn,
            session_store_db_path=session_store_db_path,
            context_window=context_window,
            yolo_mode=bool(yolo_mode),
        )
        if agent_cfg.get("mcp_config") is not None:
            create_kwargs["mcp_config"] = agent_cfg["mcp_config"]
        if agent_cfg.get("shell_timeout") is not None:
            create_kwargs["shell_timeout"] = int(agent_cfg["shell_timeout"])
        if agent_cfg.get("max_output") is not None:
            create_kwargs["max_output"] = int(agent_cfg["max_output"])
        if agent_cfg.get("enabled_tools") is not None:
            create_kwargs["enabled_tools"] = set(agent_cfg["enabled_tools"])
        if agent_cfg.get("session_store_backend") is not None:
            create_kwargs["session_store_backend"] = agent_cfg["session_store_backend"]
        if agent_cfg.get("session_store_dsn") is not None:
            create_kwargs["session_store_dsn"] = agent_cfg["session_store_dsn"]
        if agent_cfg.get("session_store_db_path") is not None:
            create_kwargs["session_store_db_path"] = agent_cfg["session_store_db_path"]
        if agent_cfg.get("context_window") is not None:
            create_kwargs["context_window"] = int(agent_cfg["context_window"])
        if agent_cfg.get("tool_profile"):
            create_kwargs["tool_profile"] = agent_cfg["tool_profile"]
        if agent_cfg.get("max_iterations"):
            create_kwargs["max_iterations"] = int(agent_cfg["max_iterations"])

        agent = await create_agent(**create_kwargs)
        app_module._agent = agent

        # Log tool/skill counts
        tool_count = len(agent.tools) if hasattr(agent.tools, '__len__') else len(agent.tools.list_tools()) if hasattr(agent.tools, 'list_tools') else 0
        skill_count = len(agent.skills) if hasattr(agent, 'skills') and agent.skills else 0
        logger.info(f"Agent initialized: tools={tool_count}, skills={skill_count}")
    except Exception as e:
        import traceback
        logger.error(f"Failed to initialize agent: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        logger.warning("Gateway will start without agent - set valid API keys to enable")

    # Initialize spoon-core services (optional)
    try:
        identity = SpoonCoreIdentity()
        payments = SpoonCorePayments()
        app_module._identity = identity
        app_module._payments = payments

        if identity.available:
            logger.info("ERC8004 identity service enabled")
        if payments.available:
            logger.info("X402 payment service enabled")
    except Exception as e:
        logger.debug(f"Optional spoon-core services not available: {e}")

    # ------------------------------------------------------------------
    # Start channels (Telegram / Discord / Feishu) if configured.
    # Channels run as asyncio tasks inside ChannelManager so they share
    # the same event loop with the FastAPI server.
    # ------------------------------------------------------------------
    channel_manager = None
    if app_module._agent is not None:
        try:
            from spoon_bot.bootstrap import init_channels

            channel_manager = await init_channels(app_module._agent)
            app_module._channel_manager = channel_manager
            logger.info(
                f"Channels loaded: {channel_manager.running_channels_count} running"
            )
        except FileNotFoundError:
            logger.info("No channel config found, running gateway without channels")
        except ImportError as e:
            logger.warning(f"Channel dependencies missing: {e}")
        except Exception as e:
            logger.warning(f"Could not start channels: {e}")

    # ---- Startup summary ----
    logger.info("===== spoon-bot Docker gateway ready =====")
    logger.info(f"  Provider : {agent_cfg.get('provider', '(not set)')}")
    logger.info(f"  Model    : {agent_cfg.get('model', '(not set)')}")
    logger.info(f"  Workspace: {agent_cfg.get('workspace', '/data/workspace')}")
    logger.info(f"  YOLO mode: {bool(yolo_mode)}")
    if channel_manager:
        logger.info(
            f"  Channels : {channel_manager.running_channels_count} running "
            f"({', '.join(channel_manager.channel_names) or 'none'})"
        )
    else:
        logger.info("  Channels : none")
    logger.info("==========================================")

    yield

    # Shutdown
    logger.info("Shutting down spoon-bot gateway...")
    if channel_manager:
        await channel_manager.stop()
        logger.info("Channels stopped")
    if connection_manager:
        await connection_manager.stop()
    await close_shared_http_client()


def create_app() -> FastAPI:
    """
    Create the FastAPI application with auto agent initialization.

    This factory function is designed for uvicorn --factory usage in Docker.
    It reads all configuration from environment variables.

    Returns:
        Configured FastAPI application with agent auto-initialization.
    """
    config = GatewayConfig.from_env()

    # Configure API key from environment if provided
    gateway_api_key = os.environ.get("GATEWAY_API_KEY", "")
    if gateway_api_key:
        config.api_keys[gateway_api_key] = "docker-user"
        logger.info("Gateway API key configured from GATEWAY_API_KEY env var")

    # Configure auth requirement
    auth_required = os.environ.get("GATEWAY_AUTH_REQUIRED", "true").lower()
    if auth_required == "false" or (not gateway_api_key and auth_required != "true"):
        app_module._auth_required = False
        logger.info("Authentication DISABLED - all endpoints are public")
    else:
        app_module._auth_required = True
        logger.info("Authentication ENABLED")

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
    from spoon_bot.gateway.api.webhooks import router as webhook_router
    from spoon_bot.gateway.api.v1.router import router as v1_router
    from spoon_bot.gateway.websocket.handler import websocket_endpoint

    app.include_router(health_router)
    app.include_router(v1_router, prefix=config.api_prefix)
    app.add_api_websocket_route(
        f"{config.api_prefix}/ws",
        websocket_endpoint,
        name="websocket",
    )
    app.include_router(webhook_router)

    logger.info(
        f"Gateway configured: host={config.host}, port={config.port}, "
        f"debug={config.debug}"
    )

    return app
