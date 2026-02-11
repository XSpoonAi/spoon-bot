"""
spoon-bot Gateway Module

Provides WebSocket and REST API for remote agent control.
Integrates with spoon-core SDK for maximum component reuse.

Usage with spoon-core (recommended):
    from spoon_bot.gateway import create_app, GatewayConfig, create_spoon_core_agent
    from spoon_bot.gateway.app import set_agent

    # Create agent using spoon-core SDK
    agent = await create_spoon_core_agent(
        model="claude-sonnet-4-20250514",
        provider="anthropic",
        mcp_servers={
            "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]}
        },
        enable_skills=True,
    )

    # Create gateway app
    config = GatewayConfig(host="0.0.0.0", port=8080)
    app = create_app(config)
    set_agent(agent)

    # Run with uvicorn
    uvicorn.run(app, host=config.host, port=config.port)

Fallback (without spoon-core):
    from spoon_bot.gateway import create_app, GatewayConfig
    from spoon_bot.agent.loop import create_agent
    from spoon_bot.gateway.app import set_agent

    agent = await create_agent(provider="anthropic")
    app = create_app(GatewayConfig())
    set_agent(agent)
"""

from spoon_bot.gateway.config import GatewayConfig
from spoon_bot.gateway.app import create_app, create_spoon_core_agent, set_agent
from spoon_bot.gateway.core_integration import (
    SpoonCoreAgent,
    SpoonCoreIdentity,
    SpoonCorePayments,
    is_spoon_core_available,
    get_available_modules,
)

__all__ = [
    "create_app",
    "create_spoon_core_agent",
    "set_agent",
    "GatewayConfig",
    "SpoonCoreAgent",
    "SpoonCoreIdentity",
    "SpoonCorePayments",
    "is_spoon_core_available",
    "get_available_modules",
]
