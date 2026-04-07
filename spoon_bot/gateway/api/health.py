"""Health check endpoints."""

from __future__ import annotations

import time
from datetime import datetime

from fastapi import APIRouter

from spoon_bot.gateway.models.responses import HealthResponse, HealthCheck

router = APIRouter(tags=["health"])

# Track startup time
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns basic health status for load balancers.
    Includes channel health when channels are running.
    """
    from spoon_bot.gateway.app import _agent, _channel_manager

    uptime = int(time.time() - _start_time)

    checks = [
        HealthCheck(name="gateway", status="healthy"),
        HealthCheck(
            name="agent",
            status="healthy" if _agent is not None else "unhealthy",
            message=None if _agent is not None else "Agent not initialized",
        ),
    ]

    if _channel_manager is not None:
        running = _channel_manager.running_channels_count
        total = len(_channel_manager.channel_names)
        ch_status = "healthy" if running == total else ("degraded" if running > 0 else "unhealthy")
        checks.append(HealthCheck(
            name="channels",
            status=ch_status,
            message=f"{running}/{total} running",
        ))

    overall = "healthy"
    if any(c.status == "unhealthy" for c in checks):
        overall = "degraded"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        uptime=uptime,
        checks=checks,
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Readiness check endpoint.

    Checks if the service is ready to accept requests.
    """
    from spoon_bot.gateway.app import _agent, _channel_manager

    checks: dict[str, bool] = {
        "gateway": True,
        "agent": _agent is not None,
    }

    if _channel_manager is not None:
        # Channels are optional for gateway readiness. When none are configured
        # (or all are intentionally disabled), the HTTP/WebSocket gateway should
        # still be considered ready to serve runtime traffic.
        checks["channels"] = True

    all_ready = all(checks.values())

    return {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": "spoon-bot Gateway",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api": "/v1",
    }
