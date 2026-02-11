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
    """
    uptime = int(time.time() - _start_time)

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        uptime=uptime,
        checks=[
            HealthCheck(name="gateway", status="healthy"),
        ],
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Readiness check endpoint.

    Checks if the service is ready to accept requests.
    """
    from spoon_bot.gateway.app import _agent

    checks = {
        "gateway": True,
        "agent": _agent is not None,
    }

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
