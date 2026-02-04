"""Agent control endpoints."""

from __future__ import annotations

import time
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from spoon_bot.gateway.app import get_agent
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.models.requests import ChatRequest
from spoon_bot.gateway.models.responses import (
    APIResponse,
    MetaInfo,
    ChatResponse,
    UsageInfo,
    ToolCallInfo,
    StatusResponse,
    AgentStats,
)

router = APIRouter()


@router.post("/chat", response_model=APIResponse[ChatResponse])
async def chat(
    request: ChatRequest,
    user: CurrentUser,
) -> APIResponse[ChatResponse]:
    """
    Send a message to the agent and get a response.

    This is a synchronous endpoint that waits for the agent to complete.
    """
    request_id = f"req_{uuid4().hex[:12]}"
    start_time = time.time()

    try:
        agent = get_agent()

        # Get session key from user token or request
        session_key = request.session_key
        if hasattr(user, "session_key"):
            session_key = user.session_key

        # Process with agent
        response_text = await agent.process(
            message=request.message,
            media=request.media if request.media else None,
            session_key=session_key,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        return APIResponse(
            success=True,
            data=ChatResponse(
                response=response_text,
                tool_calls=[],  # TODO: Extract tool calls from response
                usage=None,  # TODO: Track usage
            ),
            meta=MetaInfo(request_id=request_id, duration_ms=duration_ms),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "AGENT_ERROR", "message": str(e)},
        )


@router.post("/chat/async")
async def chat_async(
    request: ChatRequest,
    user: CurrentUser,
) -> dict:
    """
    Send a message asynchronously.

    Returns a task ID that can be polled for results.
    """
    # TODO: Implement async task queue
    task_id = f"task_{uuid4().hex[:12]}"

    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Async chat not yet implemented",
    }


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    user: CurrentUser,
) -> dict:
    """Get the status of an async task."""
    # TODO: Implement task status lookup
    return {
        "task_id": task_id,
        "status": "not_found",
        "message": "Async tasks not yet implemented",
    }


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    user: CurrentUser,
) -> dict:
    """Cancel an async task."""
    # TODO: Implement task cancellation
    return {"cancelled": False, "message": "Async tasks not yet implemented"}


@router.get("/status", response_model=APIResponse[StatusResponse])
async def get_status(user: CurrentUser) -> APIResponse[StatusResponse]:
    """Get agent status and statistics."""
    request_id = f"req_{uuid4().hex[:12]}"

    try:
        agent = get_agent()

        return APIResponse(
            success=True,
            data=StatusResponse(
                status="ready",
                current_task=None,
                uptime=0,  # TODO: Track uptime
                stats=AgentStats(
                    total_requests=0,
                    active_sessions=len(agent.sessions.list_sessions()),
                    tools_available=len(agent.tools.list_tools()),
                    skills_loaded=len(agent.skills.list()),
                ),
            ),
            meta=MetaInfo(request_id=request_id),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": str(e)},
        )
