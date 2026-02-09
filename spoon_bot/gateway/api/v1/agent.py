"""Agent control endpoints."""

from __future__ import annotations

import json
import time
from typing import Annotated, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from spoon_bot.gateway.app import get_agent
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.models.requests import ChatRequest
from spoon_bot.gateway.models.responses import (
    APIResponse,
    MetaInfo,
    ChatResponse,
    StreamChunk,
    UsageInfo,
    ToolCallInfo,
    StatusResponse,
    AgentStats,
)

router = APIRouter()


async def _stream_sse(agent, message: str, media: list[str] | None, thinking: bool) -> AsyncGenerator[str, None]:
    """Generate SSE events from agent streaming."""
    kwargs = {"message": message, "thinking": thinking}
    if media:
        kwargs["media"] = media

    try:
        async for chunk_data in agent.stream(**kwargs):
            # Filter out "done" chunks — clients use [DONE] as the completion signal
            if chunk_data.get("type") == "done":
                continue
            chunk = StreamChunk(
                type=chunk_data["type"],
                delta=chunk_data["delta"],
                metadata=chunk_data.get("metadata", {}),
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
    except Exception as e:
        error_chunk = StreamChunk(
            type="error",
            delta="",
            metadata={"error": str(e)},
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    user: CurrentUser,
):
    """
    Send a message to the agent and get a response.

    When options.stream=true, returns Server-Sent Events (SSE).
    Otherwise returns a standard JSON response.
    """
    request_id = f"req_{uuid4().hex[:12]}"
    start_time = time.time()

    # Determine options
    stream = request.options.stream if request.options else False
    thinking = request.options.thinking if request.options else False

    try:
        agent = get_agent()

        # Get session key from user token or request
        session_key = request.session_key
        if hasattr(user, "session_key"):
            session_key = user.session_key

        # Streaming mode: return SSE
        if stream:
            return StreamingResponse(
                _stream_sse(agent, request.message, request.media or None, thinking),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                },
            )

        # Non-streaming mode
        kwargs = {"message": request.message}
        if request.media:
            kwargs["media"] = request.media

        thinking_content = None
        if thinking:
            response_text, thinking_content = await agent.process_with_thinking(**kwargs)
        else:
            response_text = await agent.process(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        return APIResponse(
            success=True,
            data=ChatResponse(
                response=response_text,
                tool_calls=[],  # TODO: Extract tool calls from response
                usage=None,  # TODO: Track usage
                thinking_content=thinking_content,
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
                    active_sessions=len(agent.sessions.list_sessions()) if hasattr(agent.sessions, 'list_sessions') else 0,
                    tools_available=len(agent.tools.list_tools()) if hasattr(agent.tools, 'list_tools') else len(agent.tools) if hasattr(agent.tools, '__len__') else 0,
                    skills_loaded=len(agent.skills.list_skills()) if hasattr(agent.skills, 'list_skills') else len(agent.skills) if hasattr(agent.skills, '__len__') else 0,  # Skills count via agent if available
                ),
            ),
            meta=MetaInfo(request_id=request_id),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": str(e)},
        )
