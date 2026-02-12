"""Agent control endpoints."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from enum import Enum
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


async def _stream_sse(
    agent,
    message: str,
    media: list[str] | None,
    thinking: bool,
    *,
    session_key: str | None = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE events from agent streaming."""
    kwargs = {"message": message, "thinking": thinking}
    if media:
        kwargs["media"] = media
    if session_key:
        kwargs["session_key"] = session_key

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
                _stream_sse(
                    agent,
                    request.message,
                    request.media or None,
                    thinking,
                    session_key=session_key,
                ),
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
        if session_key:
            kwargs["session_key"] = session_key

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

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "AGENT_ERROR", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# In-process async task queue
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclasses.dataclass
class AsyncTask:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    created_at: float = dataclasses.field(default_factory=time.time)
    completed_at: float | None = None
    _cancel_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    _bg_task: asyncio.Task | None = dataclasses.field(default=None, repr=False)


# Simple in-memory task store (sufficient for single-process deployments)
_task_store: dict[str, AsyncTask] = {}


async def _run_async_task(task: AsyncTask, agent, message: str, **kwargs):
    """Background coroutine that drives agent processing for an async task."""
    task.status = TaskStatus.RUNNING
    try:
        if task._cancel_event.is_set():
            task.status = TaskStatus.CANCELLED
            return
        response = await agent.process(message=message, **kwargs)
        if task._cancel_event.is_set():
            task.status = TaskStatus.CANCELLED
            return
        task.result = response
        task.status = TaskStatus.COMPLETED
    except asyncio.CancelledError:
        task.status = TaskStatus.CANCELLED
    except Exception as exc:
        task.error = str(exc)
        task.status = TaskStatus.FAILED
    finally:
        task.completed_at = time.time()


@router.post("/chat/async")
async def chat_async(
    request: ChatRequest,
    user: CurrentUser,
) -> dict:
    """
    Send a message asynchronously.

    Returns a task ID that can be polled for results.
    """
    task_id = f"task_{uuid4().hex[:12]}"

    try:
        agent = get_agent()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "AGENT_NOT_READY", "message": "Agent is not initialized yet"},
        )

    session_key = request.session_key
    extra_kwargs: dict[str, str] = {}
    if session_key:
        extra_kwargs["session_key"] = session_key

    task = AsyncTask(task_id=task_id)
    _task_store[task_id] = task

    bg = asyncio.create_task(
        _run_async_task(task, agent, request.message, **extra_kwargs)
    )
    task._bg_task = bg

    return {
        "task_id": task_id,
        "status": task.status.value,
        "created_at": task.created_at,
    }


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    user: CurrentUser,
) -> dict:
    """Get the status of an async task."""
    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TASK_NOT_FOUND", "message": f"Task '{task_id}' not found"},
        )

    payload: dict = {
        "task_id": task.task_id,
        "status": task.status.value,
        "created_at": task.created_at,
    }
    if task.result is not None:
        payload["result"] = task.result
    if task.error is not None:
        payload["error"] = task.error
    if task.completed_at is not None:
        payload["completed_at"] = task.completed_at
    return payload


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    user: CurrentUser,
) -> dict:
    """Cancel an async task."""
    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TASK_NOT_FOUND", "message": f"Task '{task_id}' not found"},
        )

    if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
        return {"cancelled": False, "message": f"Task already in terminal state: {task.status.value}"}

    task._cancel_event.set()
    if task._bg_task and not task._bg_task.done():
        task._bg_task.cancel()
    task.status = TaskStatus.CANCELLED
    task.completed_at = time.time()

    return {"cancelled": True, "task_id": task_id}


@router.get("/status", response_model=APIResponse[StatusResponse])
async def get_status(user: CurrentUser) -> APIResponse[StatusResponse]:
    """Get agent status and statistics."""
    request_id = f"req_{uuid4().hex[:12]}"

    try:
        agent = get_agent()

        # Safely count tools and skills — AgentLoop exposes these as
        # simple list[str] properties, not manager objects.
        tools_count = 0
        skills_count = 0
        sessions_count = 0
        try:
            tools_count = len(agent.tools) if hasattr(agent, 'tools') and agent.tools else 0
        except Exception:
            pass
        try:
            skills_count = len(agent.skills) if hasattr(agent, 'skills') and agent.skills else 0
        except Exception:
            pass
        try:
            if hasattr(agent, 'sessions') and agent.sessions:
                sessions_count = len(agent.sessions.list_sessions()) if hasattr(agent.sessions, 'list_sessions') else 0
        except Exception:
            pass

        return APIResponse(
            success=True,
            data=StatusResponse(
                status="ready",
                current_task=None,
                uptime=0,  # TODO: Track uptime
                stats=AgentStats(
                    total_requests=0,
                    active_sessions=sessions_count,
                    tools_available=tools_count,
                    skills_loaded=skills_count,
                ),
            ),
            meta=MetaInfo(request_id=request_id),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": str(e)},
        )
