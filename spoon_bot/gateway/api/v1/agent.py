"""Agent control endpoints."""

from __future__ import annotations

import asyncio
import time
from typing import Annotated
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
    StatusResponse,
    AgentStats,
)

router = APIRouter()
_TASKS: dict[str, dict] = {}


async def _process_with_session(agent, *, message: str, session_key: str, media: list[str] | None = None) -> str:
    kwargs = {"message": message, "session_key": session_key}
    if media:
        kwargs["media"] = media
    try:
        return await agent.process(**kwargs)
    except TypeError:
        kwargs.pop("session_key", None)
        return await agent.process(**kwargs)


@router.post("/chat", response_model=APIResponse[ChatResponse])
async def chat(
    request: ChatRequest,
    user: CurrentUser,
):
    """Send a message to the agent and get a response (JSON or SSE)."""
    request_id = f"req_{uuid4().hex[:12]}"
    start_time = time.time()

    try:
        agent = get_agent()

        session_key = request.session_key
        if hasattr(user, "session_key"):
            session_key = user.session_key

        stream = bool(request.options and request.options.stream)

        if stream:
            async def event_stream():
                sent_any_chunk = False
                buffer_parts: list[str] = []
                try:
                    kwargs = {"message": request.message, "session_key": session_key}
                    if request.media:
                        kwargs["media"] = request.media

                    try:
                        stream_iter = agent.stream(**kwargs)
                    except TypeError:
                        kwargs.pop("session_key", None)
                        stream_iter = agent.stream(**kwargs)

                    async for chunk in stream_iter:
                        chunk_text = str(chunk or "")
                        if chunk_text:
                            buffer_parts.append(chunk_text)
                            sent_any_chunk = True

                    for chunk_text in buffer_parts:
                        yield f"data: {chunk_text}\n\n"
                except Exception as e:
                    yield f"data: [ERROR] {str(e)}\n\n"
                finally:
                    if not sent_any_chunk:
                        yield "data: [ERROR] empty stream response\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        response_text = await _process_with_session(
            agent,
            message=request.message,
            session_key=session_key,
            media=request.media,
        )

        duration_ms = int((time.time() - start_time) * 1000)

        return APIResponse(
            success=True,
            data=ChatResponse(response=response_text, tool_calls=[], usage=None),
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
    """Queue async chat task and return task ID."""
    task_id = f"task_{uuid4().hex[:12]}"
    agent = get_agent()

    session_key = request.session_key
    if hasattr(user, "session_key"):
        session_key = user.session_key

    _TASKS[task_id] = {
        "task_id": task_id,
        "status": "pending",
        "result": None,
        "error": None,
    }

    async def _runner() -> None:
        _TASKS[task_id]["status"] = "running"
        try:
            result = await _process_with_session(
                agent,
                message=request.message,
                session_key=session_key,
                media=request.media,
            )
            _TASKS[task_id]["status"] = "completed"
            _TASKS[task_id]["result"] = result
        except asyncio.CancelledError:
            _TASKS[task_id]["status"] = "cancelled"
            raise
        except Exception as e:
            _TASKS[task_id]["status"] = "failed"
            _TASKS[task_id]["error"] = str(e)

    handle = asyncio.create_task(_runner())
    _TASKS[task_id]["handle"] = handle

    return {"task_id": task_id, "status": "pending"}


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    user: CurrentUser,
) -> dict:
    """Get async task status and result."""
    task = _TASKS.get(task_id)
    if not task:
        return {"task_id": task_id, "status": "not_found", "message": "Task not found"}

    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task.get("result"),
        "error": task.get("error"),
    }


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    user: CurrentUser,
) -> dict:
    """Cancel async task if still running."""
    task = _TASKS.get(task_id)
    if not task:
        return {"task_id": task_id, "cancelled": False, "message": "Task not found"}

    handle = task.get("handle")
    if handle and not handle.done():
        handle.cancel()
        task["status"] = "cancelled"
        return {"task_id": task_id, "cancelled": True}

    return {
        "task_id": task_id,
        "cancelled": False,
        "message": f"Task already {task['status']}",
    }


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
                uptime=0,
                stats=AgentStats(
                    total_requests=0,
                    active_sessions=len(agent.sessions.list_sessions()) if hasattr(agent.sessions, "list_sessions") else 0,
                    tools_available=len(agent.tools.list_tools()) if hasattr(agent.tools, "list_tools") else len(agent.tools) if hasattr(agent.tools, "__len__") else 0,
                    skills_loaded=len(agent.skills.list_skills()) if hasattr(agent.skills, "list_skills") else len(agent.skills) if hasattr(agent.skills, "__len__") else 0,
                ),
            ),
            meta=MetaInfo(request_id=request_id),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": str(e)},
        )
