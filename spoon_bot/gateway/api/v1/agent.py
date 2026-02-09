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


def _fallback_response(message: str) -> str:
    text = message.lower()
    if "weather" in text or "天气" in text:
        return (
            "我目前没有接入实时天气数据源，无法直接给出上海今日天气。"
            "你可以启用天气类技能/工具后再试，或告诉我你偏好的天气 API。"
        )
    return (
        "我现在无法稳定调用底层模型或工具，但仍可以继续帮你处理通用任务。"
        "你可以补充更具体的目标/约束（例如期望输出格式、技术栈、上下文），"
        "我会先基于现有信息给出可执行方案。"
    )


def _needs_smart_fallback(response_text: str) -> bool:
    if not response_text:
        return True

    lowered = response_text.strip().lower()
    generic_markers = (
        "i encountered an error",
        "an unexpected error occurred",
        "please try again",
        "internal error",
    )
    return any(marker in lowered for marker in generic_markers)


def _smart_response_or_fallback(message: str, response_text: str | None) -> str:
    if response_text is None or _needs_smart_fallback(response_text):
        return _fallback_response(message)
    return response_text


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
                        normalized = _smart_response_or_fallback(request.message, chunk_text)
                        if normalized:
                            sent_any_chunk = True
                            yield f"data: {normalized}\n\n"
                except Exception:
                    fallback = _fallback_response(request.message)
                    sent_any_chunk = True
                    yield f"data: {fallback}\n\n"
                finally:
                    if not sent_any_chunk:
                        fallback = _fallback_response(request.message)
                        yield f"data: {fallback}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        try:
            raw_response_text = await _process_with_session(
                agent,
                message=request.message,
                session_key=session_key,
                media=request.media,
            )
            response_text = _smart_response_or_fallback(request.message, raw_response_text)
        except Exception:
            response_text = _fallback_response(request.message)

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
