"""Agent control endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from spoon_bot.gateway.app import get_agent, get_config
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.config import BudgetConfig
from spoon_bot.gateway.errors import TimeoutCode
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
from spoon_bot.gateway.observability.budget import BudgetExhaustedError, check_budget
from spoon_bot.gateway.observability.tracing import (
    new_trace_id,
    TimerSpan,
    build_timing_payload,
)

router = APIRouter()


def _get_budget_config() -> BudgetConfig:
    """Get budget configuration, with safe fallback to defaults."""
    try:
        config = get_config()
        return config.budget
    except RuntimeError:
        return BudgetConfig()


async def _stream_sse(
    agent,
    message: str,
    media: list[str] | None,
    thinking: bool,
    *,
    trace_id: str | None = None,
    request_id: str | None = None,
    cancel_event: asyncio.Event | None = None,
    budget_config: BudgetConfig | None = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE events from agent streaming.

    Yields trace and timing side-channel events alongside content chunks.
    Supports cancellation via cancel_event and stream timeout via budget_config.
    """
    span = TimerSpan("stream")
    budget = budget_config or _get_budget_config()

    # Emit trace side-channel event at the start
    if trace_id or request_id:
        trace_payload = {}
        if trace_id:
            trace_payload["trace_id"] = trace_id
        if request_id:
            trace_payload["request_id"] = request_id
        yield f"event: trace\ndata: {json.dumps(trace_payload)}\n\n"

    kwargs = {"message": message, "thinking": thinking}
    if media:
        kwargs["media"] = media

    try:
        async for chunk_data in agent.stream(**kwargs):
            # Check cancellation
            if cancel_event is not None and cancel_event.is_set():
                break

            # Check stream budget
            if budget.stream_timeout_ms > 0:
                try:
                    check_budget("stream", budget.stream_timeout_ms, span.elapsed_ms)
                except BudgetExhaustedError:
                    error_chunk = StreamChunk(
                        type="error",
                        delta="",
                        metadata={
                            "error": "Stream timeout exceeded",
                            "code": TimeoutCode.TIMEOUT_TOTAL.value,
                        },
                    )
                    yield f"data: {error_chunk.model_dump_json()}\n\n"
                    break

            # Filter out "done" chunks — clients use [DONE] as the completion signal
            if chunk_data.get("type") == "done":
                continue
            chunk = StreamChunk(
                type=chunk_data["type"],
                delta=chunk_data["delta"],
                metadata=chunk_data.get("metadata", {}),
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    except asyncio.TimeoutError:
        error_chunk = StreamChunk(
            type="error",
            delta="",
            metadata={
                "error": "Upstream service timed out",
                "code": TimeoutCode.TIMEOUT_UPSTREAM.value,
            },
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"

    except BudgetExhaustedError as exc:
        error_chunk = StreamChunk(
            type="error",
            delta="",
            metadata={
                "error": str(exc),
                "code": TimeoutCode.TIMEOUT_TOTAL.value,
            },
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"

    except Exception as exc:
        error_chunk = StreamChunk(
            type="error",
            delta="",
            metadata={"error": str(exc)},
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"

    # Emit timing side-channel event before [DONE]
    span.stop()
    timing_payload = build_timing_payload(span)
    yield f"event: timing\ndata: {json.dumps(timing_payload)}\n\n"

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
    trace_id = new_trace_id()
    budget = _get_budget_config()
    span = TimerSpan("request")

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
                    trace_id=trace_id,
                    request_id=request_id,
                    budget_config=budget,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                    "X-Trace-ID": trace_id,
                },
            )

        # Non-streaming mode
        kwargs = {"message": request.message}
        if request.media:
            kwargs["media"] = request.media

        thinking_content = None
        timeout_sec = (
            budget.request_timeout_ms / 1000
            if budget.request_timeout_ms > 0
            else None
        )

        try:
            if thinking:
                if timeout_sec:
                    response_text, thinking_content = await asyncio.wait_for(
                        agent.process_with_thinking(**kwargs),
                        timeout=timeout_sec,
                    )
                else:
                    response_text, thinking_content = await agent.process_with_thinking(**kwargs)
            else:
                if timeout_sec:
                    response_text = await asyncio.wait_for(
                        agent.process(**kwargs),
                        timeout=timeout_sec,
                    )
                else:
                    response_text = await agent.process(**kwargs)

        except asyncio.TimeoutError:
            span.stop()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "code": TimeoutCode.TIMEOUT_UPSTREAM.value,
                    "message": "Upstream service timed out",
                    "details": {
                        "elapsed_ms": span.elapsed_ms,
                        "limit_ms": budget.request_timeout_ms,
                    },
                },
            )

        span.stop()
        timing = build_timing_payload(span)

        return APIResponse(
            success=True,
            data=ChatResponse(
                response=response_text,
                tool_calls=[],  # TODO: Extract tool calls from response
                usage=None,  # TODO: Track usage
                thinking_content=thinking_content,
            ),
            meta=MetaInfo(
                request_id=request_id,
                duration_ms=span.elapsed_ms,
                trace_id=trace_id,
                timing=timing,
            ),
        )

    except HTTPException:
        raise

    except BudgetExhaustedError as exc:
        span.stop()
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail={
                "code": TimeoutCode.TIMEOUT_TOTAL.value,
                "message": str(exc),
                "details": {
                    "elapsed_ms": exc.elapsed_ms,
                    "limit_ms": exc.limit_ms,
                },
            },
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "AGENT_ERROR", "message": str(exc)},
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


# ========== Tool management ==========


@router.get("/tools")
async def list_tools(user: CurrentUser):
    """
    List all registered tools with their active/inactive status.

    Active tools are loaded into the agent; inactive tools can be
    activated dynamically via POST /tools/{name}/activate.
    """
    agent = get_agent()
    tools = agent.get_available_tools()
    active = [t for t in tools if t["active"]]
    inactive = [t for t in tools if not t["active"]]
    return APIResponse(
        success=True,
        data={
            "active": active,
            "inactive": inactive,
            "total": len(tools),
            "active_count": len(active),
        },
        meta=MetaInfo(request_id=f"req_{uuid4().hex[:12]}"),
    )


@router.post("/tools/{name}/activate")
async def activate_tool(name: str, user: CurrentUser):
    """Dynamically activate a tool and inject it into the running agent."""
    agent = get_agent()
    success = agent.add_tool(name)
    if not success:
        # Check if tool exists at all
        if name not in agent.tools:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "TOOL_NOT_FOUND", "message": f"Tool '{name}' is not registered"},
            )
        return APIResponse(
            success=True,
            data={"name": name, "activated": False, "reason": "already active"},
            meta=MetaInfo(request_id=f"req_{uuid4().hex[:12]}"),
        )
    return APIResponse(
        success=True,
        data={"name": name, "activated": True},
        meta=MetaInfo(request_id=f"req_{uuid4().hex[:12]}"),
    )


@router.post("/tools/{name}/deactivate")
async def deactivate_tool(name: str, user: CurrentUser):
    """Dynamically deactivate a tool and remove it from the running agent."""
    agent = get_agent()
    success = agent.remove_tool(name)
    if not success:
        return APIResponse(
            success=True,
            data={"name": name, "deactivated": False, "reason": "not active or not found"},
            meta=MetaInfo(request_id=f"req_{uuid4().hex[:12]}"),
        )
    return APIResponse(
        success=True,
        data={"name": name, "deactivated": True},
        meta=MetaInfo(request_id=f"req_{uuid4().hex[:12]}"),
    )


@router.post("/tools/profile/{profile}")
async def set_tool_profile(profile: str, user: CurrentUser):
    """
    Switch to a named tool profile.

    Available profiles: core, coding, web3, research, full.
    """
    agent = get_agent()
    try:
        agent.tools.set_tool_filter(tool_profile=profile)
        # Rebuild agent's ToolManager to match
        active_names = list(agent.tools.get_active_tools().keys())
        return APIResponse(
            success=True,
            data={"profile": profile, "active_tools": active_names},
            meta=MetaInfo(request_id=f"req_{uuid4().hex[:12]}"),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_PROFILE", "message": str(e)},
        )


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
