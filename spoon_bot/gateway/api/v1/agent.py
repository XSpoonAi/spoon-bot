"""Agent control endpoints."""

from __future__ import annotations

import asyncio
import json
import time
from logging import getLogger
from typing import Annotated, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
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
    TranscriptionInfo,
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

logger = getLogger(__name__)

router = APIRouter()


async def _process_audio_input(
    audio_data: str | bytes,
    audio_format: str = "wav",
    mime_type: str | None = None,
    message: str = "",
    language: str | None = None,
) -> tuple[str, TranscriptionInfo | None]:
    """Process audio input through the audio pipeline.

    Returns (message_text, transcription_info) — the message_text is either
    the original text combined with transcription, or the original message if
    the provider supports native audio (in which case the caller should build
    a multimodal message instead).
    """
    config = get_config()
    provider = getattr(get_agent(), "config", None)
    provider_name = getattr(provider, "provider", "anthropic") if provider else "anthropic"

    try:
        from spoon_bot.services.audio.pipeline import AudioPipeline

        pipeline = AudioPipeline(
            provider=provider_name,
            stt_provider=config.audio.stt_provider,
            stt_model=config.audio.stt_model,
            native_audio_providers=frozenset(config.audio.native_audio_providers),
        )

        if isinstance(audio_data, str):
            # Base64 encoded
            msg, transcription = await pipeline.process_base64(
                b64_audio=audio_data,
                audio_format=audio_format,
                mime_type=mime_type,
                text=message,
                language=language,
            )
        else:
            # Raw bytes
            msg, transcription = await pipeline.process(
                audio_data=audio_data,
                audio_format=audio_format,
                text=message,
                language=language,
            )

        # Extract the text content from the message
        result_text = msg.text_content if msg.is_multimodal else (msg.content or "")
        transcription_info = None
        if transcription:
            transcription_info = TranscriptionInfo(
                text=transcription.text,
                language=transcription.language,
                duration_seconds=transcription.duration_seconds,
                confidence=transcription.confidence,
                provider=transcription.provider,
            )
        return result_text, transcription_info

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "AUDIO_PROCESSING_ERROR", "message": str(e)},
        )


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

        # Process audio input if provided
        transcription_info = None
        effective_message = request.message
        if request.audio:
            effective_message, transcription_info = await _process_audio_input(
                audio_data=request.audio.data,
                audio_format=request.audio.format,
                message=request.message,
                language=request.audio.language,
            )
            if not effective_message:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"code": "EMPTY_INPUT", "message": "No text or audio content provided"},
                )

        # Streaming mode: return SSE
        if stream:
            return StreamingResponse(
                _stream_sse(
                    agent,
                    effective_message,
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
        kwargs = {"message": effective_message}
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
                transcription=transcription_info,
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


@router.post("/chat/audio")
async def chat_audio(
    user: CurrentUser,
    audio: UploadFile = File(..., description="Audio file (wav, mp3, ogg, webm, flac, m4a)"),
    message: str = Form(default="", description="Optional text message accompanying the audio"),
    session_key: str = Form(default="default", description="Session key"),
    stream: bool = Form(default=False, description="Enable streaming response"),
    language: str | None = Form(default=None, description="Language hint (ISO 639-1)"),
    thinking: bool = Form(default=False, description="Enable extended thinking"),
):
    """
    Send an audio message via multipart file upload.

    Accepts audio files up to 25MB in common formats (wav, mp3, ogg, webm, flac, m4a).
    The audio is either transcribed via STT or passed natively to the LLM, depending
    on the active provider's capabilities.
    """
    request_id = f"req_{uuid4().hex[:12]}"
    trace_id = new_trace_id()
    budget = _get_budget_config()
    span = TimerSpan("request")
    config = get_config()

    # Validate audio config is enabled
    if not config.audio.enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "AUDIO_DISABLED", "message": "Audio input is disabled"},
        )

    # Read audio file
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "EMPTY_AUDIO", "message": "Audio file is empty"},
        )

    # Check size limit
    max_size = config.audio.max_audio_size_mb * 1024 * 1024
    if len(audio_bytes) > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "AUDIO_TOO_LARGE",
                "message": f"Audio file too large: {len(audio_bytes) / (1024 * 1024):.1f}MB "
                           f"(max: {config.audio.max_audio_size_mb}MB)",
            },
        )

    # Detect format from content type or filename
    audio_format = "wav"
    if audio.content_type:
        from spoon_bot.services.audio.utils import mime_to_format
        try:
            audio_format = mime_to_format(audio.content_type)
        except ValueError:
            pass
    elif audio.filename:
        ext = audio.filename.rsplit(".", 1)[-1].lower() if "." in audio.filename else ""
        if ext in config.audio.supported_formats:
            audio_format = ext

    try:
        # Process through audio pipeline
        effective_message, transcription_info = await _process_audio_input(
            audio_data=audio_bytes,
            audio_format=audio_format,
            mime_type=audio.content_type,
            message=message,
            language=language or config.audio.default_language,
        )

        if not effective_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "EMPTY_INPUT", "message": "No speech detected in audio"},
            )

        agent = get_agent()

        # Streaming mode
        if stream:
            return StreamingResponse(
                _stream_sse(
                    agent,
                    effective_message,
                    None,
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
        timeout_sec = (
            budget.request_timeout_ms / 1000
            if budget.request_timeout_ms > 0
            else None
        )

        thinking_content = None
        try:
            if thinking:
                if timeout_sec:
                    response_text, thinking_content = await asyncio.wait_for(
                        agent.process_with_thinking(message=effective_message),
                        timeout=timeout_sec,
                    )
                else:
                    response_text, thinking_content = await agent.process_with_thinking(
                        message=effective_message
                    )
            else:
                if timeout_sec:
                    response_text = await asyncio.wait_for(
                        agent.process(message=effective_message),
                        timeout=timeout_sec,
                    )
                else:
                    response_text = await agent.process(message=effective_message)

        except asyncio.TimeoutError:
            span.stop()
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "code": TimeoutCode.TIMEOUT_UPSTREAM.value,
                    "message": "Upstream service timed out",
                },
            )

        span.stop()
        timing = build_timing_payload(span)

        return APIResponse(
            success=True,
            data=ChatResponse(
                response=response_text,
                tool_calls=[],
                usage=None,
                thinking_content=thinking_content,
                transcription=transcription_info,
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
