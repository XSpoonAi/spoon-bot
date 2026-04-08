"""Agent control endpoints."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from enum import Enum
from typing import Any, Annotated, AsyncGenerator
from uuid import uuid4

from logging import getLogger

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from spoon_bot.gateway.app import (
    get_agent,
    get_config,
    get_agent_execution_lock,
    get_session_execution_lock,
)
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.errors import TimeoutCode, build_timeout_error_detail
from spoon_bot.gateway.models.requests import AsyncChatRequest, ChatRequest
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
    ChannelsInfo,
    ChannelStatusInfo,
)
from spoon_bot.gateway.observability.budget import BudgetExhaustedError
from spoon_bot.gateway.observability.tracing import (
    TimerSpan,
    build_timing_payload,
    new_trace_id,
)
from spoon_bot.gateway.websocket.handler import (
    _derive_media_from_attachments,
    _merge_attachment_context,
    _normalize_attachment_refs,
    _validate_attachment_paths,
    _validate_media_paths,
)

logger = getLogger(__name__)

router = APIRouter()


def _format_sse_event(event: str, payload: dict[str, Any]) -> str:
    """Format one SSE event frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _resolve_session_key(request_session_key: str, user: CurrentUser) -> str:
    """Resolve session key using request override, then user default."""
    session_key = request_session_key
    if (
        session_key == "default"
        and hasattr(user, "session_key")
        and user.session_key
        and user.session_key != "default"
    ):
        session_key = user.session_key
    return session_key


def _raise_request_validation_error(exc: ValueError) -> None:
    """Convert client-supplied validation failures into 4xx responses."""
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"code": "INVALID_REQUEST", "message": str(exc)},
    )


def _prepare_attachment_inputs(
    message: str,
    media: list[str] | None,
    attachments: list[dict[str, Any]] | None,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """Normalize and validate attachment/media inputs for agent requests."""
    try:
        normalized_attachments = _validate_attachment_paths(
            _normalize_attachment_refs(attachments or [])
        )
        normalized_media = _validate_media_paths(
            list(
                dict.fromkeys(
                    (media or []) + _derive_media_from_attachments(normalized_attachments)
                )
            )
        )
    except ValueError as exc:
        _raise_request_validation_error(exc)

    return (
        _merge_attachment_context(message, normalized_attachments),
        normalized_media,
        normalized_attachments,
    )


# ---------------------------------------------------------------------------
# Audio processing helper
# ---------------------------------------------------------------------------

async def _process_audio_input(
    audio_data: str | bytes,
    audio_format: str = "wav",
    mime_type: str | None = None,
    message: str = "",
    language: str | None = None,
) -> tuple[str, TranscriptionInfo | None]:
    """Process audio input through the audio pipeline.

    Returns (message_text, transcription_info).
    """
    config = get_config()
    provider_obj = getattr(get_agent(), "config", None)
    provider_name = getattr(provider_obj, "provider", "anthropic") if provider_obj else "anthropic"

    try:
        from spoon_bot.services.audio.pipeline import AudioPipeline

        pipeline = AudioPipeline(
            provider=provider_name,
            stt_provider=config.audio.stt_provider,
            stt_model=config.audio.stt_model,
            native_audio_providers=frozenset(config.audio.native_audio_providers),
        )

        if isinstance(audio_data, str):
            processed = await pipeline.process_base64(
                b64_audio=audio_data,
                audio_format=audio_format,
                mime_type=mime_type,
                text=message,
                language=language,
            )
        else:
            processed = await pipeline.process(
                audio_data=audio_data,
                audio_format=audio_format,
                text=message,
                language=language,
            )

        transcription_info = None
        if processed.transcription:
            transcription_info = TranscriptionInfo(
                text=processed.transcription.text,
                language=processed.transcription.language,
                duration_seconds=processed.transcription.duration_seconds,
                provider=processed.transcription.provider,
            )
        return processed.text, transcription_info

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "AUDIO_PROCESSING_ERROR", "message": str(e)},
        )


def _switch_session(agent, session_key: str | None) -> None:
    """Switch the agent's active session if the agent supports it."""
    if not session_key or session_key == "default":
        return
    # AgentLoop stores sessions via SessionManager
    if hasattr(agent, "sessions") and hasattr(agent, "_session"):
        try:
            agent._session = agent.sessions.get_or_create(session_key)
            agent.session_key = session_key
        except Exception:
            pass  # Gracefully degrade — keep default session


async def _stream_sse(
    agent,
    message: str,
    media: list[str] | None,
    attachments: list[dict[str, Any]] | None,
    thinking: bool,
    *,
    session_key: str | None = None,
    user_id: str | None = None,
    trace_id: str | None = None,
    request_id: str | None = None,
    cancel_event: asyncio.Event | None = None,
) -> AsyncGenerator[str, None]:
    """Generate SSE events from agent streaming."""
    resolved_trace_id = trace_id or new_trace_id()
    resolved_request_id = request_id or f"req_{uuid4().hex[:12]}"
    span = TimerSpan("rest_sse")
    streamed_content = ""
    yield _format_sse_event(
        "trace",
        {
            "trace_id": resolved_trace_id,
            "request_id": resolved_request_id,
        },
    )

    resolved_session_key = session_key or "default"
    session_lock = get_session_execution_lock(resolved_session_key)
    agent_lock = get_agent_execution_lock()

    async with session_lock:
        async with agent_lock:
            _switch_session(agent, resolved_session_key)
            setattr(agent, "user_id", user_id or "anonymous")

            kwargs = {"message": message, "thinking": thinking}
            if media:
                kwargs["media"] = media
            if attachments:
                kwargs["attachments"] = attachments

            try:
                async for chunk_data in agent.stream(**kwargs):
                    if cancel_event and cancel_event.is_set():
                        break

                    chunk_type = chunk_data.get("type", "content")

                    # Propagate error chunks to the client
                    if chunk_type == "error":
                        error_chunk = StreamChunk(
                            type="error",
                            delta=chunk_data.get("delta", ""),
                            metadata=chunk_data.get("metadata", {}),
                        )
                        yield f"data: {error_chunk.model_dump_json()}\n\n"
                        continue

                    # Filter out "done" chunks — clients use [DONE] as the completion signal
                    if chunk_type == "done":
                        done_metadata = chunk_data.get("metadata", {})
                        done_content = (
                            done_metadata.get("content", "")
                            if isinstance(done_metadata, dict)
                            else ""
                        )
                        if not streamed_content and isinstance(done_content, str) and done_content:
                            fallback_chunk = StreamChunk(
                                type="content",
                                delta=done_content,
                                metadata={"fallback": "done_metadata_content"},
                            )
                            yield f"data: {fallback_chunk.model_dump_json()}\n\n"
                            streamed_content = done_content
                        continue

                    chunk = StreamChunk(
                        type=chunk_type,
                        delta=chunk_data["delta"],
                        metadata=chunk_data.get("metadata", {}),
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
                    if chunk_type == "content" and chunk.delta:
                        streamed_content += chunk.delta
            except Exception as e:
                error_chunk = StreamChunk(
                    type="error",
                    delta="",
                    metadata={"error": str(e), "trace_id": resolved_trace_id},
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"

    span.stop()
    yield _format_sse_event(
        "timing",
        build_timing_payload(
            span,
            extra={
                "trace_id": resolved_trace_id,
                "request_id": resolved_request_id,
            },
        ),
    )
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
    request_span = TimerSpan("rest_chat")
    trace_id = new_trace_id()

    # Determine options
    stream = request.options.stream if request.options else False
    thinking = request.options.thinking if request.options else False

    try:
        agent = get_agent()
        owner_user_id = getattr(user, "user_id", "anonymous")

        # Session key: request body takes priority over user token.
        session_key = _resolve_session_key(request.session_key, user)

        # Process audio input if provided
        transcription_info = None
        message = request.message
        if request.audio_data:
            message, transcription_info = await _process_audio_input(
                audio_data=request.audio_data,
                audio_format=request.audio_format or "wav",
                mime_type=request.audio_mime_type,
                message=request.message,
                language=request.audio_language,
            )

        message, media, attachments = _prepare_attachment_inputs(
            message,
            request.media,
            request.attachments,
        )

        # Streaming mode: return SSE
        if stream:
            return StreamingResponse(
                _stream_sse(
                    agent,
                    message,
                    media or None,
                    attachments or None,
                    thinking,
                    session_key=session_key,
                    user_id=owner_user_id,
                    trace_id=trace_id,
                    request_id=request_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Request-ID": request_id,
                    "X-Trace-ID": trace_id,
                },
            )

        session_lock = get_session_execution_lock(session_key)
        agent_lock = get_agent_execution_lock()
        async with session_lock:
            async with agent_lock:
                # Non-streaming mode — switch session before processing
                _switch_session(agent, session_key)
                setattr(agent, "user_id", owner_user_id)

                kwargs = {"message": message}
                if media:
                    kwargs["media"] = media
                if attachments:
                    kwargs["attachments"] = attachments

                thinking_content = None
                if thinking:
                    response_text, thinking_content = await agent.process_with_thinking(**kwargs)
                else:
                    response_text = await agent.process(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)
        request_span.stop()
        timing = build_timing_payload(request_span, extra={"trace_id": trace_id})

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
                duration_ms=duration_ms,
                trace_id=trace_id,
                timing=timing,
            ),
        )

    except HTTPException:
        raise
    except BudgetExhaustedError as exc:
        request_span.stop()
        limit_ms = exc.limit_ms
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "code": TimeoutCode.TIMEOUT_TOTAL.value,
                "message": str(exc),
                "details": {
                    "budget_type": exc.budget_type,
                    "elapsed_ms": exc.elapsed_ms,
                    "limit_ms": limit_ms,
                },
            },
        )
    except asyncio.TimeoutError:
        request_span.stop()
        try:
            request_limit_ms = get_config().budget.request_timeout_ms
        except Exception:
            request_limit_ms = 0
        timeout_detail = build_timeout_error_detail(
            TimeoutCode.TIMEOUT_UPSTREAM,
            elapsed_ms=request_span.elapsed_ms,
            limit_ms=request_limit_ms,
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=timeout_detail.model_dump(),
        )

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
    owner_user_id: str
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    created_at: float = dataclasses.field(default_factory=time.time)
    completed_at: float | None = None
    _cancel_event: asyncio.Event = dataclasses.field(default_factory=asyncio.Event)
    _bg_task: asyncio.Task | None = dataclasses.field(default=None, repr=False)


# Simple in-memory task store (sufficient for single-process deployments)
_task_store: dict[str, AsyncTask] = {}


async def _run_async_task(
    task: AsyncTask,
    agent,
    message: str,
    session_key: str | None = None,
    user_id: str | None = None,
):
    """Background coroutine that drives agent processing for an async task."""
    task.status = TaskStatus.RUNNING
    try:
        resolved_session_key = session_key or "default"
        session_lock = get_session_execution_lock(resolved_session_key)
        agent_lock = get_agent_execution_lock()
        async with session_lock:
            async with agent_lock:
                _switch_session(agent, resolved_session_key)
                setattr(agent, "user_id", user_id or "anonymous")
                if task._cancel_event.is_set():
                    task.status = TaskStatus.CANCELLED
                    return
                response = await agent.process(message=message)
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
    request: AsyncChatRequest,
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

    session_key = _resolve_session_key(request.session_key, user)
    owner_user_id = getattr(user, "user_id", "anonymous")

    task = AsyncTask(task_id=task_id, owner_user_id=owner_user_id)
    _task_store[task_id] = task

    bg = asyncio.create_task(
        _run_async_task(
            task,
            agent,
            request.message,
            session_key=session_key,
            user_id=owner_user_id,
        )
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
    current_user_id = getattr(user, "user_id", "anonymous")
    if task.owner_user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "Task does not belong to current user"},
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
    current_user_id = getattr(user, "user_id", "anonymous")
    if task.owner_user_id != current_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "Task does not belong to current user"},
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
    """Get agent status and statistics, including channel health."""
    from spoon_bot.gateway.app import _channel_manager

    request_id = f"req_{uuid4().hex[:12]}"

    try:
        agent = get_agent()

        # Build channel info if manager is available
        channels_info = None
        if _channel_manager is not None:
            ch_list: list[ChannelStatusInfo] = []
            for name in _channel_manager.channel_names:
                ch = _channel_manager.get_channel(name)
                ch_status = "running" if ch and ch.is_running else "stopped"
                ch_list.append(ChannelStatusInfo(name=name, status=ch_status))
            channels_info = ChannelsInfo(
                running=_channel_manager.running_channels_count,
                total=len(_channel_manager.channel_names),
                channels=ch_list,
            )

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
                channels=channels_info,
            ),
            meta=MetaInfo(request_id=request_id),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": str(e)},
        )


# ---------------------------------------------------------------------------
# Voice / audio endpoints
# ---------------------------------------------------------------------------


@router.post("/voice/transcribe")
async def voice_transcribe(
    audio: Annotated[UploadFile, File(description="Audio file to transcribe")],
    language: Annotated[str | None, Form(description="ISO 639-1 language hint")] = None,
    user: CurrentUser = None,
):
    """Transcribe an audio file to text (STT only, no agent processing)."""
    request_id = f"req_{uuid4().hex[:12]}"
    start_time = time.time()

    config = get_config()
    if not config.audio.enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "AUDIO_DISABLED", "message": "Audio input is disabled"},
        )

    # Read and validate upload
    audio_bytes = await audio.read()
    max_bytes = config.audio.max_audio_size_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "AUDIO_TOO_LARGE",
                "message": f"Audio file exceeds {config.audio.max_audio_size_mb}MB limit",
            },
        )

    # Determine format from filename or content type
    audio_format = None
    if audio.filename:
        ext = audio.filename.rsplit(".", 1)[-1].lower() if "." in audio.filename else None
        if ext in {"wav", "mp3", "ogg", "webm", "flac", "m4a", "aac"}:
            audio_format = ext

    _, transcription_info = await _process_audio_input(
        audio_data=audio_bytes,
        audio_format=audio_format or "wav",
        mime_type=audio.content_type,
        message="",
        language=language,
    )

    duration_ms = int((time.time() - start_time) * 1000)

    return APIResponse(
        success=True,
        data=transcription_info,
        meta=MetaInfo(request_id=request_id, duration_ms=duration_ms),
    )


@router.post("/voice/chat")
async def voice_chat(
    audio: Annotated[UploadFile, File(description="Audio file")],
    message: Annotated[str, Form(description="Optional text message")] = "",
    session_key: Annotated[str, Form(description="Session key")] = "default",
    language: Annotated[str | None, Form(description="ISO 639-1 language hint")] = None,
    stream: Annotated[bool, Form(description="Stream response")] = False,
    user: CurrentUser = None,
):
    """Send voice + optional text to the agent (multipart upload)."""
    request_id = f"req_{uuid4().hex[:12]}"
    start_time = time.time()
    trace_id = new_trace_id()
    owner_user_id = getattr(user, "user_id", "anonymous") if user else "anonymous"

    config = get_config()
    if not config.audio.enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "AUDIO_DISABLED", "message": "Audio input is disabled"},
        )

    agent = get_agent()

    audio_bytes = await audio.read()
    max_bytes = config.audio.max_audio_size_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "AUDIO_TOO_LARGE",
                "message": f"Audio file exceeds {config.audio.max_audio_size_mb}MB limit",
            },
        )

    # Determine format
    audio_format = None
    if audio.filename:
        ext = audio.filename.rsplit(".", 1)[-1].lower() if "." in audio.filename else None
        if ext in {"wav", "mp3", "ogg", "webm", "flac", "m4a", "aac"}:
            audio_format = ext

    processed_message, transcription_info = await _process_audio_input(
        audio_data=audio_bytes,
        audio_format=audio_format or "wav",
        mime_type=audio.content_type,
        message=message,
        language=language,
    )

    if stream:
        return StreamingResponse(
            _stream_sse(
                agent,
                processed_message,
                None,
                None,
                False,
                session_key=session_key,
                user_id=owner_user_id,
                trace_id=trace_id,
                request_id=request_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Request-ID": request_id,
                "X-Trace-ID": trace_id,
            },
        )

    _switch_session(agent, session_key)
    setattr(agent, "user_id", owner_user_id)
    response_text = await agent.process(message=processed_message)
    duration_ms = int((time.time() - start_time) * 1000)

    return APIResponse(
        success=True,
        data=ChatResponse(
            response=response_text,
            transcription=transcription_info,
        ),
        meta=MetaInfo(request_id=request_id, duration_ms=duration_ms),
    )
