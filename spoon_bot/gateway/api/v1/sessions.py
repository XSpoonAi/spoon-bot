"""Session management endpoints."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, status

from spoon_bot.gateway.app import get_agent
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.models.requests import SessionCreateRequest
from spoon_bot.gateway.models.responses import (
    APIResponse,
    MetaInfo,
    SessionResponse,
    SessionListResponse,
    SessionInfo,
    SessionSearchHit,
    SessionSearchResponse,
)

router = APIRouter()


def _get_sessions_manager() -> Any:
    agent: Any = get_agent()
    sessions_manager: Any = getattr(agent, "sessions", None)
    if sessions_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "INTERNAL_ERROR", "message": "Session manager unavailable"},
        )
    return sessions_manager


def _run_search(
    sessions_manager: Any,
    *,
    query: str,
    session_key: str | None,
    regex: bool,
    case_sensitive: bool,
    roles: list[str] | None,
    include_extras: bool,
    limit: int,
    offset: int,
    max_content_length: int | None,
) -> list[SessionSearchHit]:
    """Call the session manager's ``search_messages`` and shape the results."""
    try:
        hits = sessions_manager.search_messages(
            query,
            session_key=session_key,
            regex=regex,
            case_sensitive=case_sensitive,
            roles=roles,
            include_extras=include_extras,
            limit=limit,
            offset=offset,
            max_content_length=max_content_length,
        )
    except re.error as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_REGEX", "message": str(exc)},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_QUERY", "message": str(exc)},
        )

    return [
        SessionSearchHit(
            session_key=hit.session_key,
            seq=hit.seq,
            role=hit.role,
            content=hit.content,
            timestamp=hit.timestamp,
            matched_in=hit.matched_in,
            snippet=hit.snippet,
            extras=hit.extras or {},
        )
        for hit in hits
    ]


@router.get("", response_model=APIResponse[SessionListResponse])
async def list_sessions(user: CurrentUser) -> APIResponse[SessionListResponse]:
    """List all sessions."""
    request_id = f"req_{uuid4().hex[:12]}"
    sessions_manager = _get_sessions_manager()

    session_keys = sessions_manager.list_sessions()
    sessions: list[Any] = []

    for key in session_keys:
        session = sessions_manager.get(key)
        if session is None:
            continue
        sessions.append(session)

    return APIResponse(
        success=True,
        data=SessionListResponse(
            sessions=[
                SessionInfo(
                    key=s.session_key,
                    created_at=s.created_at,
                    message_count=len(s.messages),
                )
                for s in sessions
            ]
        ),
        meta=MetaInfo(request_id=request_id),
    )


@router.post("", response_model=APIResponse[SessionResponse])
async def create_session(
    request: SessionCreateRequest,
    user: CurrentUser,
) -> APIResponse[SessionResponse]:
    """Create a new session."""
    request_id = f"req_{uuid4().hex[:12]}"
    sessions_manager = _get_sessions_manager()

    existing = sessions_manager.get(request.key)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "SESSION_EXISTS", "message": f"Session '{request.key}' already exists"},
        )

    session = sessions_manager.get_or_create(request.key)
    sessions_manager.save(session)

    return APIResponse(
        success=True,
        data=SessionResponse(
            session=SessionInfo(
                key=session.session_key,
                created_at=session.created_at,
                message_count=0,
            )
        ),
        meta=MetaInfo(request_id=request_id),
    )


# NOTE: Route registration order matters in FastAPI — static paths like
# ``/search`` MUST be registered before ``/{session_key}`` or the latter
# will swallow them as ``session_key="search"``.  Keep the search routes
# ABOVE the ``/{session_key}`` handlers.
@router.get(
    "/search",
    response_model=APIResponse[SessionSearchResponse],
)
async def search_sessions(
    user: CurrentUser,
    q: str = Query(..., min_length=1, description="Search query"),
    regex: bool = Query(False, description="Interpret query as a regular expression"),
    case_sensitive: bool = Query(False, description="Match case-sensitively"),
    roles: list[str] | None = Query(
        None,
        description="Restrict to these message roles (e.g. tool, assistant, user, system)",
    ),
    include_extras: bool = Query(
        True,
        description="Also match tool_call_id / tool_calls / attachments serialized extras",
    ),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    max_content_length: int | None = Query(
        2000,
        ge=1,
        le=100_000,
        description="Truncate each returned content to at most this many characters",
    ),
) -> APIResponse[SessionSearchResponse]:
    """Search across *all* persisted sessions for the current agent.

    This is the fallback used when runtime context has been compacted and
    callers (user or model) want to recover earlier tool calls / results
    or message bodies.  The underlying store picks the most efficient
    strategy available (SQL ``LIKE``/``ILIKE``/regex, or an in-memory
    grep fallback for file-backed stores).
    """
    request_id = f"req_{uuid4().hex[:12]}"
    sessions_manager = _get_sessions_manager()

    hits = _run_search(
        sessions_manager,
        query=q,
        session_key=None,
        regex=regex,
        case_sensitive=case_sensitive,
        roles=roles,
        include_extras=include_extras,
        limit=limit,
        offset=offset,
        max_content_length=max_content_length,
    )

    return APIResponse(
        success=True,
        data=SessionSearchResponse(
            query=q,
            total=len(hits),
            limit=limit,
            offset=offset,
            hits=hits,
        ),
        meta=MetaInfo(request_id=request_id),
    )


@router.get("/{session_key}", response_model=APIResponse[SessionResponse])
async def get_session(
    session_key: str,
    user: CurrentUser,
) -> APIResponse[SessionResponse]:
    """Get session details."""
    request_id = f"req_{uuid4().hex[:12]}"
    sessions_manager = _get_sessions_manager()

    session = sessions_manager.get(session_key)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Session '{session_key}' not found"},
        )

    return APIResponse(
        success=True,
        data=SessionResponse(
            session=SessionInfo(
                key=session.session_key,
                created_at=session.created_at,
                message_count=len(session.messages),
            )
        ),
        meta=MetaInfo(request_id=request_id),
    )


@router.get(
    "/{session_key}/search",
    response_model=APIResponse[SessionSearchResponse],
)
async def search_session(
    session_key: str,
    user: CurrentUser,
    q: str = Query(..., min_length=1, description="Search query"),
    regex: bool = Query(False),
    case_sensitive: bool = Query(False),
    roles: list[str] | None = Query(None),
    include_extras: bool = Query(True),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    max_content_length: int | None = Query(2000, ge=1, le=100_000),
) -> APIResponse[SessionSearchResponse]:
    """Search persisted messages within a single session.

    Primary fallback path for the user / model to recover an earlier
    tool call, tool result, or message body after runtime context
    compaction has trimmed it from memory.
    """
    request_id = f"req_{uuid4().hex[:12]}"
    sessions_manager = _get_sessions_manager()

    if sessions_manager.get(session_key) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Session '{session_key}' not found"},
        )

    hits = _run_search(
        sessions_manager,
        query=q,
        session_key=session_key,
        regex=regex,
        case_sensitive=case_sensitive,
        roles=roles,
        include_extras=include_extras,
        limit=limit,
        offset=offset,
        max_content_length=max_content_length,
    )

    return APIResponse(
        success=True,
        data=SessionSearchResponse(
            query=q,
            total=len(hits),
            limit=limit,
            offset=offset,
            hits=hits,
        ),
        meta=MetaInfo(request_id=request_id),
    )


@router.delete("/{session_key}")
async def delete_session(
    session_key: str,
    user: CurrentUser,
) -> dict:
    """Delete a session."""
    sessions_manager = _get_sessions_manager()
    deleted = sessions_manager.delete(session_key)
    return {"deleted": deleted}


@router.post("/{session_key}/clear")
async def clear_session(
    session_key: str,
    user: CurrentUser,
) -> dict:
    """Clear session history."""
    sessions_manager = _get_sessions_manager()

    session = sessions_manager.get(session_key)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Session '{session_key}' not found"},
        )

    count = len(session.messages)
    session.messages.clear()
    sessions_manager.save(session)

    return {"cleared": True, "messages_removed": count}
