"""Session management endpoints."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status

from spoon_bot.gateway.app import get_agent
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.models.requests import SessionCreateRequest
from spoon_bot.gateway.models.responses import (
    APIResponse,
    MetaInfo,
    SessionResponse,
    SessionListResponse,
    SessionInfo,
)

router = APIRouter()


@router.get("", response_model=APIResponse[SessionListResponse])
async def list_sessions(user: CurrentUser) -> APIResponse[SessionListResponse]:
    """List all sessions."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()

    sessions = agent.sessions.list_sessions()

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
    agent = get_agent()

    # Check if session already exists
    existing = agent.sessions.get(request.key)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "SESSION_EXISTS", "message": f"Session '{request.key}' already exists"},
        )

    # Create session
    session = agent.sessions.get_or_create(request.key)

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


@router.get("/{session_key}", response_model=APIResponse[SessionResponse])
async def get_session(
    session_key: str,
    user: CurrentUser,
) -> APIResponse[SessionResponse]:
    """Get session details."""
    request_id = f"req_{uuid4().hex[:12]}"
    agent = get_agent()

    session = agent.sessions.get(session_key)
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


@router.delete("/{session_key}")
async def delete_session(
    session_key: str,
    user: CurrentUser,
) -> dict:
    """Delete a session."""
    agent = get_agent()

    deleted = agent.sessions.delete(session_key)

    return {"deleted": deleted}


@router.post("/{session_key}/clear")
async def clear_session(
    session_key: str,
    user: CurrentUser,
) -> dict:
    """Clear session history."""
    agent = get_agent()

    session = agent.sessions.get(session_key)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"Session '{session_key}' not found"},
        )

    count = len(session.messages)
    session.messages.clear()
    agent.sessions.save(session)

    return {"cleared": True, "messages_removed": count}
