"""Authentication endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from spoon_bot.gateway.app import get_config
from spoon_bot.gateway.auth.jwt import (
    create_access_token,
    create_refresh_token,
    verify_token,
)
from spoon_bot.gateway.auth.api_key import verify_api_key
from spoon_bot.gateway.auth.dependencies import CurrentUser
from spoon_bot.gateway.models.requests import LoginRequest, RefreshRequest
from spoon_bot.gateway.models.responses import TokenResponse

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """
    Authenticate and get tokens.

    Supports API key authentication.
    """
    config = get_config()
    user_id = None

    # Try API key authentication
    if request.api_key:
        api_key_data = verify_api_key(request.api_key, config)
        if api_key_data:
            user_id = api_key_data.user_id

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID", "message": "Invalid credentials"},
        )

    # Create tokens
    access_token = create_access_token(
        user_id=user_id,
        session_key="default",
        scopes=["agent:read", "agent:write"],
        secret_key=config.jwt.secret_key,
        algorithm=config.jwt.algorithm,
        expires_minutes=config.jwt.access_token_expire_minutes,
    )

    refresh_token = create_refresh_token(
        user_id=user_id,
        secret_key=config.jwt.secret_key,
        algorithm=config.jwt.algorithm,
        expires_days=config.jwt.refresh_token_expire_days,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=config.jwt.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest) -> TokenResponse:
    """Refresh access token using refresh token."""
    config = get_config()

    token_data = verify_token(
        request.refresh_token,
        config.jwt.secret_key,
        config.jwt.algorithm,
        expected_type="refresh",
    )

    if not token_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_INVALID", "message": "Invalid or expired refresh token"},
        )

    # Create new access token
    access_token = create_access_token(
        user_id=token_data.user_id,
        session_key="default",
        scopes=["agent:read", "agent:write"],
        secret_key=config.jwt.secret_key,
        algorithm=config.jwt.algorithm,
        expires_minutes=config.jwt.access_token_expire_minutes,
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=None,  # Don't issue new refresh token
        expires_in=config.jwt.access_token_expire_minutes * 60,
    )


@router.post("/logout")
async def logout(request: RefreshRequest) -> dict:
    """
    Logout and invalidate refresh token.

    Note: In production, store revoked tokens in a blacklist.
    """
    # TODO: Add token to blacklist
    return {"success": True}


@router.get("/verify")
async def verify(user: CurrentUser) -> dict:
    """Verify current authentication."""
    return {
        "valid": True,
        "user_id": user.user_id,
    }
