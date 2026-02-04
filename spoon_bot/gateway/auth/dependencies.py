"""FastAPI authentication dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from spoon_bot.gateway.app import get_config
from spoon_bot.gateway.auth.jwt import verify_token, TokenData
from spoon_bot.gateway.auth.api_key import verify_api_key, APIKeyData

# Bearer token security scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> TokenData | APIKeyData:
    """
    Get the current authenticated user.

    Supports both JWT Bearer tokens and API keys.

    Args:
        credentials: Bearer token credentials.
        x_api_key: API key from header.

    Returns:
        TokenData or APIKeyData for authenticated user.

    Raises:
        HTTPException: If authentication fails.
    """
    config = get_config()

    # Try API key first
    if x_api_key:
        api_key_data = verify_api_key(x_api_key, config)
        if api_key_data:
            return api_key_data

    # Try Bearer token
    if credentials:
        token_data = verify_token(
            credentials.credentials,
            config.jwt.secret_key,
            config.jwt.algorithm,
            expected_type="access",
        )
        if token_data:
            return token_data

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "code": "AUTH_REQUIRED",
            "message": "Authentication required. Provide Bearer token or X-API-Key header.",
        },
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> TokenData | APIKeyData | None:
    """
    Get the current user if authenticated, None otherwise.

    Same as get_current_user but doesn't raise on failure.
    """
    config = get_config()

    if x_api_key:
        api_key_data = verify_api_key(x_api_key, config)
        if api_key_data:
            return api_key_data

    if credentials:
        token_data = verify_token(
            credentials.credentials,
            config.jwt.secret_key,
            config.jwt.algorithm,
            expected_type="access",
        )
        if token_data:
            return token_data

    return None


def require_scope(required_scope: str):
    """
    Dependency factory to require a specific scope.

    Args:
        required_scope: Required permission scope.

    Returns:
        Dependency function.
    """

    async def check_scope(
        user: Annotated[TokenData | APIKeyData, Depends(get_current_user)]
    ) -> TokenData | APIKeyData:
        scopes = user.scopes if hasattr(user, "scopes") else []

        if required_scope not in scopes and "admin" not in scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "FORBIDDEN",
                    "message": f"Required scope '{required_scope}' not granted.",
                },
            )

        return user

    return check_scope


# Typed dependencies
CurrentUser = Annotated[TokenData | APIKeyData, Depends(get_current_user)]
OptionalUser = Annotated[TokenData | APIKeyData | None, Depends(get_current_user_optional)]
