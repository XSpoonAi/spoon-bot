"""JWT token handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt
from loguru import logger


@dataclass
class TokenData:
    """Decoded token data."""

    user_id: str
    session_key: str
    token_type: str  # "access" or "refresh"
    scopes: list[str]
    issued_at: datetime
    expires_at: datetime
    token_id: str | None = None  # For refresh token revocation

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now(timezone.utc) > self.expires_at


def create_access_token(
    user_id: str,
    session_key: str,
    scopes: list[str],
    secret_key: str,
    algorithm: str = "HS256",
    expires_minutes: int = 15,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User identifier.
        session_key: Agent session key.
        scopes: Permission scopes.
        secret_key: JWT secret key.
        algorithm: JWT algorithm.
        expires_minutes: Token expiration in minutes.

    Returns:
        Encoded JWT token string.
    """
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=expires_minutes)

    payload = {
        "sub": user_id,
        "session": session_key,
        "type": "access",
        "scope": scopes,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def create_refresh_token(
    user_id: str,
    secret_key: str,
    algorithm: str = "HS256",
    expires_days: int = 7,
    token_id: str | None = None,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        user_id: User identifier.
        secret_key: JWT secret key.
        algorithm: JWT algorithm.
        expires_days: Token expiration in days.
        token_id: Optional token ID for revocation tracking.

    Returns:
        Encoded JWT token string.
    """
    import uuid

    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=expires_days)

    payload = {
        "sub": user_id,
        "type": "refresh",
        "jti": token_id or str(uuid.uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def verify_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
    expected_type: str | None = None,
) -> TokenData | None:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string.
        secret_key: JWT secret key.
        algorithm: JWT algorithm.
        expected_type: Expected token type ("access" or "refresh").

    Returns:
        TokenData if valid, None if invalid.
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])

        token_type = payload.get("type", "access")
        if expected_type and token_type != expected_type:
            logger.warning(f"Token type mismatch: expected {expected_type}, got {token_type}")
            return None

        return TokenData(
            user_id=payload["sub"],
            session_key=payload.get("session", "default"),
            token_type=token_type,
            scopes=payload.get("scope", []),
            issued_at=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            token_id=payload.get("jti"),
        )

    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid token: {e}")
        return None
