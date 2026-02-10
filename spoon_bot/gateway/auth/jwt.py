"""JWT token handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from loguru import logger

_REVOKED_TOKEN_IDS: set[str] = set()
_REVOKED_TOKEN_HASHES: set[str] = set()


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


def revoke_token(token: str, token_id: str | None = None) -> None:
    """Revoke a token by raw token hash and/or token id."""
    if token:
        _REVOKED_TOKEN_HASHES.add(token)
    if token_id:
        _REVOKED_TOKEN_IDS.add(token_id)


def is_token_revoked(token: str, token_id: str | None = None) -> bool:
    """Check token revocation status."""
    if token in _REVOKED_TOKEN_HASHES:
        return True
    if token_id and token_id in _REVOKED_TOKEN_IDS:
        return True
    return False


def create_access_token(
    user_id: str,
    session_key: str,
    scopes: list[str],
    secret_key: str,
    algorithm: str = "HS256",
    expires_minutes: int = 15,
) -> str:
    """Create a JWT access token."""
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
    """Create a JWT refresh token."""
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
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])

        token_type = payload.get("type", "access")
        if expected_type and token_type != expected_type:
            logger.warning(f"Token type mismatch: expected {expected_type}, got {token_type}")
            return None

        token_id = payload.get("jti")
        if is_token_revoked(token, token_id):
            logger.warning("Token is revoked")
            return None

        return TokenData(
            user_id=payload["sub"],
            session_key=payload.get("session", "default"),
            token_type=token_type,
            scopes=payload.get("scope", []),
            issued_at=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            token_id=token_id,
        )

    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid token: {e}")
        return None
