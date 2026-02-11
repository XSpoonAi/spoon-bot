"""API key authentication."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from spoon_bot.gateway.config import GatewayConfig


@dataclass
class APIKeyData:
    """API key data."""

    key_id: str
    user_id: str
    environment: str  # "live", "test", "dev"
    scopes: list[str]


def generate_api_key(environment: str = "live") -> tuple[str, str]:
    """
    Generate a new API key.

    Args:
        environment: Key environment (live, test, dev).

    Returns:
        Tuple of (api_key, key_hash).
    """
    random_part = secrets.token_urlsafe(24)  # 32 chars
    api_key = f"sk_{environment}_{random_part}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return api_key, key_hash


def verify_api_key(
    api_key: str,
    config: "GatewayConfig",
) -> APIKeyData | None:
    """
    Verify an API key.

    Args:
        api_key: API key string (format: sk_<env>_<key>).
        config: Gateway configuration containing valid keys.

    Returns:
        APIKeyData if valid, None if invalid.
    """
    # Validate format
    if not api_key or not api_key.startswith("sk_"):
        return None

    parts = api_key.split("_", 2)
    if len(parts) != 3:
        return None

    _, environment, key_part = parts

    if environment not in ("live", "test", "dev"):
        logger.warning(f"Invalid API key environment: {environment}")
        return None

    # Look up in configured keys
    user_id = config.api_keys.get(api_key)
    if not user_id:
        logger.debug("API key not found in configuration")
        return None

    return APIKeyData(
        key_id=hashlib.sha256(api_key.encode()).hexdigest()[:16],
        user_id=user_id,
        environment=environment,
        scopes=["agent:read", "agent:write"],  # Default scopes
    )
