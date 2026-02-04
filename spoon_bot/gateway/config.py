"""Gateway configuration."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CORSConfig:
    """CORS configuration."""

    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_methods: list[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
    )
    allow_headers: list[str] = field(
        default_factory=lambda: ["Authorization", "X-API-Key", "Content-Type"]
    )
    allow_credentials: bool = True
    max_age: int = 600


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""

    enabled: bool = True
    chat_requests_per_minute: int = 60
    tool_requests_per_minute: int = 30
    auth_requests_per_minute: int = 5
    websocket_messages_per_minute: int = 100


@dataclass
class JWTConfig:
    """JWT configuration."""

    secret_key: str = field(default_factory=lambda: os.environ.get("JWT_SECRET", ""))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    def __post_init__(self):
        if not self.secret_key:
            # Generate a random secret if not provided (dev mode)
            self.secret_key = secrets.token_urlsafe(32)


@dataclass
class GatewayConfig:
    """Gateway configuration."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080
    debug: bool = False

    # API settings
    api_prefix: str = "/v1"
    docs_url: str | None = "/docs"
    redoc_url: str | None = "/redoc"

    # Authentication
    jwt: JWTConfig = field(default_factory=JWTConfig)
    api_keys: dict[str, str] = field(default_factory=dict)  # key -> user_id

    # CORS
    cors: CORSConfig = field(default_factory=CORSConfig)

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Agent settings
    default_session_key: str = "default"
    max_message_length: int = 100000
    max_media_items: int = 10

    # WebSocket settings
    websocket_ping_interval: int = 30
    websocket_ping_timeout: int = 10

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Create config from environment variables."""
        return cls(
            host=os.environ.get("GATEWAY_HOST", "127.0.0.1"),
            port=int(os.environ.get("GATEWAY_PORT", "8080")),
            debug=os.environ.get("GATEWAY_DEBUG", "").lower() == "true",
            jwt=JWTConfig(
                secret_key=os.environ.get("JWT_SECRET", ""),
                access_token_expire_minutes=int(
                    os.environ.get("JWT_ACCESS_EXPIRE_MINUTES", "15")
                ),
            ),
        )
