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
class BudgetConfig:
    """Execution budget configuration."""

    request_timeout_ms: int = 120_000   # 2 minutes default
    tool_timeout_ms: int = 60_000       # 1 minute default
    stream_timeout_ms: int = 300_000    # 5 minutes default


@dataclass
class AudioConfig:
    """Audio/voice input configuration."""

    enabled: bool = True
    stt_provider: str = "whisper"
    stt_model: str = "whisper-1"
    max_audio_size_mb: int = 25  # Whisper API limit
    max_audio_duration_seconds: int = 600  # 10 minutes
    supported_formats: list[str] = field(
        default_factory=lambda: ["wav", "mp3", "ogg", "webm", "flac", "m4a", "aac"]
    )
    streaming_chunk_duration_ms: int = 3000
    enable_streaming: bool = True
    default_language: str | None = None  # auto-detect if None
    # Providers that support native audio input (skip transcription)
    native_audio_providers: list[str] = field(
        default_factory=lambda: ["openai", "gemini"]
    )


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

    # Budget / timeouts
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    # Audio / voice input
    audio: AudioConfig = field(default_factory=AudioConfig)

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
            budget=BudgetConfig(
                request_timeout_ms=int(
                    os.environ.get("GATEWAY_TIMEOUT_REQUEST_MS", "120000")
                ),
                tool_timeout_ms=int(
                    os.environ.get("GATEWAY_TIMEOUT_TOOL_MS", "60000")
                ),
                stream_timeout_ms=int(
                    os.environ.get("GATEWAY_TIMEOUT_STREAM_MS", "300000")
                ),
            ),
            audio=AudioConfig(
                enabled=os.environ.get("GATEWAY_AUDIO_ENABLED", "true").lower() == "true",
                stt_provider=os.environ.get("GATEWAY_AUDIO_STT_PROVIDER", "whisper"),
                stt_model=os.environ.get("GATEWAY_AUDIO_STT_MODEL", "whisper-1"),
                default_language=os.environ.get("GATEWAY_AUDIO_DEFAULT_LANGUAGE"),
                enable_streaming=os.environ.get("GATEWAY_AUDIO_STREAMING", "true").lower() == "true",
                native_audio_providers=[
                    p.strip()
                    for p in os.environ.get(
                        "GATEWAY_AUDIO_NATIVE_PROVIDERS", "openai,gemini"
                    ).split(",")
                    if p.strip()
                ],
            ),
        )
