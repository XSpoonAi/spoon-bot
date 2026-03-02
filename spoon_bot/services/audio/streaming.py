"""Streaming audio recognition --- buffer and transcribe real-time audio chunks.

Handles the server side of the WebSocket audio streaming protocol:
1. Client sends audio.stream.start (JSON) with metadata
2. Client sends binary audio chunks
3. Client sends audio.stream.end (JSON) to finalize
4. Server transcribes accumulated audio and responds

Supports optional intermediate transcriptions for long audio streams.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from logging import getLogger
from typing import AsyncGenerator

from spoon_bot.services.audio.base import AudioTranscriber, TranscriptionResult
from spoon_bot.services.audio.utils import detect_audio_format, validate_audio_data

logger = getLogger(__name__)


@dataclass
class StreamSession:
    """Tracks state for a single audio streaming session."""

    session_id: str
    audio_format: str = "wav"
    language: str | None = None
    sample_rate: int = 16000
    channels: int = 1
    text_message: str = ""

    # Internal state
    chunks: list[bytes] = field(default_factory=list)
    total_bytes: int = 0
    started_at: float = field(default_factory=time.time)
    last_chunk_at: float = field(default_factory=time.time)
    is_active: bool = True

    @property
    def duration_seconds(self) -> float:
        """Elapsed time since stream started."""
        return time.time() - self.started_at

    @property
    def audio_data(self) -> bytes:
        """Concatenate all buffered chunks."""
        return b"".join(self.chunks)

    def add_chunk(self, data: bytes) -> None:
        """Buffer an audio chunk."""
        self.chunks.append(data)
        self.total_bytes += len(data)
        self.last_chunk_at = time.time()

    def reset(self) -> None:
        """Clear buffered audio (e.g., after intermediate transcription)."""
        self.chunks.clear()
        self.total_bytes = 0


class AudioStreamManager:
    """Manages multiple concurrent audio streaming sessions.

    Each WebSocket connection can have at most one active stream session.
    """

    def __init__(
        self,
        transcriber: AudioTranscriber,
        max_stream_duration_seconds: int = 600,
        max_audio_size_bytes: int = 25 * 1024 * 1024,
        intermediate_chunk_seconds: float = 0,
    ):
        self._transcriber = transcriber
        self._max_duration = max_stream_duration_seconds
        self._max_size = max_audio_size_bytes
        self._intermediate_interval = intermediate_chunk_seconds
        self._sessions: dict[str, StreamSession] = {}

    def start_session(
        self,
        connection_id: str,
        audio_format: str = "wav",
        language: str | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        text_message: str = "",
    ) -> StreamSession:
        """Start a new audio streaming session for a connection.

        Args:
            connection_id: WebSocket connection identifier.
            audio_format: Expected audio format of incoming chunks.
            language: Optional language hint.
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels.
            text_message: Optional accompanying text message.

        Returns:
            New StreamSession.

        Raises:
            ValueError: If a session is already active for this connection.
        """
        if connection_id in self._sessions and self._sessions[connection_id].is_active:
            raise ValueError(f"Audio stream already active for connection {connection_id}")

        session = StreamSession(
            session_id=connection_id,
            audio_format=audio_format,
            language=language,
            sample_rate=sample_rate,
            channels=channels,
            text_message=text_message,
        )
        self._sessions[connection_id] = session
        logger.info(
            f"Audio stream started: conn={connection_id}, format={audio_format}, "
            f"rate={sample_rate}, channels={channels}"
        )
        return session

    def add_chunk(self, connection_id: str, data: bytes) -> None:
        """Add an audio chunk to an active session.

        Args:
            connection_id: WebSocket connection identifier.
            data: Raw audio bytes.

        Raises:
            ValueError: If no active session exists or limits exceeded.
        """
        session = self._sessions.get(connection_id)
        if session is None or not session.is_active:
            raise ValueError(f"No active audio stream for connection {connection_id}")

        # Check size limit
        if session.total_bytes + len(data) > self._max_size:
            raise ValueError(
                f"Audio stream size limit exceeded: "
                f"{(session.total_bytes + len(data)) / (1024 * 1024):.1f}MB > "
                f"{self._max_size / (1024 * 1024):.0f}MB"
            )

        # Check duration limit
        if session.duration_seconds > self._max_duration:
            raise ValueError(
                f"Audio stream duration limit exceeded: "
                f"{session.duration_seconds:.0f}s > {self._max_duration}s"
            )

        session.add_chunk(data)

    async def end_session(
        self, connection_id: str
    ) -> TranscriptionResult:
        """End an audio stream session and transcribe the accumulated audio.

        Args:
            connection_id: WebSocket connection identifier.

        Returns:
            TranscriptionResult with full transcription.

        Raises:
            ValueError: If no active session exists.
        """
        session = self._sessions.get(connection_id)
        if session is None:
            raise ValueError(f"No audio stream session for connection {connection_id}")

        session.is_active = False
        audio_data = session.audio_data

        logger.info(
            f"Audio stream ended: conn={connection_id}, "
            f"size={len(audio_data) / 1024:.1f}KB, "
            f"duration={session.duration_seconds:.1f}s"
        )

        if not audio_data:
            self._sessions.pop(connection_id, None)
            return TranscriptionResult(
                text="",
                language=session.language,
                duration_seconds=session.duration_seconds,
                provider="none",
            )

        # Validate
        is_valid, error = validate_audio_data(audio_data, self._max_size)
        if not is_valid:
            self._sessions.pop(connection_id, None)
            raise ValueError(f"Invalid audio data: {error}")

        # Auto-detect format if possible
        audio_format = session.audio_format
        detected = detect_audio_format(audio_data)
        if detected:
            audio_format = detected

        try:
            result = await self._transcriber.transcribe(
                audio_data=audio_data,
                audio_format=audio_format,
                language=session.language,
            )
            return result
        finally:
            self._sessions.pop(connection_id, None)

    def cancel_session(self, connection_id: str) -> None:
        """Cancel and clean up a streaming session."""
        session = self._sessions.pop(connection_id, None)
        if session:
            session.is_active = False
            logger.info(f"Audio stream cancelled: conn={connection_id}")

    def get_session(self, connection_id: str) -> StreamSession | None:
        """Get the active session for a connection, if any."""
        return self._sessions.get(connection_id)

    def has_active_session(self, connection_id: str) -> bool:
        """Check if a connection has an active audio stream."""
        session = self._sessions.get(connection_id)
        return session is not None and session.is_active
