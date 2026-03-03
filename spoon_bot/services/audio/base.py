"""Base classes and data models for audio transcription."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class AudioFormat(str, Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    WEBM = "webm"
    FLAC = "flac"
    M4A = "m4a"
    AAC = "aac"
    MPEG = "mpeg"
    PCM = "pcm"


@dataclass
class AudioSegment:
    """A segment of transcribed audio with timestamps."""
    text: str
    start: float  # seconds
    end: float  # seconds
    confidence: float | None = None


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    text: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[AudioSegment] = field(default_factory=list)
    confidence: float | None = None
    provider: str = "unknown"

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class AudioTranscriber(ABC):
    """Abstract base class for audio transcription providers."""

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio data to text.

        Args:
            audio_data: Raw audio bytes.
            audio_format: Audio format (wav, mp3, ogg, etc.).
            language: Optional language hint (ISO 639-1 code).

        Returns:
            TranscriptionResult with text and metadata.
        """
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this transcriber is available and configured."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the transcription provider."""
        ...
