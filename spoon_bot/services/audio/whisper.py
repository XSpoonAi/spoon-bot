"""OpenAI Whisper API transcription provider."""

from __future__ import annotations

import io
import os
from logging import getLogger

from spoon_bot.services.audio.base import (
    AudioTranscriber,
    TranscriptionResult,
    AudioSegment,
)

logger = getLogger(__name__)

# Map audio format strings to file extensions for Whisper API
FORMAT_TO_EXTENSION = {
    "wav": "wav",
    "mp3": "mp3",
    "ogg": "ogg",
    "webm": "webm",
    "flac": "flac",
    "m4a": "m4a",
    "aac": "aac",
    "mpeg": "mp3",
    "pcm": "wav",
}

# Map format to MIME type for file upload
FORMAT_TO_MIME = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "webm": "audio/webm",
    "flac": "audio/flac",
    "m4a": "audio/m4a",
    "aac": "audio/aac",
    "mpeg": "audio/mpeg",
    "pcm": "audio/wav",
}


class WhisperTranscriber(AudioTranscriber):
    """Transcription using OpenAI's Whisper API.

    Requires OPENAI_API_KEY environment variable or explicit api_key.
    Supports: wav, mp3, ogg, webm, flac, m4a (up to 25MB).

    Note: Always uses the direct OpenAI API endpoint for Whisper,
    ignoring OPENAI_BASE_URL (which may point to a proxy like OpenRouter
    that doesn't support the /audio/transcriptions endpoint).
    """

    # The Whisper API is only available on the direct OpenAI endpoint.
    _OPENAI_API_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1",
        base_url: str | None = None,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        # Always default to direct OpenAI API to avoid OPENAI_BASE_URL
        # being picked up from environment (e.g. pointing to OpenRouter).
        self._base_url = base_url or self._OPENAI_API_URL
        self._client = None

    def _get_client(self):
        """Lazy-initialize the OpenAI async client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    async def transcribe(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio using OpenAI Whisper API.

        Args:
            audio_data: Raw audio bytes.
            audio_format: Audio format (wav, mp3, ogg, etc.).
            language: Optional ISO 639-1 language code.

        Returns:
            TranscriptionResult with transcribed text and metadata.
        """
        client = self._get_client()
        ext = FORMAT_TO_EXTENSION.get(audio_format, audio_format)
        filename = f"audio.{ext}"

        # Create file-like object for upload
        audio_file = io.BytesIO(audio_data)
        audio_file.name = filename

        kwargs = {
            "model": self._model,
            "file": audio_file,
            "response_format": "verbose_json",
        }
        if language:
            kwargs["language"] = language

        try:
            response = await client.audio.transcriptions.create(**kwargs)

            # Extract segments if available
            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    segments.append(
                        AudioSegment(
                            text=seg.get("text", "") if isinstance(seg, dict) else getattr(seg, "text", ""),
                            start=seg.get("start", 0.0) if isinstance(seg, dict) else getattr(seg, "start", 0.0),
                            end=seg.get("end", 0.0) if isinstance(seg, dict) else getattr(seg, "end", 0.0),
                            confidence=seg.get("avg_logprob") if isinstance(seg, dict) else getattr(seg, "avg_logprob", None),
                        )
                    )

            text = response.text if hasattr(response, "text") else str(response)
            detected_language = getattr(response, "language", None)
            duration = getattr(response, "duration", None)

            return TranscriptionResult(
                text=text,
                language=detected_language or language,
                duration_seconds=float(duration) if duration else None,
                segments=segments,
                provider="whisper",
            )

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise RuntimeError(f"Audio transcription failed: {e}") from e

    async def is_available(self) -> bool:
        """Check if Whisper API is available."""
        return bool(self._api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def provider_name(self) -> str:
        return "whisper"
