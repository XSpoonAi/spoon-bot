"""Audio processing pipeline — transcribe-or-passthrough middleware.

This pipeline sits between the gateway transports and the agent/LLM layer.
It decides whether audio should be:
1. Passed through as native AudioContent (for capable providers like GPT-4o, Gemini)
2. Transcribed to text via external STT (for providers like Anthropic Claude)
"""

from __future__ import annotations

import base64
from logging import getLogger
from typing import Any

from spoon_ai.schema import (
    AudioContent,
    AudioSource,
    ContentBlock,
    Message,
    TextContent,
)

from spoon_bot.services.audio.base import AudioTranscriber, TranscriptionResult
from spoon_bot.services.audio.factory import create_transcriber
from spoon_bot.services.audio.utils import (
    decode_audio_base64,
    detect_audio_format,
    mime_to_format,
    validate_audio_data,
)

logger = getLogger(__name__)

# Providers that handle audio content natively in their LLM APIs
NATIVE_AUDIO_PROVIDERS = frozenset({"openai", "gemini"})


class AudioPipeline:
    """Processes audio input into the appropriate format for the active LLM provider.

    Usage::

        pipeline = AudioPipeline(provider="anthropic")
        message, transcription = await pipeline.process(
            audio_data=raw_bytes,
            audio_format="wav",
            text="What does this say?",
        )
        # For Anthropic: message contains transcribed text, transcription has details
        # For OpenAI:    message contains AudioContent block, transcription is None
    """

    def __init__(
        self,
        provider: str = "anthropic",
        stt_provider: str = "whisper",
        stt_model: str = "whisper-1",
        native_audio_providers: frozenset[str] | None = None,
        **stt_kwargs: Any,
    ):
        self._provider = provider.lower()
        self._stt_provider = stt_provider
        self._stt_model = stt_model
        self._stt_kwargs = stt_kwargs
        self._native_providers = native_audio_providers or NATIVE_AUDIO_PROVIDERS
        self._transcriber: AudioTranscriber | None = None

    @property
    def supports_native_audio(self) -> bool:
        """Whether the current LLM provider can handle audio natively."""
        return self._provider in self._native_providers

    def _get_transcriber(self) -> AudioTranscriber:
        """Lazy-initialize the transcriber."""
        if self._transcriber is None:
            kwargs = dict(self._stt_kwargs)
            if self._stt_model:
                kwargs["model"] = self._stt_model
            self._transcriber = create_transcriber(self._stt_provider, **kwargs)
        return self._transcriber

    async def process(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        text: str = "",
        language: str | None = None,
    ) -> tuple[Message, TranscriptionResult | None]:
        """Process audio input into a Message suitable for the active LLM provider.

        Args:
            audio_data: Raw audio bytes.
            audio_format: Audio format (wav, mp3, ogg, etc.).
            text: Optional accompanying text message.
            language: Optional language hint (ISO 639-1).

        Returns:
            Tuple of (Message, TranscriptionResult or None).
            - For native-audio providers: Message has AudioContent block, transcription is None.
            - For other providers: Message has transcribed text, transcription has full result.
        """
        # Validate
        is_valid, error = validate_audio_data(audio_data)
        if not is_valid:
            raise ValueError(f"Invalid audio data: {error}")

        if self.supports_native_audio:
            return self._build_native_message(audio_data, audio_format, text, language), None
        else:
            return await self._transcribe_and_build(audio_data, audio_format, text, language)

    async def process_base64(
        self,
        b64_audio: str,
        audio_format: str | None = None,
        mime_type: str | None = None,
        text: str = "",
        language: str | None = None,
    ) -> tuple[Message, TranscriptionResult | None]:
        """Process base64-encoded audio input.

        Args:
            b64_audio: Base64-encoded audio data (or data URL).
            audio_format: Audio format (auto-detected if None).
            mime_type: MIME type (used for format detection if audio_format is None).
            text: Optional accompanying text.
            language: Optional language hint.

        Returns:
            Tuple of (Message, TranscriptionResult or None).
        """
        audio_bytes = decode_audio_base64(b64_audio)

        # Determine format
        if audio_format is None:
            if mime_type:
                audio_format = mime_to_format(mime_type)
            else:
                detected = detect_audio_format(audio_bytes)
                audio_format = detected or "wav"

        return await self.process(audio_bytes, audio_format, text, language)

    def _build_native_message(
        self,
        audio_data: bytes,
        audio_format: str,
        text: str,
        language: str | None,
    ) -> Message:
        """Build a multimodal Message with AudioContent for native providers."""
        b64_data = base64.b64encode(audio_data).decode("ascii")

        # Map format to MIME type
        mime_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
            "webm": "audio/webm",
            "flac": "audio/flac",
            "m4a": "audio/m4a",
            "aac": "audio/aac",
            "pcm": "audio/pcm",
        }
        media_type = mime_map.get(audio_format, f"audio/{audio_format}")

        return Message.create_with_audio(
            role="user",
            text=text,
            audio_data=b64_data,
            media_type=media_type,
            language=language,
        )

    async def _transcribe_and_build(
        self,
        audio_data: bytes,
        audio_format: str,
        text: str,
        language: str | None,
    ) -> tuple[Message, TranscriptionResult]:
        """Transcribe audio and build a text Message."""
        transcriber = self._get_transcriber()
        result = await transcriber.transcribe(audio_data, audio_format, language)

        if result.is_empty:
            logger.warning("Transcription returned empty text")
            combined = text or "[Empty audio — no speech detected]"
        else:
            if text:
                combined = f"{text}\n\n[Voice input]: {result.text}"
            else:
                combined = result.text

        message = Message.create_text("user", combined)
        return message, result
