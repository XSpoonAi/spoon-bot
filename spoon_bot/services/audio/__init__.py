"""Audio services for speech recognition and processing."""

from spoon_bot.services.audio.base import AudioTranscriber, TranscriptionResult, AudioSegment
from spoon_bot.services.audio.whisper import WhisperTranscriber
from spoon_bot.services.audio.factory import create_transcriber
from spoon_bot.services.audio.utils import (
    detect_audio_format,
    validate_audio_data,
    decode_audio_base64,
    estimate_duration,
    SUPPORTED_FORMATS,
)

__all__ = [
    "AudioTranscriber",
    "TranscriptionResult",
    "AudioSegment",
    "WhisperTranscriber",
    "create_transcriber",
    "detect_audio_format",
    "validate_audio_data",
    "decode_audio_base64",
    "estimate_duration",
    "SUPPORTED_FORMATS",
]
