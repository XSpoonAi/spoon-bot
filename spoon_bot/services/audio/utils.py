"""Audio processing utilities."""

from __future__ import annotations

import base64
import struct
from logging import getLogger

logger = getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {"wav", "mp3", "ogg", "webm", "flac", "m4a", "aac", "mpeg", "pcm"}

# MIME type to format mapping
MIME_TO_FORMAT = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/ogg": "ogg",
    "audio/webm": "webm",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/m4a": "m4a",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/pcm": "pcm",
}

# Magic bytes for audio format detection
_MAGIC_BYTES = {
    b"RIFF": "wav",
    b"\xff\xfb": "mp3",
    b"\xff\xf3": "mp3",
    b"\xff\xf2": "mp3",
    b"ID3": "mp3",
    b"OggS": "ogg",
    b"fLaC": "flac",
    b"\x1aE\xdf\xa3": "webm",
}

# Maximum audio file size: 25MB (Whisper API limit)
MAX_AUDIO_SIZE = 25 * 1024 * 1024


def detect_audio_format(data: bytes) -> str | None:
    """Detect audio format from file magic bytes.

    Args:
        data: Raw audio bytes (at least first 12 bytes needed).

    Returns:
        Detected format string or None if unrecognized.
    """
    if len(data) < 4:
        return None

    # Check magic bytes
    for magic, fmt in _MAGIC_BYTES.items():
        if data[:len(magic)] == magic:
            # Extra check for WAV: must also have WAVE at offset 8
            if magic == b"RIFF" and len(data) >= 12:
                if data[8:12] != b"WAVE":
                    continue
            return fmt

    # Check for M4A/AAC (ftyp box)
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "m4a"

    return None


def mime_to_format(mime_type: str) -> str:
    """Convert MIME type to audio format string.

    Args:
        mime_type: MIME type string (e.g., 'audio/wav').

    Returns:
        Format string (e.g., 'wav').

    Raises:
        ValueError: If MIME type is not recognized.
    """
    fmt = MIME_TO_FORMAT.get(mime_type.lower())
    if fmt is None:
        # Try stripping parameters (e.g., 'audio/ogg; codecs=opus')
        base_mime = mime_type.split(";")[0].strip().lower()
        fmt = MIME_TO_FORMAT.get(base_mime)
    if fmt is None:
        raise ValueError(f"Unsupported audio MIME type: {mime_type}")
    return fmt


def validate_audio_data(
    data: bytes,
    max_size: int = MAX_AUDIO_SIZE,
) -> tuple[bool, str]:
    """Validate audio data.

    Args:
        data: Raw audio bytes.
        max_size: Maximum allowed size in bytes.

    Returns:
        Tuple of (is_valid, error_message). error_message is empty if valid.
    """
    if not data:
        return False, "Audio data is empty"

    if len(data) > max_size:
        size_mb = len(data) / (1024 * 1024)
        limit_mb = max_size / (1024 * 1024)
        return False, f"Audio data too large: {size_mb:.1f}MB (limit: {limit_mb:.0f}MB)"

    return True, ""


def decode_audio_base64(b64_data: str) -> bytes:
    """Decode base64-encoded audio data.

    Handles both standard base64 and data URLs (e.g., 'data:audio/wav;base64,...').

    Args:
        b64_data: Base64 string or data URL.

    Returns:
        Decoded audio bytes.

    Raises:
        ValueError: If base64 data is invalid.
    """
    # Strip data URL prefix if present
    if b64_data.startswith("data:"):
        parts = b64_data.split(",", 1)
        if len(parts) != 2:
            raise ValueError("Invalid data URL format")
        b64_data = parts[1]

    # Strip whitespace
    b64_data = b64_data.strip()

    try:
        return base64.b64decode(b64_data)
    except Exception as e:
        raise ValueError(f"Invalid base64 audio data: {e}") from e


def estimate_duration(data: bytes, audio_format: str) -> float | None:
    """Estimate audio duration from raw data.

    Only supports WAV format for exact calculation.
    Returns None for other formats.

    Args:
        data: Raw audio bytes.
        audio_format: Audio format string.

    Returns:
        Estimated duration in seconds, or None if cannot determine.
    """
    if audio_format == "wav" and len(data) >= 44:
        try:
            # WAV header: bytes 24-27 = sample rate, bytes 34-35 = bits per sample
            # bytes 22-23 = num channels, bytes 40-43 = data size
            sample_rate = struct.unpack_from("<I", data, 24)[0]
            num_channels = struct.unpack_from("<H", data, 22)[0]
            bits_per_sample = struct.unpack_from("<H", data, 34)[0]

            if sample_rate == 0 or num_channels == 0 or bits_per_sample == 0:
                return None

            # Find 'data' chunk
            offset = 12
            while offset < len(data) - 8:
                chunk_id = data[offset:offset + 4]
                chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
                if chunk_id == b"data":
                    bytes_per_sample = bits_per_sample // 8
                    bytes_per_second = sample_rate * num_channels * bytes_per_sample
                    if bytes_per_second > 0:
                        return chunk_size / bytes_per_second
                    return None
                offset += 8 + chunk_size

        except (struct.error, ZeroDivisionError):
            pass

    return None
