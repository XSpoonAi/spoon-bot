"""Factory for creating audio transcriber instances."""

from __future__ import annotations

from spoon_bot.services.audio.base import AudioTranscriber
from spoon_bot.services.audio.whisper import WhisperTranscriber


_PROVIDERS = {
    "whisper": WhisperTranscriber,
    "openai": WhisperTranscriber,  # alias
}


def create_transcriber(
    provider: str = "whisper",
    **kwargs,
) -> AudioTranscriber:
    """Create an audio transcriber instance.

    Args:
        provider: Transcriber provider name ('whisper', 'openai').
        **kwargs: Provider-specific configuration.

    Returns:
        AudioTranscriber instance.

    Raises:
        ValueError: If provider is not supported.
    """
    provider_lower = provider.lower()
    cls = _PROVIDERS.get(provider_lower)
    if cls is None:
        available = ", ".join(sorted(_PROVIDERS.keys()))
        raise ValueError(
            f"Unsupported STT provider: '{provider}'. Available: {available}"
        )
    return cls(**kwargs)
