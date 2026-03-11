"""Utility modules for spoon-bot."""

from spoon_bot.utils.errors import (
    SpoonBotError,
    ConfigurationError,
    APIError,
    ToolExecutionError,
    RateLimitExceeded,
    format_user_error,
)
from spoon_bot.utils.rate_limit import (
    RateLimiter,
    TokenBucketLimiter,
    SlidingWindowLimiter,
    RateLimitConfig,
)
from spoon_bot.utils.privacy import mask_secrets

__all__ = [
    # Errors
    "SpoonBotError",
    "ConfigurationError",
    "APIError",
    "ToolExecutionError",
    "RateLimitExceeded",
    "format_user_error",
    # Rate Limiting
    "RateLimiter",
    "TokenBucketLimiter",
    "SlidingWindowLimiter",
    "RateLimitConfig",
    # Privacy
    "mask_secrets",
]
