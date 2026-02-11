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
]
