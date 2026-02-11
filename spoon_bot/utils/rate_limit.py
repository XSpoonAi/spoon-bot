"""Rate limiting utilities for spoon-bot.

Provides configurable rate limiting for:
- LLM API calls (to prevent quota exhaustion)
- Shell command execution (to prevent system abuse)
- Any other resource that needs throttling

Supports multiple algorithms:
- Token bucket (allows bursts)
- Sliding window (strict limits)
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from loguru import logger


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_second: Maximum requests per second (default 10).
        requests_per_minute: Maximum requests per minute (default 60).
        burst_size: Maximum burst size for token bucket (default 5).
        enabled: Whether rate limiting is enabled (default True).
    """

    requests_per_second: float = 10.0
    requests_per_minute: float = 60.0
    burst_size: int = 5
    enabled: bool = True

    # Preset configurations
    @classmethod
    def for_llm_api(cls) -> "RateLimitConfig":
        """Rate limit config for LLM API calls (conservative)."""
        return cls(
            requests_per_second=2.0,
            requests_per_minute=30.0,
            burst_size=3,
        )

    @classmethod
    def for_shell(cls) -> "RateLimitConfig":
        """Rate limit config for shell commands."""
        return cls(
            requests_per_second=5.0,
            requests_per_minute=100.0,
            burst_size=10,
        )

    @classmethod
    def for_web_requests(cls) -> "RateLimitConfig":
        """Rate limit config for web requests."""
        return cls(
            requests_per_second=3.0,
            requests_per_minute=50.0,
            burst_size=5,
        )

    @classmethod
    def unlimited(cls) -> "RateLimitConfig":
        """No rate limiting."""
        return cls(enabled=False)


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire permission to proceed with an operation.

        Args:
            tokens: Number of tokens/requests to acquire.

        Returns:
            True if acquired, False if rate limited.
        """
        pass

    @abstractmethod
    async def wait_and_acquire(self, tokens: int = 1) -> float:
        """
        Wait until rate limit allows and then acquire.

        Args:
            tokens: Number of tokens/requests to acquire.

        Returns:
            Time waited in seconds.
        """
        pass

    @abstractmethod
    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Get estimated wait time for acquiring tokens.

        Args:
            tokens: Number of tokens/requests.

        Returns:
            Estimated wait time in seconds, 0 if available now.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the rate limiter state."""
        pass


@dataclass
class TokenBucketLimiter(RateLimiter):
    """
    Token bucket rate limiter.

    Allows bursts up to bucket capacity, then rate limits.
    Good for APIs that allow short bursts but have per-minute limits.
    """

    rate: float  # Tokens per second
    capacity: float  # Maximum tokens (burst size)
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        self.tokens = self.capacity
        self.last_update = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens without waiting."""
        async with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_and_acquire(self, tokens: int = 1) -> float:
        """Wait until tokens available and acquire."""
        start_time = time.monotonic()

        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return time.monotonic() - start_time

                # Calculate wait time
                needed = tokens - self.tokens
                wait_time = needed / self.rate

            await asyncio.sleep(min(wait_time, 0.1))  # Check every 100ms max

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time."""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.rate

    def reset(self) -> None:
        """Reset to full capacity."""
        self.tokens = self.capacity
        self.last_update = time.monotonic()

    @classmethod
    def from_config(cls, config: RateLimitConfig, name: str = "default") -> "TokenBucketLimiter":
        """Create limiter from config."""
        if not config.enabled:
            # Return a permissive limiter
            return cls(rate=float("inf"), capacity=float("inf"))
        return cls(
            rate=config.requests_per_second,
            capacity=float(config.burst_size),
        )


@dataclass
class SlidingWindowLimiter(RateLimiter):
    """
    Sliding window rate limiter.

    Tracks exact timestamps of requests within the window.
    More accurate than token bucket but uses more memory.
    """

    limit: int  # Maximum requests in window
    window: float  # Window size in seconds
    timestamps: list[float] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def _cleanup(self) -> None:
        """Remove expired timestamps."""
        now = time.monotonic()
        cutoff = now - self.window
        self.timestamps = [ts for ts in self.timestamps if ts > cutoff]

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire without waiting."""
        async with self._lock:
            self._cleanup()
            if len(self.timestamps) + tokens <= self.limit:
                now = time.monotonic()
                for _ in range(tokens):
                    self.timestamps.append(now)
                return True
            return False

    async def wait_and_acquire(self, tokens: int = 1) -> float:
        """Wait until rate limit allows and acquire."""
        start_time = time.monotonic()

        while True:
            async with self._lock:
                self._cleanup()
                if len(self.timestamps) + tokens <= self.limit:
                    now = time.monotonic()
                    for _ in range(tokens):
                        self.timestamps.append(now)
                    return time.monotonic() - start_time

                # Calculate wait time until oldest expires
                if self.timestamps:
                    oldest = self.timestamps[0]
                    wait_time = (oldest + self.window) - time.monotonic()
                else:
                    wait_time = 0.1

            await asyncio.sleep(max(0.01, min(wait_time, 0.1)))

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get estimated wait time."""
        self._cleanup()
        if len(self.timestamps) + tokens <= self.limit:
            return 0.0
        if self.timestamps:
            oldest = self.timestamps[0]
            return max(0, (oldest + self.window) - time.monotonic())
        return 0.0

    def reset(self) -> None:
        """Clear all timestamps."""
        self.timestamps.clear()

    @classmethod
    def from_config(cls, config: RateLimitConfig, name: str = "default") -> "SlidingWindowLimiter":
        """Create limiter from config."""
        if not config.enabled:
            return cls(limit=999999, window=1.0)
        return cls(
            limit=int(config.requests_per_minute),
            window=60.0,
        )


class RateLimiterRegistry:
    """Registry for managing multiple rate limiters."""

    def __init__(self):
        self._limiters: dict[str, RateLimiter] = {}
        self._configs: dict[str, RateLimitConfig] = {}

    def register(
        self,
        name: str,
        config: RateLimitConfig,
        limiter_type: str = "token_bucket",
    ) -> RateLimiter:
        """
        Register a rate limiter with given config.

        Args:
            name: Unique identifier for the limiter.
            config: Rate limit configuration.
            limiter_type: "token_bucket" or "sliding_window".

        Returns:
            The created rate limiter.
        """
        self._configs[name] = config

        if limiter_type == "sliding_window":
            limiter = SlidingWindowLimiter.from_config(config, name)
        else:
            limiter = TokenBucketLimiter.from_config(config, name)

        self._limiters[name] = limiter
        logger.debug(f"Registered rate limiter: {name} ({limiter_type})")
        return limiter

    def get(self, name: str) -> RateLimiter | None:
        """Get a rate limiter by name."""
        return self._limiters.get(name)

    def get_or_create(
        self,
        name: str,
        config: RateLimitConfig | None = None,
    ) -> RateLimiter:
        """Get existing or create new rate limiter."""
        if name in self._limiters:
            return self._limiters[name]

        config = config or RateLimitConfig()
        return self.register(name, config)

    def reset_all(self) -> None:
        """Reset all rate limiters."""
        for limiter in self._limiters.values():
            limiter.reset()

    def remove(self, name: str) -> bool:
        """Remove a rate limiter."""
        if name in self._limiters:
            del self._limiters[name]
            if name in self._configs:
                del self._configs[name]
            return True
        return False


# Global registry instance
_global_registry = RateLimiterRegistry()


def get_rate_limiter(name: str, config: RateLimitConfig | None = None) -> RateLimiter:
    """Get or create a rate limiter from the global registry."""
    return _global_registry.get_or_create(name, config)


def reset_all_limiters() -> None:
    """Reset all rate limiters in the global registry."""
    _global_registry.reset_all()


# Type variable for decorated functions
T = TypeVar("T")


def rate_limited(
    limiter_name: str,
    config: RateLimitConfig | None = None,
    wait: bool = True,
) -> Callable:
    """
    Decorator to apply rate limiting to async functions.

    Args:
        limiter_name: Name of the rate limiter to use.
        config: Optional config for creating new limiter.
        wait: If True, wait for rate limit. If False, raise error.

    Returns:
        Decorated function.

    Example:
        @rate_limited("llm_api", RateLimitConfig.for_llm_api())
        async def call_llm():
            ...
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            limiter = get_rate_limiter(limiter_name, config)

            if wait:
                wait_time = await limiter.wait_and_acquire()
                if wait_time > 0.1:
                    logger.debug(f"Rate limited {limiter_name}: waited {wait_time:.2f}s")
            else:
                acquired = await limiter.acquire()
                if not acquired:
                    from spoon_bot.utils.errors import RateLimitExceeded
                    raise RateLimitExceeded(
                        resource=limiter_name,
                        limit=config.requests_per_minute if config else 60,
                        window=60.0,
                        retry_after=limiter.get_wait_time(),
                    )

            return await func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
