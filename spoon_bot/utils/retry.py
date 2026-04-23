"""Provider-level retry with exponential backoff.

Wraps async callables to handle transient LLM provider errors:
  - Rate limits (HTTP 429)
  - Timeouts
  - Connection errors
  - Server errors (HTTP 5xx)

Non-retryable errors (auth failures, context overflow, etc.) are raised
immediately without consuming retry budget.
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar

from loguru import logger

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Patterns that signal a transient / retryable provider failure
# ---------------------------------------------------------------------------

_RATE_LIMIT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
    re.compile(r"\b429\b"),
    re.compile(r"quota.?exceeded", re.IGNORECASE),
    re.compile(r"resource.?exhausted", re.IGNORECASE),
    re.compile(r"tokens per min", re.IGNORECASE),
    re.compile(r"requests per min", re.IGNORECASE),
]

_TRANSIENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b50[023]\b"),
    re.compile(r"server.?error", re.IGNORECASE),
    re.compile(r"internal.?error", re.IGNORECASE),
    re.compile(r"service.?unavailable", re.IGNORECASE),
    re.compile(r"bad.?gateway", re.IGNORECASE),
    re.compile(r"gateway.?timeout", re.IGNORECASE),
    re.compile(r"overloaded", re.IGNORECASE),
    re.compile(r"temporarily", re.IGNORECASE),
    re.compile(r"an error occurred while processing your request", re.IGNORECASE),
    re.compile(r"ECONNRESET", re.IGNORECASE),
    re.compile(r"ECONNREFUSED", re.IGNORECASE),
]

_NON_RETRYABLE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b401\b"),
    re.compile(r"auth.*fail", re.IGNORECASE),
    re.compile(r"invalid.*api.?key", re.IGNORECASE),
    re.compile(r"context.*length.*exceeded", re.IGNORECASE),
    re.compile(r"maximum.*context", re.IGNORECASE),
    re.compile(r"content.*filter", re.IGNORECASE),
    re.compile(r"content.*policy", re.IGNORECASE),
]

_CONTEXT_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"context.*length.*exceeded", re.IGNORECASE),
    re.compile(r"maximum.*context", re.IGNORECASE),
    re.compile(r"context.*overflow", re.IGNORECASE),
    re.compile(r"prompt.*too long", re.IGNORECASE),
    re.compile(r"too many input tokens", re.IGNORECASE),
    re.compile(r"request.*too large", re.IGNORECASE),
]

_RETRYABLE_EXCEPTION_NAMES: set[str] = {
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "OverloadedError",
    "RateLimitError",
    "ResourceExhausted",
    "ServerError",
    "ServiceUnavailableError",
    "TooManyRequestsError",
    "UnavailableError",
}

_NON_RETRYABLE_EXCEPTION_NAMES: set[str] = {
    "AuthenticationError",
    "BadRequestError",
    "ContextLengthExceededError",
    "InvalidRequestError",
    "NotFoundError",
    "PermissionDeniedError",
    "UnauthorizedError",
    "UnprocessableEntityError",
}

_CONTEXT_OVERFLOW_EXCEPTION_NAMES: set[str] = {
    "ContextLengthExceededError",
}


@dataclass
class RetryConfig:
    """Configuration for provider retry behaviour.

    Authoritative defaults live in ``spoon_bot.config.DEFAULT_PROVIDER_*``
    constants and flow through ``AgentLoopConfig`` → ``RetryConfig`` at
    runtime.  The literal values here match those constants as a convenience
    for standalone / test usage.
    """

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: float = 0.5

    def delay_for_attempt(self, attempt: int) -> float:
        """Compute delay with exponential backoff + jitter.

        attempt 0 → ~1s, 1 → ~2s, 2 → ~4s, 3 → ~8s, 4 → ~16s
        """
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        jitter_range = delay * self.jitter
        delay += random.uniform(-jitter_range, jitter_range)
        return max(0.1, min(delay, self.max_delay))


# Singleton default
DEFAULT_RETRY_CONFIG = RetryConfig()


def _extract_status_code(exc: Exception) -> int | None:
    """Try to extract an HTTP status code from common exception shapes."""
    for attr in ("status_code", "status", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val

    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            val = getattr(response, attr, None)
            if isinstance(val, int):
                return val

    return None


def _extract_retry_after(exc: Exception) -> float | None:
    """Try to pull a Retry-After hint from the exception or its response."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    return None


def is_retryable(exc: Exception) -> bool:
    """Decide whether *exc* represents a transient provider failure."""
    from spoon_bot.exceptions import (
        LLMConnectionError,
        LLMTimeoutError,
        LLMRateLimitError,
        ContextOverflowError,
        APIKeyMissingError,
    )

    if is_context_overflow_error(exc):
        return False

    if isinstance(exc, (APIKeyMissingError, ContextOverflowError)):
        return False

    if isinstance(exc, (LLMRateLimitError, LLMTimeoutError, LLMConnectionError)):
        if isinstance(exc, LLMConnectionError) and exc.status_code == 401:
            return False
        return True

    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return True

    if isinstance(exc, OSError) and not isinstance(
        exc, (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)
    ):
        return True

    status = _extract_status_code(exc)
    if status is not None:
        if status == 401:
            return False
        if status == 429 or status >= 500:
            return True

    err_str = str(exc)
    exc_name = type(exc).__name__
    exc_module = (type(exc).__module__ or "").lower()

    for pat in _NON_RETRYABLE_PATTERNS:
        if pat.search(err_str):
            return False

    if exc_name in _NON_RETRYABLE_EXCEPTION_NAMES:
        return False

    if exc_name == "APIError" and exc_module.startswith("openai"):
        return True

    if exc_name in _RETRYABLE_EXCEPTION_NAMES:
        return True

    for pat in _RATE_LIMIT_PATTERNS:
        if pat.search(err_str):
            return True
    for pat in _TRANSIENT_PATTERNS:
        if pat.search(err_str):
            return True

    return False


def is_context_overflow_error(exc: Exception) -> bool:
    """Decide whether *exc* signals a recoverable context-overflow failure."""
    from spoon_bot.exceptions import ContextOverflowError

    if isinstance(exc, ContextOverflowError):
        return True

    status = _extract_status_code(exc)
    if status == 413:
        return True

    exc_name = type(exc).__name__
    if exc_name in _CONTEXT_OVERFLOW_EXCEPTION_NAMES:
        return True

    err_str = str(exc)
    for pat in _CONTEXT_OVERFLOW_PATTERNS:
        if pat.search(err_str):
            return True

    return False


async def with_provider_retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    on_retry: Callable[[int, Exception, float], Any] | None = None,
    **kwargs: Any,
) -> T:
    """Execute *fn* with automatic retry on transient provider errors.

    Parameters
    ----------
    fn:
        The async callable to invoke.
    config:
        Retry configuration. Uses DEFAULT_RETRY_CONFIG when None.
    on_retry:
        Optional callback ``(attempt, exception, delay)`` called before
        each sleep.  Can be used for logging / metrics.

    Raises the last exception when all retries are exhausted, or immediately
    for non-retryable errors.
    """
    cfg = config or DEFAULT_RETRY_CONFIG
    last_exc: Exception | None = None

    for attempt in range(cfg.max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc

            if not is_retryable(exc):
                raise

            if attempt >= cfg.max_retries:
                logger.error(
                    f"Provider retry exhausted after {cfg.max_retries + 1} attempts: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise

            provider_delay = _extract_retry_after(exc)
            if provider_delay is None:
                exc_retry_after = getattr(exc, "retry_after", None)
                if isinstance(exc_retry_after, (int, float)) and exc_retry_after > 0:
                    provider_delay = float(exc_retry_after)
            computed_delay = cfg.delay_for_attempt(attempt)
            delay = max(computed_delay, provider_delay or 0.0)

            if on_retry:
                on_retry(attempt, exc, delay)
            else:
                logger.warning(
                    f"Provider transient error (attempt {attempt + 1}/{cfg.max_retries + 1}), "
                    f"retrying in {delay:.1f}s: {type(exc).__name__}: {exc}"
                )

            await asyncio.sleep(delay)

    assert last_exc is not None
    raise last_exc
