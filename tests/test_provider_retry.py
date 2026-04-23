"""Tests for provider retry with exponential backoff (spoon_bot.utils.retry)."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from spoon_bot.utils.retry import (
    RetryConfig,
    is_context_overflow_error,
    is_retryable,
    with_provider_retry,
)
from spoon_bot.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMTimeoutError,
    ContextOverflowError,
    APIKeyMissingError,
)


# ---------------------------------------------------------------------------
# RetryConfig
# ---------------------------------------------------------------------------

class TestRetryConfig:
    def test_default_config(self):
        cfg = RetryConfig()
        assert cfg.max_retries == 5
        assert cfg.base_delay == 1.0
        assert cfg.max_delay == 60.0
        assert cfg.backoff_factor == 2.0

    def test_delay_increases_exponentially(self):
        cfg = RetryConfig(base_delay=1.0, backoff_factor=2.0, jitter=0.0)
        delays = [cfg.delay_for_attempt(i) for i in range(5)]
        assert delays == pytest.approx([1.0, 2.0, 4.0, 8.0, 16.0])

    def test_delay_capped_by_max_delay(self):
        cfg = RetryConfig(base_delay=10.0, backoff_factor=3.0, max_delay=20.0, jitter=0.0)
        assert cfg.delay_for_attempt(0) == 10.0
        assert cfg.delay_for_attempt(5) == 20.0  # capped

    def test_jitter_adds_randomness(self):
        cfg = RetryConfig(base_delay=1.0, jitter=0.5)
        delays = {cfg.delay_for_attempt(0) for _ in range(50)}
        assert len(delays) > 1  # should not all be identical


# ---------------------------------------------------------------------------
# is_retryable
# ---------------------------------------------------------------------------

class TestIsRetryable:
    def test_rate_limit_error_is_retryable(self):
        assert is_retryable(LLMRateLimitError("openai", retry_after=5.0)) is True

    def test_timeout_error_is_retryable(self):
        assert is_retryable(LLMTimeoutError("openai", 30.0)) is True

    def test_connection_error_is_retryable(self):
        assert is_retryable(LLMConnectionError("openai", status_code=503)) is True

    def test_asyncio_timeout_is_retryable(self):
        assert is_retryable(asyncio.TimeoutError()) is True

    def test_connection_refused_is_retryable(self):
        assert is_retryable(ConnectionRefusedError()) is True

    def test_auth_error_not_retryable(self):
        assert is_retryable(LLMConnectionError("openai", status_code=401)) is False

    def test_api_key_missing_not_retryable(self):
        assert is_retryable(APIKeyMissingError("openai", "OPENAI_API_KEY")) is False

    def test_context_overflow_not_retryable(self):
        assert is_retryable(ContextOverflowError(200_000, 128_000)) is False

    def test_generic_429_in_message_is_retryable(self):
        assert is_retryable(Exception("Error: 429 Too Many Requests")) is True

    def test_rate_limit_pattern_in_message(self):
        assert is_retryable(Exception("rate limit exceeded for this model")) is True

    def test_server_error_500_in_message(self):
        assert is_retryable(Exception("HTTP 500 internal server error")) is True

    def test_502_bad_gateway_retryable(self):
        assert is_retryable(Exception("HTTP 502 Bad Gateway")) is True

    def test_service_unavailable_retryable(self):
        assert is_retryable(Exception("service unavailable, try again later")) is True

    def test_overloaded_retryable(self):
        assert is_retryable(Exception("The server is overloaded")) is True

    def test_auth_fail_pattern_not_retryable(self):
        assert is_retryable(Exception("authentication failed")) is False

    def test_invalid_api_key_not_retryable(self):
        assert is_retryable(Exception("invalid API key provided")) is False

    def test_context_length_exceeded_not_retryable(self):
        assert is_retryable(Exception("context length exceeded")) is False

    def test_generic_value_error_not_retryable(self):
        assert is_retryable(ValueError("something went wrong")) is False

    def test_network_oserror_is_retryable(self):
        assert is_retryable(OSError("Network unreachable")) is True
        assert is_retryable(ConnectionResetError("reset")) is True

    def test_local_oserror_not_retryable(self):
        assert is_retryable(FileNotFoundError("/no/such/file")) is False
        assert is_retryable(PermissionError("denied")) is False
        assert is_retryable(IsADirectoryError("/tmp")) is False

    def test_status_code_attribute_detection(self):
        class FakeError(Exception):
            status_code = 429
        assert is_retryable(FakeError("oops")) is True

    def test_response_status_attribute_detection(self):
        class FakeResponse:
            status_code = 503
        class FakeError(Exception):
            response = FakeResponse()
        assert is_retryable(FakeError("oops")) is True

    def test_openai_apierror_without_status_is_retryable(self):
        class FakeOpenAIAPIError(Exception):
            __module__ = "openai"

        exc = FakeOpenAIAPIError(
            "An error occurred while processing your request. Please retry your request."
        )
        assert is_retryable(exc) is True

    def test_common_sdk_retryable_exception_names_are_retryable(self):
        class RateLimitError(Exception):
            pass

        class APIConnectionError(Exception):
            pass

        class ServiceUnavailableError(Exception):
            pass

        assert is_retryable(RateLimitError("slow down")) is True
        assert is_retryable(APIConnectionError("socket reset")) is True
        assert is_retryable(ServiceUnavailableError("please retry")) is True

    def test_common_sdk_non_retryable_exception_names_are_not_retryable(self):
        class AuthenticationError(Exception):
            pass

        class BadRequestError(Exception):
            pass

        assert is_retryable(AuthenticationError("bad key")) is False
        assert is_retryable(BadRequestError("invalid request")) is False


class TestIsContextOverflowError:
    def test_context_overflow_exception_is_detected(self):
        assert is_context_overflow_error(ContextOverflowError(200_000, 128_000)) is True

    def test_context_length_pattern_is_detected(self):
        assert is_context_overflow_error(Exception("context length exceeded for this model")) is True

    def test_request_too_large_tpm_error_is_treated_as_overflow_not_retryable(self):
        exc = Exception(
            "Request too large for gpt-5.4 (for limit gpt-5.4-long-context) "
            "on tokens per min (TPM): Limit 400000, Requested 860809."
        )

        assert is_context_overflow_error(exc) is True
        assert is_retryable(exc) is False

    def test_request_too_large_status_is_detected(self):
        class FakeError(Exception):
            status_code = 413

        assert is_context_overflow_error(FakeError("payload too large")) is True


# ---------------------------------------------------------------------------
# with_provider_retry
# ---------------------------------------------------------------------------

class TestWithProviderRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        fn = AsyncMock(return_value="result")
        result = await with_provider_retry(fn, config=RetryConfig(max_retries=3))
        assert result == "result"
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self):
        fn = AsyncMock(side_effect=[
            LLMRateLimitError("openai"),
            LLMRateLimitError("openai"),
            "success",
        ])
        cfg = RetryConfig(max_retries=5, base_delay=0.01, jitter=0.0)
        result = await with_provider_retry(fn, config=cfg)
        assert result == "success"
        assert fn.call_count == 3

    @pytest.mark.asyncio
    async def test_raises_immediately_on_non_retryable(self):
        fn = AsyncMock(side_effect=APIKeyMissingError("openai", "OPENAI_API_KEY"))
        cfg = RetryConfig(max_retries=5, base_delay=0.01)
        with pytest.raises(APIKeyMissingError):
            await with_provider_retry(fn, config=cfg)
        assert fn.call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_raises_last_error(self):
        fn = AsyncMock(side_effect=LLMTimeoutError("openai", 30.0))
        cfg = RetryConfig(max_retries=2, base_delay=0.01, jitter=0.0)
        with pytest.raises(LLMTimeoutError):
            await with_provider_retry(fn, config=cfg)
        assert fn.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_on_retry_callback_called(self):
        fn = AsyncMock(side_effect=[
            LLMRateLimitError("openai"),
            "ok",
        ])
        callback_calls = []
        cfg = RetryConfig(max_retries=3, base_delay=0.01, jitter=0.0)
        await with_provider_retry(
            fn,
            config=cfg,
            on_retry=lambda attempt, exc, delay: callback_calls.append((attempt, type(exc).__name__, delay)),
        )
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 0  # first retry attempt
        assert callback_calls[0][1] == "LLMRateLimitError"

    @pytest.mark.asyncio
    async def test_delay_increases_between_retries(self):
        fn = AsyncMock(side_effect=[
            LLMRateLimitError("openai"),
            LLMRateLimitError("openai"),
            LLMRateLimitError("openai"),
            "ok",
        ])
        delays = []
        cfg = RetryConfig(max_retries=5, base_delay=0.5, backoff_factor=2.0, jitter=0.0)
        await with_provider_retry(
            fn,
            config=cfg,
            on_retry=lambda attempt, exc, delay: delays.append(delay),
        )
        assert len(delays) == 3
        assert delays[0] < delays[1] < delays[2]

    @pytest.mark.asyncio
    async def test_zero_retries_means_no_retry(self):
        fn = AsyncMock(side_effect=LLMRateLimitError("openai"))
        cfg = RetryConfig(max_retries=0, base_delay=0.01)
        with pytest.raises(LLMRateLimitError):
            await with_provider_retry(fn, config=cfg)
        assert fn.call_count == 1

    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self):
        class FakeResponse:
            headers = {"Retry-After": "0.02"}
        class FakeError(Exception):
            response = FakeResponse()
            status_code = 429

        fn = AsyncMock(side_effect=[FakeError("rate limited"), "ok"])
        cfg = RetryConfig(max_retries=3, base_delay=0.01, jitter=0.0)

        start = time.monotonic()
        await with_provider_retry(fn, config=cfg)
        elapsed = time.monotonic() - start

        # Should wait at least the Retry-After value (0.02s)
        assert elapsed >= 0.015
        assert fn.call_count == 2

    @pytest.mark.asyncio
    async def test_respects_exception_retry_after_attribute(self):
        exc = LLMRateLimitError("openai", retry_after=0.05)
        fn = AsyncMock(side_effect=[exc, "ok"])
        cfg = RetryConfig(max_retries=3, base_delay=0.01, jitter=0.0)

        start = time.monotonic()
        await with_provider_retry(fn, config=cfg)
        elapsed = time.monotonic() - start

        assert elapsed >= 0.04
        assert fn.call_count == 2


class TestJitterClamping:
    def test_delay_never_exceeds_max_delay(self):
        cfg = RetryConfig(max_delay=10.0, jitter=0.5, base_delay=1.0, backoff_factor=2.0)
        for attempt in range(20):
            delay = cfg.delay_for_attempt(attempt)
            assert delay <= cfg.max_delay, (
                f"attempt {attempt}: delay {delay} > max_delay {cfg.max_delay}"
            )
