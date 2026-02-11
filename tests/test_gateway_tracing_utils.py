"""Tests for gateway tracing utilities."""

import time
import pytest
from spoon_bot.gateway.observability.tracing import (
    new_trace_id,
    now_ms,
    TimerSpan,
    build_timing_payload,
)


class TestNewTraceId:
    def test_returns_string(self):
        tid = new_trace_id()
        assert isinstance(tid, str)

    def test_has_prefix(self):
        tid = new_trace_id()
        assert tid.startswith("trc_")

    def test_unique(self):
        ids = {new_trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_length(self):
        tid = new_trace_id()
        # "trc_" + 16 hex chars = 20 chars
        assert len(tid) == 20


class TestNowMs:
    def test_returns_int(self):
        assert isinstance(now_ms(), int)

    def test_reasonable_value(self):
        ms = now_ms()
        # Should be after 2024-01-01 in ms
        assert ms > 1_704_067_200_000

    def test_monotonically_increasing(self):
        a = now_ms()
        time.sleep(0.01)
        b = now_ms()
        assert b >= a


class TestTimerSpan:
    def test_elapsed_grows(self):
        span = TimerSpan("test")
        time.sleep(0.05)
        assert span.elapsed_ms >= 40  # at least ~40ms with some tolerance

    def test_stop_freezes_elapsed(self):
        span = TimerSpan("test")
        time.sleep(0.05)
        elapsed = span.stop()
        time.sleep(0.05)
        assert span.elapsed_ms == elapsed

    def test_context_manager(self):
        with TimerSpan("ctx") as span:
            time.sleep(0.05)
        assert span.elapsed_ms >= 40

    def test_name(self):
        span = TimerSpan("my_span")
        assert span.name == "my_span"

    def test_default_name(self):
        span = TimerSpan()
        assert span.name == "total"

    def test_start_ms_is_epoch(self):
        before = int(time.time() * 1000)
        span = TimerSpan()
        after = int(time.time() * 1000)
        assert before <= span.start_ms <= after


class TestBuildTimingPayload:
    def test_contains_required_fields(self):
        with TimerSpan("req") as span:
            time.sleep(0.01)
        payload = build_timing_payload(span)
        assert "total_elapsed_ms" in payload
        assert "started_at" in payload
        assert "span" in payload

    def test_total_elapsed_matches_span(self):
        with TimerSpan("req") as span:
            time.sleep(0.05)
        payload = build_timing_payload(span)
        assert payload["total_elapsed_ms"] == span.elapsed_ms

    def test_span_name(self):
        with TimerSpan("my_span") as span:
            pass
        payload = build_timing_payload(span)
        assert payload["span"] == "my_span"

    def test_started_at_is_iso(self):
        with TimerSpan() as span:
            pass
        payload = build_timing_payload(span)
        # Should contain T and timezone info
        assert "T" in payload["started_at"]

    def test_extra_fields_merged(self):
        with TimerSpan() as span:
            pass
        payload = build_timing_payload(span, extra={"tool_name": "shell", "steps": 3})
        assert payload["tool_name"] == "shell"
        assert payload["steps"] == 3
