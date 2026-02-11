"""Tracing and timing utilities for the gateway."""

from __future__ import annotations

import time
from uuid import uuid4
from datetime import datetime, timezone


def new_trace_id() -> str:
    """Generate a unique trace identifier."""
    return f"trc_{uuid4().hex[:16]}"


def now_ms() -> int:
    """Return current epoch time in milliseconds."""
    return int(time.time() * 1000)


class TimerSpan:
    """A timing span that measures elapsed time.

    Can be used as a context manager:
        with TimerSpan("request") as span:
            ... do work ...
        print(span.elapsed_ms)
    """

    def __init__(self, name: str = "total"):
        self.name = name
        self._start_ns = time.monotonic_ns()
        self.start_ms = now_ms()
        self._end_ns: int | None = None

    @property
    def elapsed_ms(self) -> int:
        """Elapsed time in milliseconds."""
        end = self._end_ns if self._end_ns is not None else time.monotonic_ns()
        return (end - self._start_ns) // 1_000_000

    def stop(self) -> int:
        """Stop the timer and return elapsed ms."""
        if self._end_ns is None:
            self._end_ns = time.monotonic_ns()
        return self.elapsed_ms

    def __enter__(self) -> "TimerSpan":
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


def build_timing_payload(span: TimerSpan, extra: dict | None = None) -> dict:
    """Build a timing payload dictionary from a TimerSpan.

    Returns:
        Dictionary with timing information including total_elapsed_ms,
        started_at ISO timestamp, and any extra fields.
    """
    payload = {
        "total_elapsed_ms": span.elapsed_ms,
        "started_at": datetime.fromtimestamp(
            span.start_ms / 1000, tz=timezone.utc
        ).isoformat(),
        "span": span.name,
    }
    if extra:
        payload.update(extra)
    return payload
