"""Execution budget controls for the gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar


T = TypeVar("T")


class BudgetExhaustedError(Exception):
    """Raised when an execution budget is exhausted."""

    def __init__(self, budget_type: str, limit_ms: int, elapsed_ms: int):
        self.budget_type = budget_type
        self.limit_ms = limit_ms
        self.elapsed_ms = elapsed_ms
        super().__init__(
            f"{budget_type} budget exhausted: {elapsed_ms}ms exceeded {limit_ms}ms limit"
        )


@dataclass
class ExecutionBudget:
    """Tracks execution time budgets for request/tool/stream.

    Attributes:
        request_ms: Maximum total request time in ms (0 = unlimited).
        tool_ms: Maximum per-tool execution time in ms (0 = unlimited).
        stream_ms: Maximum streaming duration in ms (0 = unlimited).
    """

    request_ms: int = 0
    tool_ms: int = 0
    stream_ms: int = 0

    def is_unlimited(self) -> bool:
        """Check if all budgets are unlimited (0)."""
        return self.request_ms == 0 and self.tool_ms == 0 and self.stream_ms == 0


def check_budget(
    budget_type: str,
    limit_ms: int,
    elapsed_ms: int,
) -> None:
    """Check if a budget has been exceeded.

    Args:
        budget_type: Type of budget (e.g., "request", "tool", "stream").
        limit_ms: Budget limit in milliseconds. 0 means unlimited.
        elapsed_ms: Elapsed time in milliseconds.

    Raises:
        BudgetExhaustedError: If the budget is exceeded.
    """
    if limit_ms > 0 and elapsed_ms >= limit_ms:
        raise BudgetExhaustedError(budget_type, limit_ms, elapsed_ms)


async def wait_for_budget(
    awaitable: Awaitable[T],
    *,
    budget_type: str,
    limit_ms: int,
    elapsed_ms: Callable[[], int] | int,
) -> T:
    """Await a coroutine with the remaining execution budget as a hard timeout."""

    def _elapsed() -> int:
        value = elapsed_ms() if callable(elapsed_ms) else elapsed_ms
        return int(value or 0)

    def _close_unawaited() -> None:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()

    current_elapsed = _elapsed()
    if limit_ms <= 0:
        return await awaitable

    remaining_ms = int(limit_ms) - current_elapsed
    if remaining_ms <= 0:
        _close_unawaited()
        raise BudgetExhaustedError(budget_type, int(limit_ms), current_elapsed)

    try:
        return await asyncio.wait_for(
            awaitable,
            timeout=max(0.001, remaining_ms / 1000),
        )
    except asyncio.TimeoutError as exc:
        elapsed_after_wait = _elapsed()
        if elapsed_after_wait >= int(limit_ms) - 5:
            raise BudgetExhaustedError(
                budget_type,
                int(limit_ms),
                elapsed_after_wait,
            ) from exc
        raise
