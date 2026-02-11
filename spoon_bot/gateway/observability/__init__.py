"""Gateway observability: tracing, timing, and execution budgets."""

from spoon_bot.gateway.observability.tracing import (
    new_trace_id,
    now_ms,
    TimerSpan,
    build_timing_payload,
)
from spoon_bot.gateway.observability.budget import (
    ExecutionBudget,
    BudgetExhaustedError,
    check_budget,
)

__all__ = [
    "new_trace_id",
    "now_ms",
    "TimerSpan",
    "build_timing_payload",
    "ExecutionBudget",
    "BudgetExhaustedError",
    "check_budget",
]
