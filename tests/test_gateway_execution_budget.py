"""Tests for gateway execution budget controls."""

import pytest
from spoon_bot.gateway.observability.budget import (
    ExecutionBudget,
    BudgetExhaustedError,
    check_budget,
)


class TestExecutionBudget:
    def test_defaults_unlimited(self):
        budget = ExecutionBudget()
        assert budget.request_ms == 0
        assert budget.tool_ms == 0
        assert budget.stream_ms == 0
        assert budget.is_unlimited()

    def test_custom_values(self):
        budget = ExecutionBudget(request_ms=30000, tool_ms=10000, stream_ms=60000)
        assert budget.request_ms == 30000
        assert budget.tool_ms == 10000
        assert budget.stream_ms == 60000
        assert not budget.is_unlimited()

    def test_partial_unlimited(self):
        budget = ExecutionBudget(request_ms=30000)
        assert not budget.is_unlimited()


class TestCheckBudget:
    def test_within_budget_no_error(self):
        check_budget("request", limit_ms=30000, elapsed_ms=1000)

    def test_unlimited_no_error(self):
        check_budget("request", limit_ms=0, elapsed_ms=999999)

    def test_exceeded_raises(self):
        with pytest.raises(BudgetExhaustedError) as exc_info:
            check_budget("request", limit_ms=30000, elapsed_ms=30000)
        assert exc_info.value.budget_type == "request"
        assert exc_info.value.limit_ms == 30000
        assert exc_info.value.elapsed_ms == 30000

    def test_exceeded_message(self):
        with pytest.raises(BudgetExhaustedError, match="request budget exhausted"):
            check_budget("request", limit_ms=5000, elapsed_ms=6000)

    def test_tool_budget(self):
        with pytest.raises(BudgetExhaustedError) as exc_info:
            check_budget("tool", limit_ms=10000, elapsed_ms=15000)
        assert exc_info.value.budget_type == "tool"

    def test_stream_budget(self):
        with pytest.raises(BudgetExhaustedError) as exc_info:
            check_budget("stream", limit_ms=60000, elapsed_ms=61000)
        assert exc_info.value.budget_type == "stream"

    def test_just_under_limit_ok(self):
        check_budget("request", limit_ms=30000, elapsed_ms=29999)
