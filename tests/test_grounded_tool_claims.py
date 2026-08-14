from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.execution_context import bind_request_execution_hints
from spoon_bot.agent.tools.registry import ToolRegistry
from spoon_bot.agent.turn_verifiers import (
    build_active_background_job_pending_answer,
    deterministic_completion_verdict,
    final_answer_has_unsupported_numeric_claims,
    latest_tool_event_has_active_background_job,
    latest_unresolved_tool_failure,
    tool_events_need_more_evidence,
)


class _ResultTool(Tool):
    @property
    def name(self) -> str:
        return "game_result"

    @property
    def description(self) -> str:
        return "Return a verified result."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> str:
        return "rank=1/6 reward=75 GLD"


def _event(text: str, *, name: str = "shell", status: str = "succeeded") -> dict:
    return {
        "metadata": {
            "name": name,
            "status": status,
            "full_output": text,
        }
    }


@pytest.mark.asyncio
async def test_registry_rejects_registered_but_inactive_tool() -> None:
    registry = ToolRegistry()
    registry.register(_ResultTool())
    registry.set_tool_filter(enabled_tools=set())

    result = await registry.execute("game_result", {})

    assert "registered but not active" in result
    assert "rank=1/6" not in result


@pytest.mark.asyncio
async def test_registry_executes_tool_after_explicit_activation() -> None:
    registry = ToolRegistry()
    registry.register(_ResultTool())
    registry.set_tool_filter(enabled_tools=set())
    assert registry.activate_tool("game_result") is True

    assert await registry.execute("game_result", {}) == "rank=1/6 reward=75 GLD"


def test_latest_capability_error_is_an_unresolved_failure() -> None:
    failure = latest_unresolved_tool_failure([
        _event("Error: Unknown tool 'game_history'.", name="game_history", status="failed")
    ])

    assert "Unknown tool 'game_history'" in failure


def test_later_success_supersedes_earlier_tool_failure() -> None:
    failure = latest_unresolved_tool_failure([
        _event("Error: Unknown tool 'game_history'.", name="game_history", status="failed"),
        _event("SETTLEMENT rank=1/6 reward=75 GLD"),
    ])

    assert failure == ""


def test_historical_success_cannot_replace_current_failure() -> None:
    failure = latest_unresolved_tool_failure([
        {
            "metadata": {
                "name": "shell",
                "status": "failed",
                "full_output": "Error: current request failed",
            }
        },
        {
            "metadata": {
                "name": "shell",
                "status": "succeeded",
                "freshness": "historical",
                "full_output": "previous request completed",
            }
        },
    ])

    assert "current request failed" in failure


@pytest.mark.asyncio
async def test_history_search_is_removed_from_schema_after_budget() -> None:
    registry = ToolRegistry()
    registry.register(_ResultTool())
    hints = {
        "history_search_state": {
            "calls": 0,
            "max_calls": 3,
            "exhausted": True,
            "disabled": True,
            "disabled_reason": "history_search_budget_exhausted",
        }
    }

    with bind_request_execution_hints(hints):
        names = {item["function"]["name"] for item in registry.get_definitions()}
        result = await registry.execute("search_history", {})

    assert "search_history" not in names
    assert "disabled for this request" in result


def test_numeric_claims_must_exist_in_evidence() -> None:
    evidence = "SETTLEMENT rank=1/6 reward=75 GLD balance=215 GLD"

    assert not final_answer_has_unsupported_numeric_claims(
        "Rank 1/6, reward 75 GLD; balance 215 GLD.",
        evidence,
    )
    assert final_answer_has_unsupported_numeric_claims(
        "Rank 1/6, reward 175 GLD; balance 215 GLD.",
        evidence,
    )


def test_markdown_numbered_lists_are_not_treated_as_numeric_claims() -> None:
    assert not final_answer_has_unsupported_numeric_claims(
        "1. Verified the result.\n2. Reported the blocker.",
        "result verified; blocker reported",
    )


def test_completion_verifier_fails_closed_without_terminal_evidence() -> None:
    verdict = deterministic_completion_verdict(
        "The game completed successfully.",
        [_event("JOINED game=481 phase=waiting")],
        verifier_reason="Verifier unavailable",
    )

    assert verdict["status"] == "awaiting_user"
    assert "no deterministic terminal evidence" in verdict["reason"]


def test_completion_verifier_accepts_tool_terminal_summary() -> None:
    verdict = deterministic_completion_verdict(
        "Result available.",
        [_event("Read it aloud: SETTLEMENT outcome=WIN reward=75 GLD")],
        verifier_reason="Verifier unavailable",
    )

    assert verdict["status"] == "complete"


def test_completion_verifier_requests_one_background_handoff_continuation() -> None:
    verdict = deterministic_completion_verdict(
        "The command is still running as background job sh_123; resume by checking that job.",
        [
            _event(
                "Foreground timeout (600s) exceeded - command moved to background.\n"
                "job_id: sh_123\nstatus: running (elapsed 600s)"
            )
        ],
        verifier_reason="Verifier unavailable",
    )

    assert verdict["status"] == "needs_continuation"
    assert "still running" in verdict["reason"]


def test_background_handoff_pending_answer_is_deterministic() -> None:
    events = [
        _event(
            "job_id: sh_123\n"
            "status: running\n"
            "returncode: running\n"
            "Output:\nclaim submitted"
        )
    ]

    answer = build_active_background_job_pending_answer(events)

    assert "still running" in answer
    assert "`sh_123`" in answer
    assert "turn is paused" in answer


def test_later_authoritative_evidence_supersedes_stale_running_job() -> None:
    events = [
        _event(
            "job_id: sh_123\nstatus: running\nreturncode: running",
        ),
        _event(
            "Wallet=0xabc GAS=0.3 GLD=200.0 AgentID=none",
            name="wallet",
        ),
    ]

    assert not latest_tool_event_has_active_background_job(events)
    assert not tool_events_need_more_evidence(events)


@pytest.mark.asyncio
async def test_final_synthesis_returns_capability_error_without_model_rewrite() -> None:
    from spoon_bot.agent.loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    chat = AsyncMock(return_value=SimpleNamespace(content="You won 100 GLD."))
    loop._chatbot = SimpleNamespace(llm_manager=SimpleNamespace(chat=chat))

    result = await AgentLoop._synthesize_final_answer_from_tool_events(
        loop,
        [_event("Error: Unknown tool 'latest_game'.", name="latest_game", status="failed")],
        user_message="Check the latest game result",
    )

    assert result == "Error: Unknown tool 'latest_game'."
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_terminal_summary_cannot_be_rewritten_from_loss_to_win() -> None:
    from spoon_bot.agent.loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    chat = AsyncMock(return_value=SimpleNamespace(content="You won the game."))
    loop._chatbot = SimpleNamespace(llm_manager=SimpleNamespace(chat=chat))

    result = await AgentLoop._synthesize_final_answer_from_tool_events(
        loop,
        [
            _event(
                "Read it aloud: SETTLEMENT game=480 outcome=LOSE "
                "rank=4/6 reward=0 GLD"
            )
        ],
        user_message="告诉我输赢和奖金",
    )

    assert result == "SETTLEMENT game=480 outcome=LOSE rank=4/6 reward=0 GLD"
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_synthesis_rejects_invented_reward_amount() -> None:
    from spoon_bot.agent.loop import AgentLoop

    loop = AgentLoop.__new__(AgentLoop)
    chat = AsyncMock(return_value=SimpleNamespace(content="Rank 1/6, reward 175 GLD."))
    loop._chatbot = SimpleNamespace(llm_manager=SimpleNamespace(chat=chat))
    loop.provider = "test"

    result = await AgentLoop._synthesize_final_answer_from_tool_events(
        loop,
        [_event("SETTLEMENT rank=1/6 reward=75 GLD")],
        user_message="Report the result",
    )

    assert "175" not in result
    assert "reward=75 GLD" in result
