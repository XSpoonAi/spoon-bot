from __future__ import annotations

from spoon_bot.agent.execution_ledger import (
    ExecutionLedger,
    load_recent_execution_ledger_context,
    persist_execution_ledger,
    persist_execution_ledger_snapshot,
)


def _ledger(tmp_path, *, owner: str, turn_id: str, request: str) -> ExecutionLedger:
    ledger = ExecutionLedger(
        owner=owner,
        workspace=str(tmp_path),
        session_id=owner,
        turn_id=turn_id,
        user_request=request,
    )
    ledger.record_tool("shell", {"command": "true"}, "ok", "ok")
    return ledger


def test_recent_context_only_loads_completed_ledgers(tmp_path) -> None:
    owner = "user:test|session:conversation-a"
    completed = _ledger(
        tmp_path,
        owner=owner,
        turn_id="completed-turn",
        request="completed request",
    )
    failed = _ledger(
        tmp_path,
        owner=owner,
        turn_id="failed-turn",
        request="failed request",
    )

    persist_execution_ledger(completed, outcome="completed")
    persist_execution_ledger(failed, outcome="failed")

    context = load_recent_execution_ledger_context(workspace=tmp_path, owner=owner)

    assert "completed request" in context
    assert "failed request" not in context


def test_recent_context_ignores_active_crash_snapshot(tmp_path) -> None:
    owner = "user:test|session:conversation-a"
    active = _ledger(
        tmp_path,
        owner=owner,
        turn_id="active-turn",
        request="request interrupted by process crash",
    )
    persist_execution_ledger_snapshot(active)

    assert load_recent_execution_ledger_context(workspace=tmp_path, owner=owner) == ""


def test_recent_context_isolated_by_session_owner(tmp_path) -> None:
    first_owner = "user:test|session:conversation-a"
    second_owner = "user:test|session:conversation-b"
    ledger = _ledger(
        tmp_path,
        owner=first_owner,
        turn_id="first-session-turn",
        request="first session request",
    )
    persist_execution_ledger(ledger, outcome="completed")

    assert load_recent_execution_ledger_context(
        workspace=tmp_path,
        owner=second_owner,
    ) == ""


def test_structured_numeric_facts_are_derived_deterministically(tmp_path) -> None:
    ledger = _ledger(
        tmp_path,
        owner="user:test|session:numeric",
        turn_id="numeric-turn",
        request="summarize verified totals",
    )
    ledger.record_tool(
        "shell",
        {"command": "structured-cli"},
        "completed",
        "completed",
        metadata={
            "verified_facts": [
                {"id": "gross", "label": "Gross payout", "value": "160", "unit": "GLD"},
                {"id": "cost", "label": "Total cost", "value": "60", "unit": "GLD"},
            ],
            "derived_facts": [
                {
                    "id": "net",
                    "label": "Net result",
                    "operation": "subtract",
                    "inputs": ["gross", "cost"],
                    "unit": "GLD",
                }
            ],
        },
    )

    summary = ledger.render_structured_numeric_summary()

    assert "Gross payout: 160 GLD" in summary
    assert "Total cost: 60 GLD" in summary
    assert "Net result: 100 GLD" in summary


def test_structured_numeric_derivation_rejects_mixed_units(tmp_path) -> None:
    ledger = _ledger(
        tmp_path,
        owner="user:test|session:mixed-units",
        turn_id="mixed-unit-turn",
        request="summarize verified totals",
    )
    ledger.record_tool(
        "shell",
        {"command": "structured-cli"},
        "completed",
        "completed",
        metadata={
            "verified_facts": [
                {"id": "left", "value": "10", "unit": "USD"},
                {"id": "right", "value": "2", "unit": "ETH"},
            ],
            "derived_facts": [
                {
                    "id": "invalid_total",
                    "operation": "add",
                    "inputs": ["left", "right"],
                }
            ],
        },
    )

    fact_ids = {fact["fact_id"] for fact in ledger.structured_numeric_facts()}
    assert fact_ids == {"left", "right"}
