from __future__ import annotations

from spoon_bot.agent.request_hints import (
    build_request_execution_hints,
    classify_continuation_intent,
    request_is_bare_continuation,
    request_is_plain_continuation_only,
)


def test_continuation_intent_is_structured_without_selecting_workflow() -> None:
    intent = classify_continuation_intent("继续")

    assert intent.kind == "plain"
    assert intent.is_continuation is True
    assert intent.adds_scope is False
    assert intent.to_hints() == {
        "kind": "plain",
        "is_continuation": True,
        "adds_scope": False,
    }
    assert not hasattr(intent, "skill")
    assert not hasattr(intent, "workflow")


def test_scoped_continuation_preserves_new_scope() -> None:
    intent = classify_continuation_intent("继续玩5把")

    assert intent.kind == "scoped"
    assert intent.is_continuation is True
    assert intent.adds_scope is True
    assert intent.is_plain is False


def test_compatibility_wrappers_delegate_to_structured_intent() -> None:
    assert request_is_bare_continuation("continue") is True
    assert request_is_plain_continuation_only("continue") is True
    assert request_is_bare_continuation("继续玩5把") is True
    assert request_is_plain_continuation_only("继续玩5把") is False


def test_request_hints_include_structured_continuation_intent() -> None:
    hints = build_request_execution_hints("继续", [])

    assert hints["continuation_intent"] == {
        "kind": "plain",
        "is_continuation": True,
        "adds_scope": False,
    }
    assert hints["bare_continuation"] is True
    assert hints["plain_continuation"] is True
