from __future__ import annotations

from typing import Any

from spoon_bot.agent.session_compact import build_session_compact_context


class DummySession:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self._messages = messages

    def get_messages(self) -> list[dict[str, Any]]:
        return self._messages


def test_session_compact_keeps_user_fact_as_evidence() -> None:
    session = DummySession(
        [
            {"role": "user", "content": "我刚才说的菜名是西红柿炒蛋。"},
            {"role": "assistant", "content": "记下了，菜名是西红柿炒蛋。"},
        ]
    )

    compact = build_session_compact_context(session, "刚才我说的菜名是什么？")

    assert "User evidence: 我刚才说的菜名是西红柿炒蛋。" in compact
    assert "西红柿炒蛋" in compact


def test_session_compact_strips_runtime_injected_skill_block_from_user_evidence() -> None:
    session = DummySession(
        [
            {
                "role": "user",
                "content": (
                    "继续刚才的任务，先检查状态。\n\n"
                    "---\n"
                    "[PRE-LOADED SKILL: demo-skill]\n"
                    "Base directory: /tmp/demo\n"
                    "Do not search for alternatives."
                ),
            },
            {"role": "assistant", "content": "Checked current status."},
        ]
    )

    compact = build_session_compact_context(session, "继续")

    assert "User evidence: 继续刚才的任务，先检查状态。" in compact
    assert "[PRE-LOADED SKILL:" not in compact
    assert "Do not search for alternatives" not in compact


def test_session_compact_does_not_preserve_long_prior_task_as_user_evidence() -> None:
    session = DummySession(
        [
            {
                "role": "user",
                "content": (
                    "Deploy service alpha. Run migration. Restart worker. "
                    "Keep retrying until successful."
                ),
            },
            {"role": "assistant", "content": "Deployment finished successfully."},
            {"role": "user", "content": "Remember that the deployment ticket was OPS-42."},
            {"role": "assistant", "content": "Recorded. Deployment ticket is OPS-42."},
        ]
    )

    compact = build_session_compact_context(session, "Summarize yesterday's meeting.")

    assert "Deploy service alpha. Run migration." not in compact
    assert "User evidence: Remember that the deployment ticket was OPS-42." in compact
    assert "OPS-42" in compact
