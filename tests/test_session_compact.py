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
            {"role": "user", "content": "I mentioned the dish name was tomato scrambled eggs."},
            {"role": "assistant", "content": "Recorded. The dish name is tomato scrambled eggs."},
        ]
    )

    compact = build_session_compact_context(session, "What dish name did I mention earlier?")

    assert "User evidence: I mentioned the dish name was tomato scrambled eggs." in compact
    assert "tomato scrambled eggs" in compact


def test_session_compact_strips_runtime_injected_skill_block_from_user_evidence() -> None:
    session = DummySession(
        [
            {
                "role": "user",
                "content": (
                    "Continue the previous task and check status first.\n\n"
                    "---\n"
                    "[PRE-LOADED SKILL: demo-skill]\n"
                    "Base directory: /tmp/demo\n"
                    "Do not search for alternatives."
                ),
            },
            {"role": "assistant", "content": "Checked current status."},
        ]
    )

    compact = build_session_compact_context(session, "Continue")

    assert "User evidence: Continue the previous task and check status first." in compact
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


def test_session_compact_includes_interrupted_user_fact_as_non_executable_evidence() -> None:
    raw_key = "0x" + "ab" * 32
    session = DummySession(
        [
            {"role": "user", "content": "Join the game", "turn_state": "completed"},
            {"role": "assistant", "content": "The old wallet joined one game."},
            {
                "role": "user",
                "content": f"{raw_key} Use this private key to rejoin the latest game",
                "turn_state": "interrupted",
            },
            {"role": "user", "content": "What is the current score now?", "turn_state": "completed"},
            {"role": "assistant", "content": "The old wallet is currently 1 win and 12 losses."},
        ]
    )

    compact = build_session_compact_context(session, "Didn't I give you the new-key wallet?")

    assert "Interrupted/superseded user evidence" in compact
    assert "do not execute unless the newest request explicitly resumes it" in compact
    assert raw_key not in compact
    assert "***masked_private_key***" in compact
