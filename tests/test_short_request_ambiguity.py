from __future__ import annotations

from types import MethodType

from spoon_bot.agent.loop import AgentLoop


def _loop_with_recent_game_skills() -> AgentLoop:
    loop = object.__new__(AgentLoop)
    loop._recent_invoked_skill_contexts = [
        {"name": "joker-game-agent", "location": "skills/joker-game-agent"},
        {"name": "spot-agent-cypher", "location": "skills/spot-agent-cypher"},
    ]

    def collect_request_skill_candidates(self: AgentLoop) -> list[dict[str, object]]:
        return [
            {
                "name": "joker-game-agent",
                "description": "AI agent for JokerGame on-chain card game.",
                "when_to_use": "Join games, play cards, settle.",
            },
            {
                "name": "spot-agent-cypher",
                "description": "AI agent for Spot Game.",
                "when_to_use": "Join and play spot game rounds.",
            },
        ]

    loop._collect_request_skill_candidates = MethodType(
        collect_request_skill_candidates,
        loop,
    )
    return loop


def test_generic_chinese_join_game_clarifies_between_game_workflows() -> None:
    loop = _loop_with_recent_game_skills()

    clarification = AgentLoop._build_short_request_ambiguity_clarification(
        loop,
        "加入游戏吧",
    )

    assert clarification == "我需要先确认要使用哪个 workflow：joker-game-agent、spot-agent-cypher？"


def test_generic_round_request_clarifies_between_game_workflows() -> None:
    loop = _loop_with_recent_game_skills()

    clarification = AgentLoop._build_short_request_ambiguity_clarification(
        loop,
        "来一把",
    )

    assert clarification == "我需要先确认要使用哪个 workflow：joker-game-agent、spot-agent-cypher？"


def test_typoed_joker_identifier_does_not_trigger_workflow_clarification() -> None:
    loop = _loop_with_recent_game_skills()

    clarification = AgentLoop._build_short_request_ambiguity_clarification(
        loop,
        "join jokr game",
    )

    assert clarification == ""
