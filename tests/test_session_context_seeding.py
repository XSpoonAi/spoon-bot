import asyncio

from spoon_bot.agent.loop import AgentLoop


class _FakeAgent:
    def __init__(self):
        self.memory = object()  # only used for hasattr check
        self.seeded_messages = []

    async def add_message(self, role: str, content: str):
        self.seeded_messages.append({"role": role, "content": content})


def _run(coro):
    return asyncio.run(coro)


def test_initial_history_seed_injects_recent_window_and_clips(tmp_path):
    loop = AgentLoop(
        workspace=tmp_path,
        session_key="seed-window",
        enable_skills=False,
        auto_commit=False,
    )

    # Build 14 history messages -> only last 12 should be seeded
    for i in range(14):
        role = "user" if i % 2 == 0 else "assistant"
        content = (f"msg-{i}-" + "x" * 2500)  # exceeds clip limit
        loop._session.add_message(role, content)

    # Add a non user/assistant role that should be filtered out
    loop._session.add_message("tool", "tool output")

    fake_agent = _FakeAgent()
    loop._agent = fake_agent

    _run(loop._seed_session_history_into_agent_memory())

    assert len(fake_agent.seeded_messages) == 12
    # Last 12 from msg-2 ... msg-13
    assert fake_agent.seeded_messages[0]["content"].startswith("msg-2-")
    assert fake_agent.seeded_messages[-1]["content"].startswith("msg-13-")
    # clipping applied
    assert all(len(m["content"]) <= 2000 for m in fake_agent.seeded_messages)
    # filtered to user/assistant only
    assert all(m["role"] in {"user", "assistant"} for m in fake_agent.seeded_messages)


def test_restart_can_restore_from_persisted_session(tmp_path):
    # First process instance writes session to disk
    loop1 = AgentLoop(
        workspace=tmp_path,
        session_key="persisted-restart",
        enable_skills=False,
        auto_commit=False,
    )
    loop1._session.add_message("user", "remember code: blue-whale-729")
    loop1._session.add_message("assistant", "got it")
    loop1.sessions.save(loop1._session)

    # Simulate process restart: new AgentLoop instance
    loop2 = AgentLoop(
        workspace=tmp_path,
        session_key="persisted-restart",
        enable_skills=False,
        auto_commit=False,
    )
    fake_agent = _FakeAgent()
    loop2._agent = fake_agent

    _run(loop2._seed_session_history_into_agent_memory())

    assert len(fake_agent.seeded_messages) == 2
    assert fake_agent.seeded_messages[0]["content"] == "remember code: blue-whale-729"
    assert fake_agent.seeded_messages[1]["content"] == "got it"


def test_seeding_is_idempotent_and_wont_bloat_on_repeat(tmp_path):
    loop = AgentLoop(
        workspace=tmp_path,
        session_key="idempotent",
        enable_skills=False,
        auto_commit=False,
    )
    loop._session.add_message("user", "u1")
    loop._session.add_message("assistant", "a1")

    fake_agent = _FakeAgent()
    loop._agent = fake_agent

    _run(loop._seed_session_history_into_agent_memory())
    first_count = len(fake_agent.seeded_messages)

    # Repeat seeding (equivalent to repeated initialization attempts)
    _run(loop._seed_session_history_into_agent_memory())
    second_count = len(fake_agent.seeded_messages)

    assert first_count == 2
    assert second_count == 2
