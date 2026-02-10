import pytest

from spoon_bot.agent.loop import AgentLoop


class DummyMemory:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})


class DummyRunResult:
    def __init__(self, content="ok"):
        self.content = content


class DummyAgent:
    def __init__(self):
        self.memory = DummyMemory()

    async def initialize(self):
        return None

    async def run(self, _message):
        return DummyRunResult("ok")


class DummyChatBot:
    def __init__(self, *args, **kwargs):
        pass


class DummyToolManager:
    def __init__(self, *_args, **_kwargs):
        pass


class DummySkillManager:
    def __init__(self, *args, **kwargs):
        pass


@pytest.fixture
def patch_core(monkeypatch):
    monkeypatch.setattr("spoon_bot.agent.loop.ChatBot", DummyChatBot)
    monkeypatch.setattr("spoon_bot.agent.loop.ToolManager", DummyToolManager)
    monkeypatch.setattr("spoon_bot.agent.loop.SkillManager", DummySkillManager)
    monkeypatch.setattr("spoon_bot.agent.loop.SpoonReactMCP", lambda *a, **k: DummyAgent())
    monkeypatch.setattr("spoon_bot.agent.loop.SpoonReactSkill", lambda *a, **k: DummyAgent())


@pytest.mark.asyncio
async def test_initial_seed_effective(tmp_path, patch_core):
    loop = AgentLoop(workspace=tmp_path, session_key="seed-init", enable_skills=False, auto_commit=False)
    loop._session.add_message("user", "hello")
    loop._session.add_message("assistant", "world")
    loop.sessions.save(loop._session)

    await loop.initialize()

    assert loop._history_seeded is True
    assert len(loop._agent.memory.messages) == 2
    assert loop._agent.memory.messages[0]["role"] == "user"
    assert loop._agent.memory.messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_restart_restores_from_persisted_session(tmp_path, patch_core):
    first = AgentLoop(workspace=tmp_path, session_key="seed-restart", enable_skills=False, auto_commit=False)
    first._session.add_message("user", "persist-user")
    first._session.add_message("assistant", "persist-assistant")
    first.sessions.save(first._session)

    second = AgentLoop(workspace=tmp_path, session_key="seed-restart", enable_skills=False, auto_commit=False)
    await second.initialize()

    assert second._history_seeded is True
    assert len(second._agent.memory.messages) == 2
    assert second._agent.memory.messages[0]["content"] == "persist-user"
    assert second._agent.memory.messages[1]["content"] == "persist-assistant"


@pytest.mark.asyncio
async def test_repeated_process_does_not_bloat_seeded_memory(tmp_path, patch_core):
    loop = AgentLoop(workspace=tmp_path, session_key="seed-nobloat", enable_skills=False, auto_commit=False)
    loop._session.add_message("user", "u1")
    loop._session.add_message("assistant", "a1")
    loop.sessions.save(loop._session)

    await loop.initialize()
    seeded_count = len(loop._agent.memory.messages)

    await loop.process("one")
    await loop.process("two")

    assert len(loop._agent.memory.messages) == seeded_count
