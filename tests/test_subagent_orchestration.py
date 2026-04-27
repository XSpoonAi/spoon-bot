from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from pathlib import Path
from uuid import uuid4

import pytest
from spoon_bot.bus.events import InboundMessage
from spoon_bot.channels.manager import ChannelManager
from spoon_bot.session.manager import Session, SessionManager
from spoon_bot.subagent.manager import SubagentManager
from spoon_bot.subagent.models import (
    RoutingMode,
    SpawnMode,
    SubagentConfig,
    SubagentRecord,
    SubagentResult,
    SubagentState,
)
from spoon_bot.subagent.tools import SubagentTool


def _tool_record(
    *,
    agent_id: str = "sub_test",
    run_id: str = "run_test",
    label: str = "planner: analyse",
    depth: int = 1,
    model_name: str | None = "test-model",
    spawn_mode: str = "run",
    agent_name: str | None = None,
    config: SubagentConfig | None = None,
):
    return SimpleNamespace(
        agent_id=agent_id,
        run_id=run_id,
        label=label,
        depth=depth,
        model_name=model_name,
        spawn_mode=SimpleNamespace(value=spawn_mode),
        agent_name=agent_name,
        config=config or SubagentConfig(),
    )


@pytest.fixture
def workspace_dir():
    root = Path.cwd() / "workspace" / "pytest_subagents"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"case_{uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=True)
    yield case_dir


@pytest.mark.asyncio
async def test_subagent_tool_spawn_uses_task_id_resume(monkeypatch):
    manager = SimpleNamespace()
    manager.spawn = AsyncMock()
    manager.resume_task = AsyncMock(return_value=_tool_record(agent_id="sub_resume"))

    tool = SubagentTool(manager=manager, parent_agent_id="parent_1")
    tool.set_spawner_context(
        session_key="session_root",
        channel="telegram:bot",
        metadata={"chat_id": 1},
        reply_to="42",
    )

    result = await tool.execute(action="spawn", task="Continue implementation", task_id="sub_prev")

    manager.resume_task.assert_awaited_once()
    kwargs = manager.resume_task.await_args.kwargs
    assert kwargs["task_id"] == "sub_prev"
    assert kwargs["task"] == "Continue implementation"
    assert kwargs["parent_id"] == "parent_1"
    assert kwargs["spawner_session_key"] == "session_root"
    assert kwargs["spawner_channel"] == "telegram:bot"
    assert kwargs["spawner_metadata"] == {"chat_id": 1}
    assert kwargs["spawner_reply_to"] == "42"
    assert kwargs["config"] is None
    manager.spawn.assert_not_called()
    assert "Run ID:  run_test" in result
    assert "Task ID: sub_resume" in result


@pytest.mark.asyncio
async def test_subagent_tool_resume_passes_requester_context():
    manager = SimpleNamespace()
    manager.resume_agent = AsyncMock(
        return_value=_tool_record(
            agent_id="sub_session",
            label="session agent",
            spawn_mode="session",
            agent_name="planner-session",
        )
    )

    tool = SubagentTool(manager=manager, parent_agent_id="parent_2")
    tool.set_spawner_context(
        session_key="session_child",
        channel="discord:test",
        metadata={"thread_id": "abc"},
        reply_to="99",
    )

    result = await tool.execute(
        action="resume",
        agent_name="planner-session",
        task="Refine the plan",
    )

    manager.resume_agent.assert_awaited_once()
    kwargs = manager.resume_agent.await_args.kwargs
    assert kwargs["agent_name"] == "planner-session"
    assert kwargs["task"] == "Refine the plan"
    assert kwargs["parent_id"] == "parent_2"
    assert kwargs["spawner_session_key"] == "session_child"
    assert kwargs["spawner_channel"] == "discord:test"
    assert kwargs["spawner_metadata"] == {"thread_id": "abc"}
    assert kwargs["spawner_reply_to"] == "99"
    assert "Run ID: run_test" in result
    assert "Task ID: sub_session" in result


@pytest.mark.asyncio
async def test_subagent_tool_spawn_passes_enable_skills_and_normalized_thinking():
    manager = SimpleNamespace(
        spawn=AsyncMock(
            return_value=_tool_record(
                agent_id="sub_skills",
                config=SubagentConfig(
                    enable_skills=True,
                    thinking_level="extended",
                ),
            )
        )
    )
    tool = SubagentTool(manager=manager)

    result = await tool.execute(
        action="spawn",
        task="Review the module",
        enable_skills=True,
        thinking="deep",
    )

    kwargs = manager.spawn.await_args.kwargs
    assert kwargs["config"].enable_skills is True
    assert kwargs["config"].thinking_level == "extended"
    assert "skills: on" in result
    assert "thinking: extended" in result


@pytest.mark.asyncio
async def test_subagent_tool_management_actions_pass_requester_scope():
    manager = SimpleNamespace(
        get_status_summary=Mock(return_value={"total": 0, "active": [], "recent": []}),
        cancel=AsyncMock(return_value=True),
        cancel_all=AsyncMock(return_value=1),
        steer=AsyncMock(return_value={"status": "accepted", "message": "ok", "run_id": "run_1"}),
        get_info=AsyncMock(return_value={"run_id": "run_1", "label": "x", "state": "running", "task": "t", "depth": 1, "tool_profile": "core"}),
        registry=SimpleNamespace(get=Mock(return_value=SimpleNamespace()), get_descendants=Mock(return_value=[])),
    )
    tool = SubagentTool(manager=manager, parent_agent_id="parent_scope")
    tool.set_spawner_context(
        session_key="session_scope",
        channel="telegram:test",
        metadata={"chat_id": 7},
        reply_to="55",
    )

    await tool.execute(action="status")
    await tool.execute(action="cancel")
    await tool.execute(action="steer", agent_id="sub_visible", message="redirect")
    await tool.execute(action="info", agent_id="sub_visible")

    manager.get_status_summary.assert_called_once_with(
        parent_id="parent_scope",
        spawner_session_key="session_scope",
    )
    manager.cancel_all.assert_awaited_once_with(
        parent_id="parent_scope",
        spawner_session_key="session_scope",
    )
    manager.steer.assert_awaited_once_with(
        "sub_visible",
        "redirect",
        spawner_session_key="session_scope",
    )
    manager.get_info.assert_awaited_with(
        "sub_visible",
        spawner_session_key="session_scope",
    )


@pytest.mark.asyncio
async def test_subagent_manager_collect_results_scoped_and_retains_others(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    await manager._results.put(
        SubagentResult(
            agent_id="sub_root",
            label="root task",
            state=SubagentState.COMPLETED,
            result="root done",
            spawner_session_key="session_root",
            spawner_channel="cli",
        )
    )
    await manager._results.put(
        SubagentResult(
            agent_id="sub_child",
            label="child task",
            state=SubagentState.COMPLETED,
            result="child done",
            spawner_session_key="session_child",
            spawner_channel="cli",
        )
    )

    root_results = await manager.collect_results(spawner_session_key="session_root")
    child_results = await manager.collect_results(spawner_session_key="session_child")

    assert [result.agent_id for result in root_results] == ["sub_root"]
    assert [result.agent_id for result in child_results] == ["sub_child"]


@pytest.mark.asyncio
async def test_subagent_manager_resume_task_preserves_existing_config(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    original_config = SubagentConfig(
        role="planner",
        tool_profile="research",
        max_iterations=12,
        timeout_seconds=120,
    )
    record = SubagentRecord(
        agent_id="sub_existing",
        label="old label",
        task="old task",
        state=SubagentState.COMPLETED,
        spawner_session_key="session_nested",
        config=original_config,
        model_name="model-a",
        result="old result",
    )
    manager.registry.register(record)
    original_run_id = record.run_id

    resumed = await manager.resume_task(
        task_id="sub_existing",
        task="new task",
        spawner_session_key="session_nested",
        spawner_channel="telegram:test",
    )

    assert resumed.agent_id == "sub_existing"
    assert resumed.task == "new task"
    assert resumed.state == SubagentState.PENDING
    assert resumed.run_id != original_run_id
    assert resumed.config.tool_profile == "research"
    assert resumed.config.max_iterations == 12
    assert resumed.spawner_session_key == "session_nested"
    assert resumed.spawner_channel == "telegram:test"
    assert resumed.result is None


@pytest.mark.asyncio
async def test_subagent_manager_resume_task_rejects_invisible_task_id(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    record = SubagentRecord(
        agent_id="sub_other_scope",
        label="other scope",
        task="old task",
        state=SubagentState.COMPLETED,
        spawner_session_key="session_other",
    )
    manager.registry.register(record)
    original_run_id = record.run_id

    with pytest.raises(ValueError, match="was not found"):
        await manager.resume_task(
            task_id="sub_other_scope",
            task="steal resume",
            spawner_session_key="session_root",
            spawner_channel="telegram:root",
        )

    assert record.run_id == original_run_id
    assert record.task == "old task"
    assert record.state == SubagentState.COMPLETED
    assert record.spawner_session_key == "session_other"


@pytest.mark.asyncio
async def test_subagent_manager_collect_results_filters_by_run_id(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    await manager._results.put(
        SubagentResult(
            agent_id="sub_shared",
            run_id="run_old",
            label="shared task",
            state=SubagentState.COMPLETED,
            result="old done",
            spawner_session_key="session_root",
            spawner_channel="cli",
        )
    )
    await manager._results.put(
        SubagentResult(
            agent_id="sub_shared",
            run_id="run_new",
            label="shared task",
            state=SubagentState.COMPLETED,
            result="new done",
            spawner_session_key="session_root",
            spawner_channel="cli",
        )
    )

    new_results = await manager.collect_results(
        spawner_session_key="session_root",
        run_id="run_new",
    )
    old_results = await manager.collect_results(
        spawner_session_key="session_root",
        run_id="run_old",
    )

    assert [result.run_id for result in new_results] == ["run_new"]
    assert [result.run_id for result in old_results] == ["run_old"]


@pytest.mark.asyncio
async def test_subagent_manager_spawn_prefers_explicit_spawner_context(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager.set_spawner_context(session_key="session_root", channel="discord:root")
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    record = await manager.spawn(
        task="nested task",
        spawner_session_key="session_child",
        spawner_channel="discord:child",
    )

    assert record.spawner_session_key == "session_child"
    assert record.spawner_channel == "discord:child"


@pytest.mark.asyncio
async def test_subagent_manager_blocks_nested_spawn_without_orchestrator_flag(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    parent = SubagentRecord(
        agent_id="sub_parent",
        label="parent",
        task="parent",
        state=SubagentState.RUNNING,
        depth=1,
        config=SubagentConfig(allow_subagents=False),
    )
    manager.registry.register(parent)

    with pytest.raises(ValueError, match="Nested sub-agent spawning is disabled"):
        await manager.spawn(
            task="child task",
            parent_id="sub_parent",
        )


@pytest.mark.asyncio
async def test_subagent_manager_allows_nested_spawn_for_orchestrator_flag(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    parent = SubagentRecord(
        agent_id="sub_parent",
        label="parent",
        task="parent",
        state=SubagentState.RUNNING,
        depth=1,
        config=SubagentConfig(allow_subagents=True),
    )
    manager.registry.register(parent)

    record = await manager.spawn(
        task="child task",
        parent_id="sub_parent",
    )

    assert record.parent_id == "sub_parent"
    assert record.depth == 2


def test_subagent_tool_context_clears_stale_channel():
    manager = SimpleNamespace(
        _current_spawner_session="session_stale",
        _current_spawner_channel="telegram:stale",
        _current_spawner_metadata={"chat_id": 1},
        _current_spawner_reply_to="old",
    )
    tool = SubagentTool(manager=manager, parent_agent_id="parent_3")
    tool.set_spawner_context(
        session_key="session_first",
        channel="telegram:first",
        metadata={"chat_id": 10},
        reply_to="new",
    )
    tool.set_spawner_context(
        session_key="session_second",
        channel=None,
        metadata={},
        reply_to=None,
    )

    assert tool._effective_spawner_session_key() == "session_second"
    assert tool._effective_spawner_channel() is None
    assert tool._effective_spawner_metadata() == {}
    assert tool._effective_spawner_reply_to() is None


@pytest.mark.asyncio
async def test_subagent_manager_resume_task_reparents_and_updates_depth(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        max_depth=6,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    parent_a = SubagentRecord(
        agent_id="sub_parent_a",
        label="parent a",
        task="parent a",
        state=SubagentState.COMPLETED,
        depth=1,
    )
    parent_b = SubagentRecord(
        agent_id="sub_parent_b",
        label="parent b",
        task="parent b",
        state=SubagentState.COMPLETED,
        depth=3,
        config=SubagentConfig(allow_subagents=True),
    )
    child = SubagentRecord(
        agent_id="sub_child",
        label="child",
        task="child",
        state=SubagentState.COMPLETED,
        parent_id="sub_parent_a",
        depth=2,
    )
    grandchild = SubagentRecord(
        agent_id="sub_grandchild",
        label="grandchild",
        task="grandchild",
        state=SubagentState.COMPLETED,
        parent_id="sub_child",
        depth=3,
    )

    manager.registry.register(parent_a)
    manager.registry.register(parent_b)
    manager.registry.register(child)
    manager.registry.register(grandchild)

    resumed = await manager.resume_task(
        task_id="sub_child",
        task="reparented task",
        parent_id="sub_parent_b",
    )

    assert resumed.parent_id == "sub_parent_b"
    assert resumed.depth == 4
    assert "sub_child" not in manager.registry.get("sub_parent_a").children
    assert "sub_child" in manager.registry.get("sub_parent_b").children
    assert manager.registry.get("sub_grandchild").depth == 5


@pytest.mark.asyncio
async def test_subagent_manager_resume_task_reparent_respects_max_depth(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        max_depth=3,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    parent = SubagentRecord(
        agent_id="sub_parent",
        label="parent",
        task="parent",
        state=SubagentState.COMPLETED,
        depth=3,
        config=SubagentConfig(allow_subagents=True),
    )
    child = SubagentRecord(
        agent_id="sub_child",
        label="child",
        task="child",
        state=SubagentState.COMPLETED,
        depth=1,
    )

    manager.registry.register(parent)
    manager.registry.register(child)

    with pytest.raises(ValueError, match="Max spawn depth"):
        await manager.resume_task(
            task_id="sub_child",
            task="reparent past limit",
            parent_id="sub_parent",
        )


@pytest.mark.asyncio
async def test_subagent_manager_collect_results_does_not_block_other_scopes(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    waiter = asyncio.create_task(
        manager.collect_results(timeout=0.25, spawner_session_key="session_root")
    )
    await asyncio.sleep(0.05)

    await manager._results.put(
        SubagentResult(
            agent_id="sub_child",
            label="child task",
            state=SubagentState.COMPLETED,
            result="child done",
            spawner_session_key="session_child",
            spawner_channel="cli",
        )
    )
    await asyncio.sleep(0.05)

    child_results = await asyncio.wait_for(
        manager.collect_results(spawner_session_key="session_child"),
        timeout=0.1,
    )
    root_results = await waiter

    assert [result.agent_id for result in child_results] == ["sub_child"]
    assert root_results == []


@pytest.mark.asyncio
async def test_subagent_manager_spawn_rejects_duplicate_session_agent_name(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        max_persistent_agents=4,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    await manager.spawn(
        task="initial task",
        config=SubagentConfig(
            spawn_mode=SpawnMode.SESSION,
            agent_name="planner",
        ),
    )

    with pytest.raises(ValueError, match="already exists"):
        await manager.spawn(
            task="duplicate task",
            config=SubagentConfig(
                spawn_mode=SpawnMode.SESSION,
                agent_name="planner",
            ),
        )


@pytest.mark.asyncio
async def test_subagent_manager_spawn_counts_pending_session_agents_against_limit(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        max_persistent_agents=1,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    await manager.spawn(
        task="initial task",
        config=SubagentConfig(
            spawn_mode=SpawnMode.SESSION,
            agent_name="planner",
        ),
    )

    with pytest.raises(ValueError, match="Max persistent agents"):
        await manager.spawn(
            task="another task",
            config=SubagentConfig(
                spawn_mode=SpawnMode.SESSION,
                agent_name="reviewer",
            ),
        )


@pytest.mark.asyncio
async def test_subagent_manager_announce_result_awaits_bus_publish(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    published: list[InboundMessage] = []

    class FakeBus:
        async def publish(self, message):
            published.append(message)
            return True

    manager.set_bus(FakeBus())
    manager.registry.register(
        SubagentRecord(
            agent_id="sub_publish",
            label="publish task",
            task="publish task",
            state=SubagentState.COMPLETED,
        )
    )

    await manager._announce_result(
        SubagentResult(
            agent_id="sub_publish",
            label="publish task",
            state=SubagentState.COMPLETED,
            result="done",
            spawner_session_key="session_root",
            spawner_channel="cli",
        )
    )

    assert len(published) == 1
    assert published[0].session_key == "session_root"


@pytest.mark.asyncio
async def test_run_subagent_announces_cancelled_result(workspace_dir, monkeypatch):
    import spoon_bot.agent.loop as loop_module

    started = asyncio.Event()

    class FakeAgentLoop:
        def __init__(self, *args, **kwargs):
            self.tools = {}

        async def initialize(self):
            return None

        async def process(self, task_text):
            started.set()
            await asyncio.Future()
            return task_text

        async def cleanup(self):
            return None

    monkeypatch.setattr(loop_module, "AgentLoop", FakeAgentLoop)

    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager._announce_result = AsyncMock()  # type: ignore[method-assign]

    record = SubagentRecord(
        agent_id="sub_cancelled",
        label="cancelled",
        task="cancelled",
        state=SubagentState.PENDING,
        spawner_session_key="session_root",
        spawner_channel="telegram:test",
    )
    manager.registry.register(record)

    task = asyncio.create_task(manager._run_subagent(record))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    manager._announce_result.assert_awaited_once()
    cancelled_result = manager._announce_result.await_args.args[0]
    assert cancelled_result.agent_id == "sub_cancelled"
    assert cancelled_result.state == SubagentState.CANCELLED


def test_session_manager_archive_hides_deleted_file_sessions(workspace_dir):
    manager = SessionManager(workspace=workspace_dir)
    session = Session(session_key="alpha")
    session.add_message("user", "hello")
    manager.save(session)

    assert "alpha" in manager.list_sessions()
    assert manager.archive("alpha") is True
    assert manager.get("alpha") is None
    assert "alpha" not in manager.list_sessions()
    assert not any(".deleted." in key for key in manager.list_sessions())
    assert any(".deleted." in path.name for path in (workspace_dir / "sessions").iterdir())


def test_session_manager_archive_exports_non_file_backends(workspace_dir):
    from spoon_bot.session.store import FileSessionStore, SQLiteSessionStore

    manager = SessionManager(
        workspace=workspace_dir,
        store=SQLiteSessionStore(str(workspace_dir / "archive_sessions.db")),
    )
    session = Session(session_key="alpha")
    session.add_message("user", "hello from sqlite")
    manager.save(session)

    assert manager.archive("alpha") is True
    assert manager.get("alpha") is None

    archive_store = FileSessionStore(workspace_dir / "sessions_archived")
    archived_keys = archive_store.list_session_keys(include_archived=True)
    assert archived_keys
    archived_session = archive_store.load_session(archived_keys[0])
    assert archived_session is not None
    assert archived_session.metadata["archived_from_session_key"] == "alpha"
    assert archived_session.messages[-1]["content"] == "hello from sqlite"


@pytest.mark.asyncio
async def test_telegram_subagent_spawn_passes_requester_context():
    from spoon_bot.channels.telegram.commands import CommandHandlers

    handlers = CommandHandlers(
        SimpleNamespace(
            account_id="spoon",
            full_name="telegram:spoon",
        )
    )
    manager = SimpleNamespace(
        spawn=AsyncMock(return_value=_tool_record(agent_id="sub_cmd"))
    )
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=12345),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )

    await handlers._subagents_spawn(update, manager, ["build", "something"])

    kwargs = manager.spawn.await_args.kwargs
    assert kwargs["spawner_session_key"] == "telegram_spoon_12345"
    assert kwargs["spawner_channel"] == "telegram:spoon"
    assert kwargs["spawner_metadata"]["chat_id"] == 12345


@pytest.mark.asyncio
async def test_telegram_subagent_resume_passes_requester_context():
    from spoon_bot.channels.telegram.commands import CommandHandlers

    handlers = CommandHandlers(
        SimpleNamespace(
            account_id="spoon",
            full_name="telegram:spoon",
        )
    )
    manager = SimpleNamespace(
        resume_agent=AsyncMock(return_value=_tool_record(agent_id="sub_resume"))
    )
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=67890),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )

    await handlers._subagents_resume(update, manager, "planner", "continue")

    kwargs = manager.resume_agent.await_args.kwargs
    assert kwargs["spawner_session_key"] == "telegram_spoon_67890"
    assert kwargs["spawner_channel"] == "telegram:spoon"
    assert kwargs["spawner_metadata"]["chat_id"] == 67890


@pytest.mark.asyncio
async def test_telegram_subagent_list_scopes_status_to_requester():
    from spoon_bot.channels.telegram.commands import CommandHandlers

    handlers = CommandHandlers(
        SimpleNamespace(
            account_id="spoon",
            full_name="telegram:spoon",
        )
    )
    manager = SimpleNamespace(
        get_status_summary=Mock(return_value={"total": 0, "active": [], "recent": []}),
    )
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=12345),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )

    await handlers._subagents_list(update, manager)

    manager.get_status_summary.assert_called_once_with(
        spawner_session_key="telegram_spoon_12345"
    )


@pytest.mark.asyncio
async def test_telegram_subagent_cancel_all_scopes_requester():
    from spoon_bot.channels.telegram.commands import CommandHandlers

    handlers = CommandHandlers(
        SimpleNamespace(
            account_id="spoon",
            full_name="telegram:spoon",
        )
    )
    manager = SimpleNamespace(
        cancel_all=AsyncMock(return_value=2),
    )
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=12345),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )

    await handlers._subagents_cancel(update, manager, "all")

    manager.cancel_all.assert_awaited_once_with(
        spawner_session_key="telegram_spoon_12345"
    )


@pytest.mark.asyncio
async def test_channel_manager_formats_rate_limit_error():
    from spoon_bot.utils.errors import RateLimitExceeded

    manager = ChannelManager()
    manager._agent = SimpleNamespace(
        process=AsyncMock(
            side_effect=RateLimitExceeded(
                resource="LLM",
                limit=1,
                window=60.0,
                retry_after=4.0,
            )
        ),
        process_with_thinking=AsyncMock(),
    )
    manager._channels["telegram:spoon"] = SimpleNamespace(
        on_processing_start=AsyncMock(),
        on_processing_end=AsyncMock(),
    )

    outbound = await manager._handle_message(
        InboundMessage(
            content="hello",
            channel="telegram:spoon",
            session_key="telegram_spoon_1",
            sender_id="1",
            message_id="m1",
            metadata={},
        )
    )

    assert "retry" in outbound.content.lower() or "wait" in outbound.content.lower()


def test_subagent_manager_disables_prompt_based_auto_route_specialist(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager.registry.register(
        SubagentRecord(
            agent_id="sub_auth",
            label="auth specialist",
            task="auth specialist",
            state=SubagentState.COMPLETED,
            config=SubagentConfig(
                spawn_mode=SpawnMode.SESSION,
                agent_name="auth-specialist",
                auto_route=True,
                specialization="Handles authentication, login, and password reset requests.",
                match_keywords=["password reset", "login issue", "authentication"],
                routing_mode=RoutingMode.DIRECT,
            ),
        )
    )
    manager.registry.register(
        SubagentRecord(
            agent_id="sub_billing",
            label="billing specialist",
            task="billing specialist",
            state=SubagentState.COMPLETED,
            config=SubagentConfig(
                spawn_mode=SpawnMode.SESSION,
                agent_name="billing-specialist",
                auto_route=True,
                specialization="Handles billing and invoice tasks.",
                match_keywords=["invoice", "billing"],
                routing_mode=RoutingMode.DIRECT,
            ),
        )
    )

    match = manager.find_best_auto_route_specialist(
        "Please help reset a user's password because login is failing."
    )

    assert match is None


def test_subagent_manager_parses_natural_language_creation_request(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    parsed = manager.parse_persistent_subagent_request(
        "帮我创建一个 subagent 来总结今天的新闻"
    )

    assert parsed is not None
    assert parsed["suggested_name"] == "news-subagent"
    assert parsed["specialization"] == "总结今天的新闻"
    assert "今天的新闻" in parsed["match_keywords"]


def test_subagent_manager_parses_common_chinese_creation_request_variant(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    parsed = manager.parse_persistent_subagent_request(
        "请帮我创建一个持久 subagent，专门总结当前仓库的 README 和 docs 文档。"
    )

    assert parsed is not None
    assert parsed["suggested_name"] == "readme-docs-subagent"
    assert parsed["specialization"] == "总结当前仓库的 README 和 docs 文档"
    assert "总结当前仓库的 README 和 docs 文档" in parsed["match_keywords"]


@pytest.mark.xfail(reason="legacy mojibake fixture; covered by stable profile tests below")
def test_subagent_manager_create_persistent_subagent_from_description(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    record = manager.create_persistent_subagent(
        description="总结今天的新闻",
    )

    assert record.agent_name == "news-subagent"
    assert record.spawn_mode == SpawnMode.SESSION
    assert record.state == SubagentState.COMPLETED
    assert record.config.auto_route is False
    assert record.config.tool_profile == "research"
    assert record.config.specialization == "总结今天的新闻"
    assert "今天的新闻" in record.config.match_keywords


def test_subagent_manager_skips_ambiguous_auto_route_specialist_match(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    for agent_name in ("invoice-a", "invoice-b"):
        manager.registry.register(
            SubagentRecord(
                agent_id=f"sub_{agent_name}",
                label=agent_name,
                task=agent_name,
                state=SubagentState.COMPLETED,
                config=SubagentConfig(
                    spawn_mode=SpawnMode.SESSION,
                    agent_name=agent_name,
                    auto_route=True,
                    specialization="Handles invoice review.",
                    match_keywords=["invoice review"],
                    routing_mode=RoutingMode.DIRECT,
                ),
            )
        )

    assert manager.find_best_auto_route_specialist("Please do an invoice review.") is None


@pytest.mark.asyncio
async def test_agent_loop_does_not_prompt_route_to_specialist(workspace_dir):
    pytest.importorskip("spoon_ai")
    from spoon_bot.agent.loop import AgentLoop

    loop = AgentLoop(
        workspace=workspace_dir,
        model="gpt-5.2",
        provider="openai",
        api_key="test-key",
    )
    loop._initialized = True
    loop._subagent_manager._current_spawner_channel = "telegram:spoon"
    loop._subagent_manager.find_best_auto_route_specialist = lambda message: {
        "agent_name": "auth-specialist",
        "score": 12,
        "reasons": ["keyword:password reset"],
    }
    loop._subagent_manager.dispatch_persistent_subagent = AsyncMock(
        return_value=SubagentRecord(
            agent_id="sub_auth",
            label="auth specialist",
            task="Reset password for alice",
            state=SubagentState.PENDING,
            config=SubagentConfig(
                spawn_mode=SpawnMode.SESSION,
                agent_name="auth-specialist",
                auto_route=True,
                specialization="Handles authentication tasks.",
            ),
        )
    )
    loop._subagent_manager.collect_results = AsyncMock(
        return_value=[
            SubagentResult(
                agent_id="sub_auth",
                label="auth specialist",
                state=SubagentState.COMPLETED,
                result="Password reset flow has been implemented.",
                spawner_session_key="root_session",
                spawner_channel="telegram:spoon",
            )
        ]
    )

    assert await loop._maybe_route_to_persistent_specialist("Reset password for alice") is None
    loop._subagent_manager.dispatch_persistent_subagent.assert_not_awaited()


@pytest.mark.asyncio
async def test_telegram_subagent_spawn_sets_specialist_metadata():
    from spoon_bot.channels.telegram.commands import CommandHandlers

    handlers = CommandHandlers(
        SimpleNamespace(
            account_id="spoon",
            full_name="telegram:spoon",
        )
    )
    manager = SimpleNamespace(
        spawn=AsyncMock(return_value=_tool_record(agent_id="sub_specialist"))
    )
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=999),
        message=SimpleNamespace(reply_text=AsyncMock()),
    )

    await handlers._subagents_spawn(
        update,
        manager,
        [
            "--mode", "session",
            "--name", "auth-specialist",
            "--specialization", "Handles", "authentication", "tasks",
            "--keywords", "login,password reset",
            "--auto-route",
            "handle", "auth",
        ],
    )

    kwargs = manager.spawn.await_args.kwargs
    config = kwargs["config"]
    assert config.spawn_mode == SpawnMode.SESSION
    assert config.agent_name == "auth-specialist"
    assert config.specialization == "Handles authentication tasks"
    assert config.auto_route is False
    assert config.match_keywords == ["login", "password reset"]


def test_subagent_manager_parses_creation_request_with_sender_prefix_english(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    parsed = manager.parse_persistent_subagent_request(
        "[Alice]: Create a persistent subagent to summarize today's news"
    )

    assert parsed is not None
    assert parsed["suggested_name"] == "news-subagent"


def test_subagent_manager_creates_research_profile_from_english_description(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    profile = manager.create_persistent_subagent(
        description="handle literature search, paper discovery, paper summaries, and research material collection",
    )

    assert profile.name == "academic-research-subagent"
    assert profile.auto_route is False
    assert profile.tool_profile == "research"
    assert "paper search" in profile.match_keywords


def test_subagent_manager_matches_research_profile_for_related_request(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager.create_persistent_subagent(
        description="handle literature search, paper discovery, paper summaries, and research material collection",
        agent_name="academic-research-subagent",
    )

    match = manager.find_best_auto_route_specialist(
        "Please find representative papers on multi-agent systems and provide a brief summary."
    )

    assert match is None


def test_subagent_manager_does_not_prompt_route_any_request(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager.create_persistent_subagent(
        description="总结当前仓库的 README 和 docs 文档",
        agent_name="readme-docs-subagent",
    )

    summary_match = manager.find_best_auto_route_specialist(
        "请总结当前仓库 README 的核心功能，只要 3 条。"
    )
    cron_match = manager.find_best_auto_route_specialist(
        "请帮我创建一个 corn 定时任务：每 5 分钟检查当前工作区的 README.md 是否存在，并在当前会话里回复 README_OK。"
    )

    assert summary_match is None
    assert cron_match is None


def test_subagent_manager_creates_research_profile_from_chinese_description(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    profile = manager.create_persistent_subagent(
        description="\u4eca\u540e\u4e13\u95e8\u5904\u7406\u6587\u732e\u67e5\u8be2\u3001\u8bba\u6587\u68c0\u7d22\u3001\u8bba\u6587\u6458\u8981\u6574\u7406\u548c\u7814\u7a76\u8d44\u6599\u6536\u96c6\u76f8\u5173\u4efb\u52a1",
    )

    assert profile.name == "academic-research-subagent"
    assert profile.tool_profile == "research"
    assert "\u6587\u732e\u67e5\u8be2" in profile.match_keywords


@pytest.mark.asyncio
async def test_subagent_manager_announce_result_preserves_spawner_metadata(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )

    published: list[InboundMessage] = []

    class FakeBus:
        async def publish(self, message):
            published.append(message)
            return True

    manager.set_bus(FakeBus())
    manager.registry.register(
        SubagentRecord(
            agent_id="sub_publish_meta",
            label="publish task",
            task="publish task",
            state=SubagentState.COMPLETED,
        )
    )

    await manager._announce_result(
        SubagentResult(
            agent_id="sub_publish_meta",
            label="publish task",
            state=SubagentState.COMPLETED,
            result="done",
            spawner_session_key="telegram_spoon_1",
            spawner_channel="telegram:spoon",
            spawner_metadata={"chat_id": 1001, "chat_type": "private"},
            spawner_reply_to="777",
        )
    )

    assert len(published) == 1
    assert published[0].metadata["chat_id"] == 1001
    assert published[0].metadata["reply_to"] == "777"
    assert published[0].metadata["subagent_run_id"].startswith("run_")
    assert "run_id:" in published[0].content


@pytest.mark.asyncio
async def test_channel_manager_uses_metadata_reply_target():
    manager = ChannelManager()
    manager._agent = SimpleNamespace(
        process=AsyncMock(return_value="hello"),
        process_with_thinking=AsyncMock(),
    )
    manager._channels["telegram:spoon"] = SimpleNamespace(
        on_processing_start=AsyncMock(),
        on_processing_end=AsyncMock(),
    )

    outbound = await manager._handle_message(
        InboundMessage(
            content="hello",
            channel="telegram:spoon",
            session_key="telegram_spoon_1",
            sender_id="1",
            message_id="internal_wake_id",
            metadata={"chat_id": 1001, "reply_to": "42"},
        )
    )

    assert outbound.reply_to == "42"


def test_subagent_manager_does_not_route_from_persistent_profile_without_runtime_record(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager.create_persistent_subagent(
        description="专门处理文献查询、论文检索、论文摘要整理和研究资料收集",
        agent_name="academic-research-subagent",
        config=SubagentConfig(
            tool_profile="research",
            auto_route=True,
            specialization="专门处理文献查询、论文检索、论文摘要整理和研究资料收集",
            match_keywords=["文献查询", "论文检索", "论文摘要", "研究资料", "学术研究"],
        ),
    )

    reloaded = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    match = reloaded.find_best_auto_route_specialist(
        "帮我查一下多智能体系统方向最近的代表性论文，并做一个简要总结。"
    )

    assert match is None


@pytest.mark.asyncio
async def test_resume_agent_uses_persistent_profile_when_runtime_record_missing(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager._start_background_run = lambda *args, **kwargs: None  # type: ignore[method-assign]
    manager.create_persistent_subagent(
        description="summarize today's news",
        agent_name="news-subagent",
        config=SubagentConfig(
            tool_profile="research",
            auto_route=True,
            specialization="summarize today's news",
            match_keywords=["today's news", "daily news", "news summary"],
        ),
    )

    spawned: list[tuple[str, SubagentConfig]] = []

    original_spawn = manager.spawn

    async def wrapped_spawn(*, task, config=None, **kwargs):
        spawned.append((task, config))
        return await original_spawn(task=task, config=config, **kwargs)

    manager.spawn = wrapped_spawn  # type: ignore[method-assign]

    record = await manager.resume_agent(
        agent_name="news-subagent",
        task="Please summarize today's news",
        parent_id="sub_parent_restored",
    )

    assert record.agent_name == "news-subagent"
    assert spawned
    assert spawned[0][1] is not None
    assert spawned[0][1].spawn_mode == SpawnMode.SESSION
    assert record.parent_id == "sub_parent_restored"


@pytest.mark.asyncio
async def test_resume_agent_reuses_persistent_profile_session_key_and_migrates_legacy_history(
    workspace_dir,
):
    shared_sessions = SessionManager(workspace=workspace_dir)
    legacy_session = shared_sessions.get_or_create("subagent-sub_legacy_news")
    legacy_session.add_message("assistant", "kept transcript")
    shared_sessions.save(legacy_session)

    manager = SubagentManager(
        session_manager=shared_sessions,
        workspace=workspace_dir,
    )
    manager._start_background_run = lambda *args, **kwargs: None  # type: ignore[method-assign]
    profile = manager.create_persistent_subagent(
        description="summarize today's news",
        agent_name="news-subagent",
    )
    profile.session_key = "subagent-sub_legacy_news"
    profile.last_run_agent_id = "sub_legacy_news"
    manager._save_persistent_profile(profile)

    record = await manager.resume_agent(
        agent_name="news-subagent",
        task="Please summarize today's news",
    )

    scoped_sessions = manager._session_manager_for_record(record)
    migrated = scoped_sessions.get(record.session_key)

    assert record.session_key == "subagent-sub_legacy_news"
    assert migrated is not None
    assert migrated.messages[-1]["content"] == "kept transcript"
    assert shared_sessions.get(record.session_key) is None


@pytest.mark.asyncio
async def test_resume_agent_enforces_total_active_limit(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        max_total_subagents=1,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    active = SubagentRecord(
        agent_id="sub_active",
        label="active",
        task="active",
        state=SubagentState.RUNNING,
    )
    completed = SubagentRecord(
        agent_id="sub_session",
        label="session",
        task="session",
        state=SubagentState.COMPLETED,
        config=SubagentConfig(
            spawn_mode=SpawnMode.SESSION,
            agent_name="planner",
        ),
        agent_name="planner",
    )

    manager.registry.register(active)
    manager.registry.register(completed)

    with pytest.raises(ValueError, match="Max total active sub-agents"):
        await manager.resume_agent(
            agent_name="planner",
            task="resume despite limit",
        )


@pytest.mark.asyncio
async def test_resume_agent_discovers_missing_profile_session_key_from_agent_scope(
    workspace_dir,
):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager._start_background_run = lambda *args, **kwargs: None  # type: ignore[method-assign]
    profile = manager.create_persistent_subagent(
        description="handle literature search and paper summaries",
        agent_name="research-subagent",
    )
    profile.last_run_agent_id = "sub_saved_profile"
    manager._save_persistent_profile(profile)

    scoped_sessions = manager._session_manager_for_record(
        SubagentRecord(
            agent_id="sub_saved_profile",
            session_key="subagent-sub_saved_profile",
            label="research",
            task="research",
            state=SubagentState.COMPLETED,
            config=SubagentConfig(
                spawn_mode=SpawnMode.SESSION,
                agent_name="research-subagent",
            ),
        )
    )
    scoped_session = scoped_sessions.get_or_create("subagent-sub_saved_profile")
    scoped_session.add_message("assistant", "scoped transcript")
    scoped_sessions.save(scoped_session)

    reloaded = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    reloaded._start_background_run = lambda *args, **kwargs: None  # type: ignore[method-assign]

    record = await reloaded.resume_agent(
        agent_name="research-subagent",
        task="Continue the literature review",
    )

    assert record.session_key == "subagent-sub_saved_profile"
    assert reloaded._get_persistent_profile("research-subagent").session_key == "subagent-sub_saved_profile"


def test_create_persistent_subagent_enforces_cap(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        max_persistent_agents=1,
    )

    manager.create_persistent_subagent(
        description="summarize today's news",
        agent_name="news-subagent",
    )

    with pytest.raises(ValueError, match="Max persistent agents"):
        manager.create_persistent_subagent(
            description="handle literature search and paper summaries",
            agent_name="research-subagent",
        )


def test_subagent_manager_builds_leaf_and_orchestrator_prompts(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        max_depth=2,
    )

    leaf_prompt = manager._build_system_prompt(
        SubagentRecord(
            agent_id="sub_leaf",
            run_id="run_leaf",
            label="leaf",
            task="inspect files",
            depth=1,
            config=SubagentConfig(),
        )
    )
    orchestrator_prompt = manager._build_system_prompt(
        SubagentRecord(
            agent_id="sub_orchestrator",
            run_id="run_orchestrator",
            label="orchestrator",
            task="coordinate workers",
            depth=1,
            config=SubagentConfig(allow_subagents=True),
        )
    )

    assert "leaf worker" in leaf_prompt.lower()
    assert "run_leaf" in leaf_prompt
    assert "explicitly allowed to spawn nested sub-agents" in orchestrator_prompt


def test_subagent_manager_stacks_runtime_prompt_on_custom_prompt(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        max_depth=2,
    )
    record = SubagentRecord(
        agent_id="sub_custom",
        run_id="run_custom",
        label="custom",
        task="inspect files",
        depth=1,
        config=SubagentConfig(
            system_prompt="You are a specialist reviewer.",
            allow_subagents=True,
        ),
    )

    prompt = manager._compose_system_prompt(record)

    assert prompt.startswith("You are a specialist reviewer.")
    assert "# Subagent Context" in prompt
    assert "auto-announced back to the requester" in prompt
    assert "explicitly allowed to spawn nested sub-agents" in prompt


@pytest.mark.asyncio
async def test_subagent_manager_applies_default_model_and_tool_profile(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        default_model="gpt-subagent-default",
        default_tool_profile="research",
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    defaulted = await manager.spawn(task="default task")
    explicit = await manager.spawn(
        task="explicit task",
        config=SubagentConfig(
            model="gpt-explicit",
            tool_profile="coding",
        ),
    )

    assert defaulted.model_name == "gpt-subagent-default"
    assert defaulted.config.model == "gpt-subagent-default"
    assert defaulted.config.tool_profile == "research"
    assert explicit.model_name == "gpt-explicit"
    assert explicit.config.tool_profile == "coding"


def test_subagent_manager_inherits_parent_skill_setting_by_default(workspace_dir):
    enabled_manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        parent_enable_skills=True,
    )
    disabled_manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
        parent_enable_skills=False,
    )

    assert enabled_manager._resolve_effective_enable_skills(SubagentConfig()) is True
    assert disabled_manager._resolve_effective_enable_skills(SubagentConfig()) is False
    assert (
        disabled_manager._resolve_effective_enable_skills(
            SubagentConfig(enable_skills=True)
        )
        is True
    )


def test_subagent_manager_status_summary_filters_requester_lineage(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager.registry.register(
        SubagentRecord(
            agent_id="sub_root_owned",
            label="root owned",
            task="root",
            state=SubagentState.RUNNING,
            spawner_session_key="session_root",
        )
    )
    manager.registry.register(
        SubagentRecord(
            agent_id="sub_desc_owned",
            parent_id="sub_root_owned",
            label="desc owned",
            task="desc",
            state=SubagentState.PENDING,
            spawner_session_key="subagent-sub_root_owned",
        )
    )
    manager.registry.register(
        SubagentRecord(
            agent_id="sub_other",
            label="other",
            task="other",
            state=SubagentState.RUNNING,
            spawner_session_key="session_other",
        )
    )

    summary = manager.get_status_summary(spawner_session_key="session_root")

    visible_ids = {item["agent_id"] for item in summary["active"] + summary["recent"]}
    assert visible_ids == {"sub_root_owned", "sub_desc_owned"}


@pytest.mark.asyncio
async def test_subagent_manager_scopes_info_cancel_and_steer_to_requester_lineage(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    owned = SubagentRecord(
        agent_id="sub_owned",
        label="owned",
        task="owned",
        state=SubagentState.RUNNING,
        spawner_session_key="session_root",
    )
    other = SubagentRecord(
        agent_id="sub_other",
        label="other",
        task="other",
        state=SubagentState.RUNNING,
        spawner_session_key="session_other",
    )
    manager.registry.register(owned)
    manager.registry.register(other)

    owned_task = asyncio.create_task(asyncio.sleep(10))
    other_task = asyncio.create_task(asyncio.sleep(10))
    manager._tasks[owned.agent_id] = owned_task
    manager._tasks[other.agent_id] = other_task

    try:
        info = await manager.get_info("sub_other", spawner_session_key="session_root")
        cancel_other = await manager.cancel(
            "sub_other",
            cascade=True,
            spawner_session_key="session_root",
        )
        steer_other = await manager.steer(
            "sub_other",
            "redirect",
            spawner_session_key="session_root",
        )
        steer_owned = await manager.steer(
            "sub_owned",
            "redirect",
            spawner_session_key="session_root",
        )

        assert info is None
        assert cancel_other is False
        assert steer_other["status"] == "not_found"
        assert steer_owned["status"] == "accepted"
    finally:
        owned_task.cancel()
        other_task.cancel()
        await asyncio.gather(owned_task, other_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_subagent_manager_disables_orchestrated_prompt_auto_route(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager.registry.register(
        SubagentRecord(
            agent_id="sub_orchestrated",
            label="research orchestrator",
            task="research orchestrator",
            state=SubagentState.COMPLETED,
            config=SubagentConfig(
                spawn_mode=SpawnMode.SESSION,
                agent_name="research-orchestrator",
                auto_route=True,
                allow_subagents=True,
                specialization="Handles literature search and parallel paper review.",
                match_keywords=["literature search", "paper review", "research summary"],
                routing_mode=RoutingMode.ORCHESTRATED,
            ),
        )
    )

    match = manager.find_best_auto_route_specialist(
        "Please do a literature search and summarize the most relevant papers."
    )

    assert match is None


@pytest.mark.asyncio
async def test_subagent_manager_orchestrated_parent_can_spawn_nested_workers(
    workspace_dir,
    monkeypatch,
):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    monkeypatch.setattr(manager, "_start_background_run", lambda *args, **kwargs: None)

    parent = SubagentRecord(
        agent_id="sub_orchestrated_parent",
        label="research orchestrator",
        task="coordinate paper review",
        state=SubagentState.RUNNING,
        depth=1,
        config=SubagentConfig(
            routing_mode=RoutingMode.ORCHESTRATED,
            allow_subagents=False,
        ),
    )
    manager.registry.register(parent)

    child = await manager.spawn(
        task="review paper cluster",
        parent_id=parent.agent_id,
    )

    assert child.parent_id == parent.agent_id
    assert child.depth == 2


@pytest.mark.asyncio
async def test_dispatch_persistent_subagent_wraps_orchestrated_request(workspace_dir):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
    )
    manager.create_persistent_subagent(
        description="handle literature search and parallel paper review",
        agent_name="research-orchestrator",
        config=SubagentConfig(
            auto_route=True,
            routing_mode=RoutingMode.ORCHESTRATED,
            specialization="Handle literature search and parallel paper review.",
            match_keywords=["literature search", "paper review"],
        ),
    )
    manager.resume_agent = AsyncMock(  # type: ignore[method-assign]
        return_value=SubagentRecord(
            agent_id="sub_dispatch",
            label="summarize papers",
            task="summarize papers",
            state=SubagentState.PENDING,
            config=SubagentConfig(
                spawn_mode=SpawnMode.SESSION,
                agent_name="research-orchestrator",
            ),
        )
    )

    await manager.dispatch_persistent_subagent(
        agent_name="research-orchestrator",
        task="Summarize the latest papers on multi-agent systems.",
    )

    kwargs = manager.resume_agent.await_args.kwargs
    assert kwargs["label"] == "Summarize the latest papers on multi-agent systems."[:60]
    assert "owning orchestrator" in kwargs["task"]
    assert "User request:" in kwargs["task"]


@pytest.mark.asyncio
async def test_subagent_manager_run_process_falls_back_when_thinking_is_unsupported():
    child_agent = SimpleNamespace(
        process_with_thinking=AsyncMock(
            side_effect=TypeError("run() got an unexpected keyword argument 'thinking'")
        ),
        process=AsyncMock(return_value="plain result"),
    )

    result = await SubagentManager._run_process(
        child_agent,
        "analyze",
        SubagentConfig(thinking_level="extended"),
    )

    assert result == "plain result"
    child_agent.process.assert_awaited_once_with("analyze")


@pytest.mark.asyncio
async def test_subagent_manager_run_process_passes_requested_thinking_level():
    child_agent = SimpleNamespace(
        process_with_thinking=AsyncMock(return_value=("thoughtful result", "trace")),
        process=AsyncMock(),
    )

    result = await SubagentManager._run_process(
        child_agent,
        "analyze",
        SubagentConfig(thinking_level="extended"),
    )

    assert result == "thoughtful result"
    child_agent.process_with_thinking.assert_awaited_once_with(
        "analyze",
        thinking_level="extended",
    )
    child_agent.process.assert_not_called()


@pytest.mark.asyncio
async def test_channel_manager_passes_metadata_to_set_subagent_context():
    manager = ChannelManager()
    agent = SimpleNamespace(
        process=AsyncMock(return_value="hello"),
        process_with_thinking=AsyncMock(),
        set_subagent_context=Mock(),
    )
    manager._agent = agent
    manager._channels["telegram:spoon"] = SimpleNamespace(
        on_processing_start=AsyncMock(),
        on_processing_end=AsyncMock(),
    )

    await manager._handle_message(
        InboundMessage(
            content="hello",
            channel="telegram:spoon",
            session_key="telegram_spoon_1",
            sender_id="1",
            message_id="m1",
            metadata={"chat_id": 1001, "reply_to": "42"},
        )
    )

    agent.set_subagent_context.assert_called_once_with(
        session_key="telegram_spoon_1",
        channel="telegram:spoon",
        metadata={"chat_id": 1001, "reply_to": "42"},
        reply_to="m1",
    )
    agent.process.assert_awaited_once_with(
        message="hello",
        media=None,
        session_key="telegram_spoon_1",
        channel="telegram:spoon",
        metadata={"chat_id": 1001, "reply_to": "42"},
        reply_to="m1",
        attachments=None,
    )


def test_set_subagent_context_clears_stale_channel_metadata_when_omitted():
    from spoon_bot.agent.loop import AgentLoop
    from spoon_bot.subagent.tools import SubagentTool

    manager = SimpleNamespace(set_spawner_context=Mock())
    spawn_tool = SubagentTool(manager=manager)

    loop = AgentLoop.__new__(AgentLoop)
    loop.session_key = "default"
    loop._subagent_manager = manager
    loop.tools = {"spawn": spawn_tool}

    AgentLoop.set_subagent_context(
        loop,
        session_key="telegram-session",
        channel="telegram:spoon",
        metadata={"chat_id": 1001},
        reply_to="m1",
    )
    AgentLoop.set_subagent_context(loop, session_key="rest-session")

    assert spawn_tool._spawner_session_key == "rest-session"
    assert spawn_tool._spawner_channel is None
    assert spawn_tool._spawner_metadata == {}
    assert spawn_tool._spawner_reply_to is None
    assert manager.set_spawner_context.call_args_list[-1].kwargs == {
        "session_key": "rest-session",
        "channel": None,
        "metadata": None,
        "reply_to": None,
    }


def test_persistent_session_storage_migrates_legacy_shared_transcript(workspace_dir):
    shared_sessions = SessionManager(workspace=workspace_dir)
    legacy_session = shared_sessions.get_or_create("subagent-sub_persist")
    legacy_session.add_message("user", "legacy hello")
    shared_sessions.save(legacy_session)

    manager = SubagentManager(
        session_manager=shared_sessions,
        workspace=workspace_dir,
    )
    record = SubagentRecord(
        agent_id="sub_persist",
        session_key="subagent-sub_persist",
        label="planner",
        task="planner task",
        state=SubagentState.COMPLETED,
        config=SubagentConfig(
            spawn_mode=SpawnMode.SESSION,
            agent_name="planner",
        ),
    )
    manager.registry.register(record)

    scoped_sessions = manager._session_manager_for_record(record)
    migrated = scoped_sessions.get(record.session_key)
    agent_root = workspace_dir / "agents" / "planner"
    scoped_file = agent_root / "sessions" / "subagent-sub_persist.jsonl"
    archived_legacy = list((workspace_dir / "sessions").glob("subagent-sub_persist.deleted.*.jsonl"))

    assert scoped_sessions is not shared_sessions
    assert scoped_sessions.sessions_dir == agent_root / "sessions"
    assert migrated is not None
    assert migrated.messages[-1]["content"] == "legacy hello"
    assert scoped_file.exists()
    assert archived_legacy
    assert shared_sessions.get(record.session_key) is None
    assert record.agent_dir == str(agent_root)


def test_persistent_session_storage_writes_new_transcripts_only_to_agent_scope(workspace_dir):
    shared_sessions = SessionManager(workspace=workspace_dir)
    manager = SubagentManager(
        session_manager=shared_sessions,
        workspace=workspace_dir,
    )
    record = SubagentRecord(
        agent_id="sub_persist_new",
        session_key="subagent-sub_persist_new",
        label="research",
        task="research task",
        state=SubagentState.PENDING,
        config=SubagentConfig(
            spawn_mode=SpawnMode.SESSION,
            agent_name="researcher",
        ),
    )
    manager.registry.register(record)

    scoped_sessions = manager._session_manager_for_record(record)
    scoped_session = scoped_sessions.get_or_create(record.session_key)
    scoped_session.add_message("assistant", "agent-scoped transcript")
    scoped_sessions.save(scoped_session)

    scoped_file = workspace_dir / "agents" / "researcher" / "sessions" / "subagent-sub_persist_new.jsonl"
    shared_file = workspace_dir / "sessions" / "subagent-sub_persist_new.jsonl"

    assert scoped_file.exists()
    assert not shared_file.exists()
    assert shared_sessions.get(record.session_key) is None


def test_persistent_session_storage_migrates_from_shared_sqlite_backend(workspace_dir):
    from spoon_bot.session.store import SQLiteSessionStore

    shared_sessions = SessionManager(
        workspace=workspace_dir,
        store=SQLiteSessionStore(str(workspace_dir / "shared_sessions.db")),
    )
    legacy_session = shared_sessions.get_or_create("subagent-sub_sqlite")
    legacy_session.add_message("user", "sqlite legacy")
    shared_sessions.save(legacy_session)

    manager = SubagentManager(
        session_manager=shared_sessions,
        workspace=workspace_dir,
    )
    record = SubagentRecord(
        agent_id="sub_sqlite",
        session_key="subagent-sub_sqlite",
        label="sqlite planner",
        task="planner task",
        state=SubagentState.COMPLETED,
        config=SubagentConfig(
            spawn_mode=SpawnMode.SESSION,
            agent_name="sqlite-planner",
        ),
    )
    manager.registry.register(record)

    scoped_sessions = manager._session_manager_for_record(record)
    migrated = scoped_sessions.get(record.session_key)
    scoped_file = workspace_dir / "agents" / "sqlite-planner" / "sessions" / "subagent-sub_sqlite.jsonl"

    assert migrated is not None
    assert migrated.messages[-1]["content"] == "sqlite legacy"
    assert scoped_file.exists()
    assert shared_sessions.get(record.session_key) is None
