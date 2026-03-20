from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
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
    label: str = "planner: analyse",
    depth: int = 1,
    model_name: str | None = "test-model",
    spawn_mode: str = "run",
    agent_name: str | None = None,
):
    return SimpleNamespace(
        agent_id=agent_id,
        label=label,
        depth=depth,
        model_name=model_name,
        spawn_mode=SimpleNamespace(value=spawn_mode),
        agent_name=agent_name,
        config=SubagentConfig(),
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
    tool.set_spawner_context(session_key="session_root", channel="telegram:bot")

    result = await tool.execute(action="spawn", task="Continue implementation", task_id="sub_prev")

    manager.resume_task.assert_awaited_once()
    kwargs = manager.resume_task.await_args.kwargs
    assert kwargs["task_id"] == "sub_prev"
    assert kwargs["task"] == "Continue implementation"
    assert kwargs["parent_id"] == "parent_1"
    assert kwargs["spawner_session_key"] == "session_root"
    assert kwargs["spawner_channel"] == "telegram:bot"
    assert kwargs["config"] is None
    manager.spawn.assert_not_called()
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
    tool.set_spawner_context(session_key="session_child", channel="discord:test")

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
    assert "Task ID: sub_session" in result


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
        config=original_config,
        model_name="model-a",
    )
    manager.registry.register(record)

    resumed = await manager.resume_task(
        task_id="sub_existing",
        task="new task",
        spawner_session_key="session_nested",
        spawner_channel="telegram:test",
    )

    assert resumed.agent_id == "sub_existing"
    assert resumed.task == "new task"
    assert resumed.state == SubagentState.PENDING
    assert resumed.config.tool_profile == "research"
    assert resumed.config.max_iterations == 12
    assert resumed.spawner_session_key == "session_nested"
    assert resumed.spawner_channel == "telegram:test"


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


def test_subagent_tool_context_clears_stale_channel():
    manager = SimpleNamespace(
        _current_spawner_session="session_stale",
        _current_spawner_channel="telegram:stale",
    )
    tool = SubagentTool(manager=manager, parent_agent_id="parent_3")
    tool.set_spawner_context(session_key="session_first", channel="telegram:first")
    tool.set_spawner_context(session_key="session_second", channel=None)

    assert tool._effective_spawner_session_key() == "session_second"
    assert tool._effective_spawner_channel() is None


@pytest.mark.asyncio
async def test_subagent_manager_resume_task_reparents_and_updates_depth(workspace_dir, monkeypatch):
    manager = SubagentManager(
        session_manager=SessionManager(workspace=workspace_dir),
        workspace=workspace_dir,
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


def test_session_manager_archive_hides_deleted_file_sessions(workspace_dir):
    manager = SessionManager(workspace=workspace_dir)
    session = Session(session_key="alpha")
    session.add_message("user", "hello")
    manager.save(session)

    assert "alpha" in manager.list_sessions()
    assert manager.archive("alpha") is True
    assert manager.get("alpha") is None
    assert "alpha" not in manager.list_sessions()
    assert any(".deleted." in path.name for path in (workspace_dir / "sessions").iterdir())


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


def test_subagent_manager_finds_best_auto_route_specialist(workspace_dir):
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

    assert match is not None
    assert match["agent_name"] == "auth-specialist"
    assert match["score"] >= 7


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
    assert record.config.auto_route is True
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
async def test_agent_loop_process_auto_routes_to_specialist(workspace_dir):
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
    loop._subagent_manager.resume_agent = AsyncMock(
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

    result = await loop.process(
        "Reset password for alice",
        session_key="root_session",
    )

    assert "auth-specialist" in result
    assert "Password reset flow has been implemented." in result
    loop._subagent_manager.resume_agent.assert_awaited_once()
    resume_kwargs = loop._subagent_manager.resume_agent.await_args.kwargs
    assert resume_kwargs["agent_name"] == "auth-specialist"
    assert resume_kwargs["task"] == "Reset password for alice"
    assert resume_kwargs["spawner_session_key"] == "root_session"
    assert resume_kwargs["spawner_channel"] == "telegram:spoon"


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
    assert config.auto_route is True
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
    assert profile.auto_route is True
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

    assert match is not None
    assert match["agent_name"] == "academic-research-subagent"


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


def test_subagent_manager_routes_from_persistent_profile_without_runtime_record(workspace_dir):
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

    assert match is not None
    assert match["agent_name"] == "academic-research-subagent"


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
    )

    assert record.agent_name == "news-subagent"
    assert spawned
    assert spawned[0][1] is not None
    assert spawned[0][1].spawn_mode == SpawnMode.SESSION
