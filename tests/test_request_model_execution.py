import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from spoon_bot.agent.execution_options import (
    AgentExecutionOptions,
    bind_agent_execution_options,
    current_agent_execution_options,
    request_model_execution,
)
from spoon_bot.agent.loop import AgentLoop
from spoon_bot.subagent.manager import SubagentManager
from spoon_bot.subagent.models import SubagentConfig


class _ExecutionProbe:
    model = "startup-model"
    context_window = 100_000

    def __init__(self, barrier: asyncio.Event, entered: list[str]) -> None:
        self.barrier = barrier
        self.entered = entered

    @request_model_execution
    async def run(
        self,
        *,
        model: str | None = None,
        model_sku: str | None = None,
        model_catalog_version: str | None = None,
        context_window: int | None = None,
    ) -> tuple[AgentExecutionOptions, AgentExecutionOptions]:
        before = current_agent_execution_options()
        assert before is not None
        self.entered.append(before.model)
        if len(self.entered) == 2:
            self.barrier.set()
        await asyncio.wait_for(self.barrier.wait(), timeout=1)
        after = current_agent_execution_options()
        assert after is not None
        return before, after


class _SnapshotProbe:
    model = "startup-model"
    context_window = 100_000

    @request_model_execution
    async def run(
        self,
        *,
        model: str | None = None,
        context_window: int | None = None,
    ) -> AgentExecutionOptions:
        options = current_agent_execution_options()
        assert options is not None
        return options

    @request_model_execution
    async def fail(self, *, model: str | None = None) -> None:
        assert current_agent_execution_options() is not None
        raise RuntimeError("request failed")

    @request_model_execution
    async def stream(
        self,
        *,
        model: str | None = None,
    ):
        first = current_agent_execution_options()
        assert first is not None
        yield first
        second = current_agent_execution_options()
        assert second is not None
        yield second


@pytest.mark.asyncio
async def test_request_models_are_isolated_across_concurrent_tasks() -> None:
    barrier = asyncio.Event()
    entered: list[str] = []
    probe = _ExecutionProbe(barrier, entered)

    first, second = await asyncio.gather(
        probe.run(
            model="provider/model-a",
            model_sku="model-a",
            context_window=111_000,
        ),
        probe.run(
            model="provider/model-b",
            model_sku="model-b",
            context_window=222_000,
        ),
    )

    assert first[0] == first[1]
    assert second[0] == second[1]
    assert first[0].model == "provider/model-a"
    assert first[0].context_window == 111_000
    assert second[0].model == "provider/model-b"
    assert second[0].context_window == 222_000
    assert current_agent_execution_options() is None


@pytest.mark.asyncio
async def test_sequential_requests_restore_startup_model() -> None:
    probe = _SnapshotProbe()

    first = await probe.run(model="provider/model-a")
    second = await probe.run(model="provider/model-b")
    default = await probe.run()

    assert first.model == "provider/model-a"
    assert second.model == "provider/model-b"
    assert default.model == "startup-model"
    assert current_agent_execution_options() is None


@pytest.mark.asyncio
async def test_execution_options_are_cleared_after_exception() -> None:
    probe = _SnapshotProbe()

    with pytest.raises(RuntimeError, match="request failed"):
        await probe.fail(model="provider/model-a")

    assert current_agent_execution_options() is None


@pytest.mark.asyncio
async def test_stream_context_is_cleared_between_yields() -> None:
    probe = _SnapshotProbe()
    stream = probe.stream(model="provider/stream-model")

    first = await anext(stream)
    assert first.model == "provider/stream-model"
    assert current_agent_execution_options() is None

    second = await anext(stream)
    assert second.model == "provider/stream-model"
    assert current_agent_execution_options() is None

    await stream.aclose()
    assert current_agent_execution_options() is None


def test_subagent_without_explicit_model_inherits_request_model(tmp_path: Path) -> None:
    manager = SubagentManager(
        session_manager=object(),
        workspace=tmp_path,
        parent_model="startup-model",
        default_model="subagent-default",
        persist_runs=False,
    )
    options = AgentExecutionOptions(
        model="provider/request-model",
        model_sku="request-model",
        model_catalog_version="v3",
        context_window=131_072,
        explicit=True,
    )

    with bind_agent_execution_options(options):
        inherited = manager._apply_default_config(SubagentConfig())
        explicit = manager._apply_default_config(
            SubagentConfig(model="provider/explicit-model")
        )

    assert inherited.model == "provider/request-model"
    assert explicit.model == "provider/explicit-model"


def test_execution_options_reject_invalid_context_window() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        AgentExecutionOptions.resolve(
            default_model="startup-model",
            default_context_window=100_000,
            model="request-model",
            context_window=0,
        )

    with pytest.raises(ValueError, match="must not exceed"):
        AgentExecutionOptions.resolve(
            default_model="startup-model",
            default_context_window=100_000,
            context_window=10_000_001,
        )


def test_compaction_and_prompt_use_request_context_window() -> None:
    loop = AgentLoop.__new__(AgentLoop)
    loop.context_window = 100_000
    loop.grounding = SimpleNamespace(
        context_compaction_ratio=0.75,
        output_reserve_ratio=0.2,
    )
    base_prompt = "System prompt\n\n[Context: 100,000 tokens - be concise.]"
    loop._agent = SimpleNamespace(
        system_prompt=base_prompt,
        _original_system_prompt=base_prompt,
    )
    loop._build_request_context_prompt = lambda _message: "request metadata"
    options = AgentExecutionOptions(
        model="provider/request-model",
        context_window=200_000,
        explicit=True,
    )

    with bind_agent_execution_options(options):
        assert loop._runtime_compaction_trigger_budget() == 150_000
        previous = loop._apply_request_context_to_system_prompt(
            "hello",
            thinking=False,
        )
        assert "[Context: 200,000 tokens - be concise.]" in loop._agent.system_prompt

    loop._restore_request_context_system_prompt(*previous)
    assert loop._agent.system_prompt == base_prompt
