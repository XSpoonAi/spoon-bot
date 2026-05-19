from __future__ import annotations

from types import SimpleNamespace

import pytest

from spoon_bot.agent.loop import AgentLoop


def _completed_tool_history() -> list[dict]:
    return [
        {
            "role": "user",
            "content": "verify",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"SKILL.md"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read",
            "name": "read_file",
            "content": "skill header",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_shell",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command":"ls"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_shell",
            "name": "shell",
            "content": "EXISTS",
        },
    ]


@pytest.mark.asyncio
async def test_openrouter_strict_tool_guard_disables_parallel_and_serializes_response():
    captured_kwargs: list[dict] = []
    tool_a = SimpleNamespace(id="call_a")
    tool_b = SimpleNamespace(id="call_b")

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return SimpleNamespace(tool_calls=[tool_a, tool_b], metadata={})

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = "openrouter"
    loop.model = "anthropic/claude-sonnet-4.6"
    loop.base_url = "https://openrouter.ai/api/v1"
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    response = await loop._agent.llm.ask_tool(messages=[], tools=[])

    assert captured_kwargs[0]["parallel_tool_calls"] is False
    assert response.tool_calls == [tool_a]
    assert response.metadata["serial_tool_calls_enforced"] is True
    assert response.metadata["dropped_parallel_tool_calls"] == 1


@pytest.mark.asyncio
async def test_tool_guard_retries_without_parallel_flag_when_provider_rejects_it():
    captured_kwargs: list[dict] = []

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            if len(captured_kwargs) == 1:
                raise RuntimeError("unsupported parameter: parallel_tool_calls")
            return SimpleNamespace(tool_calls=[], metadata={})

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = "openrouter"
    loop.model = "anthropic/claude-sonnet-4.6"
    loop.base_url = "https://openrouter.ai/api/v1"
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    response = await loop._agent.llm.ask_tool(messages=[], tools=[])

    assert response.tool_calls == []
    assert captured_kwargs[0]["parallel_tool_calls"] is False
    assert "parallel_tool_calls" not in captured_kwargs[1]


@pytest.mark.asyncio
async def test_tool_guard_retries_truncated_tool_arguments_with_larger_budget(monkeypatch):
    captured_kwargs: list[dict] = []

    truncated_call = SimpleNamespace(
        id="call_write",
        function=SimpleNamespace(
            name="write_file",
            arguments='{"path":"skills/mbti-test/data/questions.json"',
        ),
    )
    complete_call = SimpleNamespace(
        id="call_write",
        function=SimpleNamespace(
            name="write_file",
            arguments='{"path":"skills/mbti-test/data/questions.json","content":"[]"}',
        ),
    )

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            if len(captured_kwargs) == 1:
                return SimpleNamespace(
                    content="",
                    tool_calls=[truncated_call],
                    finish_reason="length",
                    native_finish_reason="length",
                    metadata={},
                )
            return SimpleNamespace(
                content="",
                tool_calls=[complete_call],
                finish_reason="tool_calls",
                native_finish_reason="tool_calls",
                metadata={},
            )

    monkeypatch.setenv("SPOON_BOT_TOOL_CALL_MAX_TOKENS", "12000")
    monkeypatch.setenv("SPOON_BOT_TOOL_CALL_RETRY_MAX_TOKENS", "24000")

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = "openrouter"
    loop.model = "anthropic/claude-sonnet-4.6"
    loop.base_url = "https://openrouter.ai/api/v1"
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    response = await loop._agent.llm.ask_tool(messages=[], tools=[])

    assert len(captured_kwargs) == 2
    assert captured_kwargs[0]["max_tokens"] == 12000
    assert captured_kwargs[1]["max_tokens"] == 24000
    assert response.tool_calls == [complete_call]


@pytest.mark.asyncio
async def test_tool_guard_blocks_still_truncated_tool_arguments_after_retry(monkeypatch):
    captured_kwargs: list[dict] = []

    truncated_call = SimpleNamespace(
        id="call_write",
        function=SimpleNamespace(
            name="write_file",
            arguments='{"path":"skills/mbti-test/data/questions.json"',
        ),
    )

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return SimpleNamespace(
                content="",
                tool_calls=[truncated_call],
                finish_reason="length",
                native_finish_reason="length",
                metadata={},
            )

    monkeypatch.setenv("SPOON_BOT_TOOL_CALL_MAX_TOKENS", "12000")
    monkeypatch.setenv("SPOON_BOT_TOOL_CALL_RETRY_MAX_TOKENS", "24000")

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = "openrouter"
    loop.model = "anthropic/claude-sonnet-4.6"
    loop.base_url = "https://openrouter.ai/api/v1"
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    response = await loop._agent.llm.ask_tool(messages=[], tools=[])

    assert len(captured_kwargs) == 2
    assert response.tool_calls == []
    assert "Tool call generation was truncated" in response.content
    assert response.metadata["incomplete_tool_calls_blocked"] is True


@pytest.mark.asyncio
async def test_tool_guard_leaves_non_strict_provider_kwargs_unchanged():
    captured_kwargs: list[dict] = []

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return SimpleNamespace(tool_calls=[], metadata={})

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = "anthropic"
    loop.model = "claude-sonnet-4.6"
    loop.base_url = "https://api.anthropic.com"
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    await loop._agent.llm.ask_tool(messages=[], tools=[])

    assert "parallel_tool_calls" not in captured_kwargs[0]


@pytest.mark.asyncio
async def test_tool_guard_can_force_serial_calls_for_recovery_on_native_provider():
    captured_kwargs: list[dict] = []
    tool_a = SimpleNamespace(id="call_a")
    tool_b = SimpleNamespace(id="call_b")

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return SimpleNamespace(tool_calls=[tool_a, tool_b], metadata={})

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = "anthropic"
    loop.model = "claude-sonnet-4.6"
    loop.base_url = "https://api.anthropic.com"
    loop._force_serial_tool_calls = True
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    response = await loop._agent.llm.ask_tool(messages=[], tools=[])

    assert captured_kwargs[0]["parallel_tool_calls"] is False
    assert response.tool_calls == [tool_a]
    assert response.metadata["serial_tool_calls_enforced"] is True


@pytest.mark.asyncio
async def test_openrouter_tool_guard_textualizes_completed_tool_history():
    captured_messages = []
    original_messages = _completed_tool_history()

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_messages.extend(kwargs["messages"])
            return SimpleNamespace(tool_calls=[], metadata={})

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = "openrouter"
    loop.model = "anthropic/claude-sonnet-4.6"
    loop.base_url = "https://openrouter.ai/api/v1"
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    await loop._agent.llm.ask_tool(messages=original_messages, tools=[])

    assert [msg.role for msg in captured_messages] == ["user", "assistant", "assistant"]
    assert all(getattr(msg, "tool_calls", None) is None for msg in captured_messages)
    assert "skill header" in captured_messages[1].content
    assert "EXISTS" in captured_messages[2].content
    assert original_messages[1]["tool_calls"][0]["id"] == "call_read"


@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [
        ("openai", "gpt-5.2", "https://api.openai.com/v1"),
        ("deepseek", "deepseek-chat", "https://api.deepseek.com/v1"),
        ("custom", "gpt-4.1", "https://llm-gateway.example/v1"),
    ],
)
@pytest.mark.asyncio
async def test_strict_openai_compatible_providers_textualize_completed_tool_history(
    provider: str,
    model: str,
    base_url: str,
):
    captured_messages = []

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_messages.extend(kwargs["messages"])
            return SimpleNamespace(tool_calls=[], metadata={})

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = provider
    loop.model = model
    loop.base_url = base_url
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    await loop._agent.llm.ask_tool(messages=_completed_tool_history(), tools=[])

    assert [msg.role for msg in captured_messages] == ["user", "assistant", "assistant"]
    assert all(getattr(msg, "tool_calls", None) is None for msg in captured_messages)
    assert "skill header" in captured_messages[1].content
    assert "EXISTS" in captured_messages[2].content


@pytest.mark.asyncio
async def test_gemini_textualizes_history_without_openai_parallel_flag():
    captured_kwargs: list[dict] = []

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return SimpleNamespace(tool_calls=[], metadata={})

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = "gemini"
    loop.model = "gemini-2.5-pro"
    loop.base_url = "https://generativelanguage.googleapis.com"
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    await loop._agent.llm.ask_tool(messages=_completed_tool_history(), tools=[])

    captured_messages = captured_kwargs[0]["messages"]
    assert "parallel_tool_calls" not in captured_kwargs[0]
    assert [msg.role for msg in captured_messages] == ["user", "assistant", "assistant"]
    assert all(getattr(msg, "tool_calls", None) is None for msg in captured_messages)


@pytest.mark.asyncio
async def test_anthropic_native_provider_keeps_tool_history_by_default():
    captured_messages = []
    original_messages = _completed_tool_history()

    class FakeLLM:
        async def ask_tool(self, *args, **kwargs):
            captured_messages.append(kwargs["messages"])
            return SimpleNamespace(tool_calls=[], metadata={})

    loop = AgentLoop.__new__(AgentLoop)
    loop.provider = "anthropic"
    loop.model = "claude-sonnet-4.6"
    loop.base_url = "https://api.anthropic.com/v1"
    loop._agent = SimpleNamespace(llm=FakeLLM())

    loop._install_tool_call_protocol_guards()
    await loop._agent.llm.ask_tool(messages=original_messages, tools=[])

    assert captured_messages[0] is original_messages
