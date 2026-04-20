from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from spoon_bot.core import SpoonBot, SpoonBotConfig, create_agent


@pytest.mark.asyncio
async def test_spoonbot_chat_uses_configured_reasoning_effort():
    bot = SpoonBot(
        SpoonBotConfig(
            model="gpt-5.4",
            provider="openai",
            reasoning_effort="high",
        )
    )
    bot._initialized = True
    bot._agent = SimpleNamespace(
        run=AsyncMock(return_value=SimpleNamespace(content="ok"))
    )

    response = await bot.chat("hi")

    assert response == "ok"
    assert bot._agent.run.await_args.kwargs["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_core_create_agent_loads_reasoning_effort_from_config():
    with patch(
        "spoon_bot.channels.config.load_agent_config",
        return_value={
            "model": "gpt-5.4",
            "provider": "openai",
            "reasoning_effort": "high",
        },
    ), patch.object(SpoonBot, "initialize", new=AsyncMock(return_value=None)):
        bot = await create_agent()

    assert bot.config.reasoning_effort == "high"
