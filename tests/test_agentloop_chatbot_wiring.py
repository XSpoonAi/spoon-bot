from unittest.mock import AsyncMock, patch

import pytest

from spoon_bot.agent.loop import AgentLoop


@pytest.mark.asyncio
async def test_agentloop_forwards_provider_api_key_base_url_to_chatbot(tmp_path):
    with patch("spoon_bot.agent.loop.ChatBot") as mock_chatbot_cls, \
         patch("spoon_bot.agent.loop.SpoonReactMCP") as mock_react_cls:

        mock_agent = mock_react_cls.return_value
        mock_agent.initialize = AsyncMock()

        loop = AgentLoop(
            workspace=tmp_path,
            provider="gemini",
            model="gemini-3-flash-preview",
            api_key="gem-key-123",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            enable_skills=False,
            auto_commit=False,
        )
        await loop.initialize()

        mock_chatbot_cls.assert_called_once()
        kwargs = mock_chatbot_cls.call_args.kwargs

        assert kwargs["llm_provider"] == "gemini"
        assert kwargs["model_name"] == "gemini-3-flash-preview"
        assert kwargs["api_key"] == "gem-key-123"
        assert kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta"
