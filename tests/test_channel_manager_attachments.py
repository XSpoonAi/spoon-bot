from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from spoon_bot.bus.events import InboundMessage
from spoon_bot.channels.manager import ChannelManager


@pytest.mark.asyncio
async def test_handle_message_passes_attachments_to_agent():
    manager = ChannelManager()
    manager._agent = MagicMock()
    manager._agent.process = AsyncMock(return_value="done")

    message = InboundMessage(
        content="look at this",
        channel="feishu:testbot",
        sender_id="u1",
        sender_name="Alice",
        metadata={
            "chat_type": "group",
            "attachments": [
                {
                    "workspace_path": ".channel_media/feishu/testbot/demo.png",
                    "name": "demo.png",
                    "mime_type": "image/png",
                }
            ],
        },
    )

    response = await manager._handle_message(message)

    assert response is not None
    manager._agent.process.assert_awaited_once_with(
        message="[Alice]: look at this",
        media=None,
        session_key="default",
        attachments=message.metadata["attachments"],
    )
