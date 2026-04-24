from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


SUBAGENT_SOURCE = {
    "type": "subagent",
    "is_subagent": True,
    "subagent_id": "sub_12345",
    "subagent_name": "research-subagent",
}


def _base_mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.sessions = MagicMock()
    agent.tools = MagicMock()
    agent.tools.list_tools = MagicMock(return_value=[])
    agent.skills = []
    agent.get_last_response_source = MagicMock(return_value=dict(SUBAGENT_SOURCE))
    return agent


def _create_test_app(mock_agent: MagicMock):
    from spoon_bot.gateway.app import create_app, set_agent
    from spoon_bot.gateway.config import GatewayConfig
    from spoon_bot.gateway.websocket.manager import ConnectionManager
    import spoon_bot.gateway.app as app_module

    app_module._auth_required = False
    app = create_app(GatewayConfig.from_env())
    set_agent(mock_agent)
    app_module._connection_manager = ConnectionManager()
    return app


def _collect_ws_messages(ws, request_id: str) -> list[dict]:
    messages: list[dict] = []
    while True:
        payload = ws.receive_json()
        messages.append(payload)
        if payload.get("type") == "response" and payload.get("id") == request_id:
            return messages


class TestHTTPSubagentSource:
    def test_non_streaming_chat_returns_subagent_source(self):
        mock_agent = _base_mock_agent()
        mock_agent.process = AsyncMock(return_value="hello from subagent")

        client = TestClient(_create_test_app(mock_agent))
        response = client.post("/v1/agent/chat", json={"message": "hello"})

        assert response.status_code == 200
        body = response.json()
        assert body["data"]["source"] == SUBAGENT_SOURCE

    def test_streaming_chat_returns_subagent_source_in_chunks(self):
        mock_agent = _base_mock_agent()

        async def mock_stream(**kwargs):
            yield {
                "type": "content",
                "delta": "chunk from subagent",
                "metadata": {},
                "source": dict(SUBAGENT_SOURCE),
            }
            yield {
                "type": "done",
                "delta": "",
                "metadata": {"content": "chunk from subagent"},
                "source": dict(SUBAGENT_SOURCE),
            }

        mock_agent.stream = mock_stream

        client = TestClient(_create_test_app(mock_agent))
        response = client.post(
            "/v1/agent/chat",
            json={"message": "hello", "options": {"stream": True}},
        )

        assert response.status_code == 200
        events = []
        for block in response.text.strip().split("\n\n"):
            if not block.startswith("data: "):
                continue
            payload = block[6:]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))

        content_events = [event for event in events if event.get("type") == "content"]
        assert content_events
        assert content_events[0]["source"] == SUBAGENT_SOURCE


class TestWSSubagentSource:
    def test_non_streaming_ws_returns_subagent_source(self):
        mock_agent = _base_mock_agent()
        mock_agent.process = AsyncMock(return_value="ws subagent response")

        client = TestClient(_create_test_app(mock_agent))
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json(
                {
                    "type": "request",
                    "id": "req-subagent",
                    "method": "chat.send",
                    "params": {"message": "hello"},
                }
            )
            messages = _collect_ws_messages(ws, "req-subagent")

        complete_events = [
            message
            for message in messages
            if message.get("type") == "event" and message.get("event") == "agent.complete"
        ]
        assert complete_events
        assert complete_events[-1]["data"]["source"] == SUBAGENT_SOURCE

        response = messages[-1]
        assert response["result"]["source"] == SUBAGENT_SOURCE

    def test_streaming_ws_returns_subagent_source(self):
        mock_agent = _base_mock_agent()

        async def mock_stream(**kwargs):
            yield {
                "type": "content",
                "delta": "stream chunk",
                "metadata": {},
                "source": dict(SUBAGENT_SOURCE),
            }
            yield {
                "type": "done",
                "delta": "",
                "metadata": {"content": "stream chunk"},
                "source": dict(SUBAGENT_SOURCE),
            }

        mock_agent.stream = mock_stream

        client = TestClient(_create_test_app(mock_agent))
        with client.websocket_connect("/v1/ws") as ws:
            ws.receive_json()
            ws.send_json(
                {
                    "type": "request",
                    "id": "req-stream-subagent",
                    "method": "chat.send",
                    "params": {"message": "hello", "stream": True},
                }
            )
            messages = _collect_ws_messages(ws, "req-stream-subagent")

        chunk_events = [
            message
            for message in messages
            if message.get("type") == "event" and message.get("event") == "agent.stream.chunk"
        ]
        done_events = [
            message
            for message in messages
            if message.get("type") == "event" and message.get("event") == "agent.stream.done"
        ]

        assert chunk_events
        assert chunk_events[0]["data"]["source"] == SUBAGENT_SOURCE
        assert done_events
        assert done_events[0]["data"]["source"] == SUBAGENT_SOURCE

        response = messages[-1]
        assert response["result"]["source"] == SUBAGENT_SOURCE
