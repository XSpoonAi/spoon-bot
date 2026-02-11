"""Tests for cancellation propagation."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRestCancellation:
    """Verify that SSE client disconnect triggers cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_event_stops_streaming(self):
        """When cancel_event is set, _stream_sse should stop yielding."""
        from spoon_bot.gateway.api.v1.agent import _stream_sse

        chunks_yielded = []
        cancel_event = asyncio.Event()

        mock_agent = AsyncMock()

        async def slow_stream(**kwargs):
            for i in range(10):
                await asyncio.sleep(0.01)
                yield {"type": "content", "delta": f"chunk{i}", "metadata": {}}
            yield {"type": "done", "delta": "", "metadata": {}}

        mock_agent.stream = slow_stream

        # Set cancel after short delay
        async def set_cancel():
            await asyncio.sleep(0.03)
            cancel_event.set()

        asyncio.create_task(set_cancel())

        async for chunk in _stream_sse(
            mock_agent, "test", None, False,
            trace_id="trc_test", request_id="req_test",
            cancel_event=cancel_event,
        ):
            chunks_yielded.append(chunk)

        # Should have stopped before all 10 chunks
        content_chunks = [c for c in chunks_yielded if "content" in c and "chunk" in c]
        assert len(content_chunks) < 10
