import asyncio

import pytest

from spoon_bot.toolkit.adapter import ToolkitToolWrapper


class _SlowAsyncTool:
    name = "slow_async"
    description = "slow tool"

    async def execute(self, **kwargs):
        await asyncio.sleep(0.2)
        return "done"


@pytest.mark.asyncio
async def test_toolkit_wrapper_enforces_timeout():
    tool = ToolkitToolWrapper(_SlowAsyncTool(), timeout_seconds=0.05)
    result = await tool.execute()
    assert "timed out" in result.lower()
    assert "0.05" in result
