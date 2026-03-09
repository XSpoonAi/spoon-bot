"""Bridge adapter: spoon-core BaseTool → spoon-bot Tool.

Allows skill-provided tools (loaded via SkillManager/SkillLoader)
to be registered in spoon-bot's ToolRegistry for discovery via
activate_tool.
"""

from __future__ import annotations

from typing import Any

from spoon_bot.agent.tools.base import Tool, ToolParameterSchema


class SkillToolBridge(Tool):
    """Adapts a spoon-core BaseTool instance to the spoon-bot Tool interface.

    The wrapped BaseTool is callable and has ``name``, ``description``,
    ``parameters``, and ``to_param()`` — all compatible with ToolManager.
    This bridge simply exposes them via spoon-bot's ``@property`` interface
    so the tool can live in the ToolRegistry alongside native tools.
    """

    def __init__(self, base_tool: Any) -> None:
        self._base_tool = base_tool

    @property
    def name(self) -> str:
        return self._base_tool.name

    @property
    def description(self) -> str:
        return self._base_tool.description

    @property
    def parameters(self) -> ToolParameterSchema:
        return self._base_tool.parameters

    async def execute(self, **kwargs: Any) -> str:
        result = await self._base_tool.execute(**kwargs)
        return result if isinstance(result, str) else str(result)
