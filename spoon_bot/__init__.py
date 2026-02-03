"""
Spoon-Bot: Local-first AI agent with native OS tools.

A nanobot-style local agent product focused on OS-level interactions,
powered by spoon-core as the underlying engine.
"""

__version__ = "0.1.0"
__author__ = "XSpoon Team"

from spoon_bot.agent.loop import AgentLoop
from spoon_bot.agent.tools.base import Tool

__all__ = ["AgentLoop", "Tool", "__version__"]
