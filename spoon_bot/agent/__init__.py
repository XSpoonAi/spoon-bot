"""Agent module: core loop and context management.

The AgentLoop class requires spoon-core SDK. Use the module-level import
to access it only when spoon-core is available.
"""

from spoon_bot.agent.context import ContextBuilder

# AgentLoop requires spoon-core SDK - conditionally export
try:
    from spoon_bot.agent.loop import AgentLoop, create_agent
    __all__ = ["AgentLoop", "ContextBuilder", "create_agent"]
except ImportError:
    AgentLoop = None
    create_agent = None
    __all__ = ["ContextBuilder"]
