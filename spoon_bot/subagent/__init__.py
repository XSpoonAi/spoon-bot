"""Sub-agent system for spoon-bot.

Provides the ability for agents to spawn child agents that run concurrently
and report results back to the parent via a push-based announcement.

Public API
----------
SubagentManager   — Orchestration engine (one per root AgentLoop).
SubagentConfig    — Per-agent config overrides.
SubagentRecord    — Lifecycle record for a single sub-agent.
SubagentResult    — Result delivered to the parent agent.
SubagentState     — State enum (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED).
SubagentTool      — LLM-facing Tool that replaces the placeholder SpawnTool.
TokenUsage        — Token usage statistics for a sub-agent run.
SubagentRunsFile  — JSON file I/O for persistence.
SubagentSweeper   — Background task that archives stale records.
"""

from spoon_bot.subagent.models import (
    SubagentConfig,
    SubagentRecord,
    SubagentResult,
    SubagentState,
    TokenUsage,
)
from spoon_bot.subagent.manager import SubagentManager
from spoon_bot.subagent.registry import SubagentRegistry
from spoon_bot.subagent.tools import SubagentTool
from spoon_bot.subagent.persistence import SubagentRunsFile, SubagentSweeper

__all__ = [
    "SubagentConfig",
    "SubagentManager",
    "SubagentRecord",
    "SubagentRegistry",
    "SubagentResult",
    "SubagentState",
    "SubagentTool",
    "TokenUsage",
    "SubagentRunsFile",
    "SubagentSweeper",
]
