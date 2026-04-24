"""Sub-agent system for spoon-bot.

Provides the ability for agents to spawn child agents that run concurrently
and report results back to the parent via a push-based announcement.

The system supports LLM-driven orchestration: the main agent autonomously
decides which specialised sub-agents to spawn and in what order, using the
AgentCatalog to discover available roles (planner, backend, frontend, etc.).

Public API
----------
SubagentManager        — Orchestration engine (one per root AgentLoop).
SubagentConfig         — Per-agent config overrides (includes ``role`` field).
SubagentRecord         — Lifecycle record for a single sub-agent.
SubagentResult         — Result delivered to the parent agent.
SubagentState          — State enum (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED).
SpawnMode              — Spawn mode enum (RUN, SESSION).
CleanupMode            — Cleanup mode enum (KEEP, DELETE).
SubagentTool           — LLM-facing Tool with role-based spawning and list_roles action.
TokenUsage             — Token usage statistics for a sub-agent run.
SubagentRunsFile       — JSON file I/O for persistence.
SubagentSweeper        — Background task that archives stale records.
AgentDirectory         — Directory manager for persistent session-mode agents.
AgentRole              — Dataclass describing a specialised sub-agent role.
AGENT_CATALOG          — Built-in role registry (planner, backend, frontend, …).
get_role               — Look up an AgentRole by name.
list_roles             — Return all registered AgentRole objects.
format_roles_for_prompt — Format roles for injection into LLM prompts.
"""

from spoon_bot.subagent.models import (
    CleanupMode,
    PersistentSubagentProfile,
    RoutingMode,
    SpawnMode,
    SubagentConfig,
    SubagentRecord,
    SubagentResult,
    SubagentState,
    TokenUsage,
)
from spoon_bot.subagent.manager import SubagentManager
from spoon_bot.subagent.registry import SubagentRegistry
from spoon_bot.subagent.tools import SubagentTool
from spoon_bot.subagent.persistence import AgentDirectory, SubagentRunsFile, SubagentSweeper
from spoon_bot.subagent.catalog import (
    AgentRole,
    AGENT_CATALOG,
    get_role,
    list_roles,
    format_roles_for_prompt,
)

__all__ = [
    # Core sub-agent system
    "AgentDirectory",
    "CleanupMode",
    "PersistentSubagentProfile",
    "RoutingMode",
    "SpawnMode",
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
    # Agent role catalog (LLM orchestration)
    "AgentRole",
    "AGENT_CATALOG",
    "get_role",
    "list_roles",
    "format_roles_for_prompt",
]
