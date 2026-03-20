"""Data models for the sub-agent system.

Uses Optional[T] throughout for Python 3.9 compatibility —
Pydantic evaluates field annotations at runtime, so str | None
syntax (Python 3.10+) is not safe even with from __future__ import annotations.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field


class SubagentState(str, Enum):
    """Lifecycle states for a sub-agent."""

    PENDING = "pending"       # Created, not yet started
    RUNNING = "running"       # AgentLoop.process() in progress
    COMPLETED = "completed"   # Finished successfully
    FAILED = "failed"         # Finished with error
    CANCELLED = "cancelled"   # Cancelled by parent or kill command


class SpawnMode(str, Enum):
    """Sub-agent spawn mode."""

    RUN = "run"          # Ephemeral (default) — archived after archive_after_minutes
    SESSION = "session"  # Persistent — own directory, never auto-archived


class CleanupMode(str, Enum):
    """How frozen_result_text is handled after result is announced to spawner."""

    KEEP = "keep"      # Retain frozen_result_text after announce (default)
    DELETE = "delete"  # Clear frozen_result_text after announce to free memory


class RoutingMode(str, Enum):
    """How a persistent specialist should be used when auto-routed."""

    DIRECT = "direct"
    ORCHESTRATED = "orchestrated"


class TokenUsage(BaseModel):
    """Token usage statistics for a sub-agent run."""

    input_tokens: int = Field(default=0, description="Input/prompt tokens consumed")
    output_tokens: int = Field(default=0, description="Output/completion tokens produced")
    total_tokens: int = Field(default=0, description="Total tokens (input + output)")
    cache_read_tokens: int = Field(default=0, description="Tokens read from cache")
    cache_write_tokens: int = Field(default=0, description="Tokens written to cache")


class SubagentConfig(BaseModel):
    """Configuration for a spawned sub-agent.

    All fields are optional — unset fields inherit from the parent agent.

    When ``role`` is set, the SubagentTool will look up the matching
    AgentRole in the catalog and automatically apply its ``system_prompt``,
    ``tool_profile``, and ``max_iterations`` — mirroring opencode's
    ``subagent_type`` parameter in task.ts.
    """

    role: Optional[str] = Field(
        default=None,
        description=(
            "Specialized agent role from the catalog "
            "(e.g. 'planner', 'backend', 'frontend', 'researcher', 'reviewer'). "
            "When set, system_prompt, tool_profile, and max_iterations are "
            "automatically loaded from the role definition unless explicitly overridden."
        ),
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model to use (inherits from parent if None)",
    )
    provider: Optional[str] = Field(
        default=None,
        description="LLM provider (inherits from parent if None)",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (inherits from parent if None)",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL (inherits from parent if None)",
    )
    max_iterations: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum tool call iterations",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Custom system prompt (auto-generated if None)",
    )
    tool_profile: str = Field(
        default="core",
        description="Tool profile: core, coding, research, or full",
    )
    enabled_tools: Optional[Set[str]] = Field(
        default=None,
        description="Explicit set of tool names to enable (None = profile default)",
    )
    enable_skills: bool = Field(
        default=False,
        description="Whether to enable the skill system (off by default for speed)",
    )
    context_window: Optional[int] = Field(
        default=None,
        description="Override context window in tokens",
    )
    thinking_level: Optional[str] = Field(
        default=None,
        description=(
            "Extended thinking level for LLM models: "
            "'basic' or 'extended'. None = disabled."
        ),
    )
    timeout_seconds: Optional[int] = Field(
        default=None,
        ge=10,
        le=3600,
        description=(
            "Hard wall-clock timeout in seconds for the sub-agent run. "
            "None = no timeout (bounded only by max_iterations)."
        ),
    )
    spawn_mode: SpawnMode = Field(
        default=SpawnMode.RUN,
        description=(
            "Spawn mode: 'run' (ephemeral, archived after archive_after_minutes) "
            "or 'session' (persistent named agent, never auto-archived)."
        ),
    )
    cleanup: CleanupMode = Field(
        default=CleanupMode.KEEP,
        description=(
            "Cleanup mode for frozen_result_text after announce: "
            "'keep' (retain) or 'delete' (clear to free memory)."
        ),
    )
    agent_name: Optional[str] = Field(
        default=None,
        description=(
            "Name for a persistent session-mode agent. "
            "Required when spawn_mode='session'. Used as directory name under agents/."
        ),
    )
    specialization: Optional[str] = Field(
        default=None,
        description=(
            "Short description of the specialist sub-agent's responsibility "
            "(e.g. 'handle authentication and account recovery tasks')."
        ),
    )
    auto_route: bool = Field(
        default=False,
        description=(
            "Whether top-level user requests that strongly match this specialist "
            "should be routed to it automatically."
        ),
    )
    match_keywords: List[str] = Field(
        default_factory=list,
        description=(
            "Explicit keywords or phrases that should strongly match this "
            "specialist for auto-routing."
        ),
    )
    match_examples: List[str] = Field(
        default_factory=list,
        description=(
            "Optional example tasks that describe the kind of work this "
            "specialist should receive."
        ),
    )
    routing_mode: RoutingMode = Field(
        default=RoutingMode.DIRECT,
        description=(
            "Routing strategy for auto-routed specialist work. "
            "'direct' means route the whole request to this specialist."
        ),
    )


class PersistentSubagentProfile(BaseModel):
    """Persistent definition of a user-created subagent.

    This is the durable routing object. Session/runs are ephemeral executions
    of this profile, not the source of truth for its existence.
    """

    name: str = Field(description="Unique persistent subagent name")
    role: Optional[str] = Field(default=None, description="Optional built-in role hint")
    model: Optional[str] = Field(default=None, description="Preferred model override")
    provider: Optional[str] = Field(default=None, description="Preferred provider override")
    api_key: Optional[str] = Field(default=None, description="Optional API key override")
    base_url: Optional[str] = Field(default=None, description="Optional base URL override")
    max_iterations: int = Field(default=15, ge=1, le=50)
    system_prompt: Optional[str] = Field(default=None)
    tool_profile: str = Field(default="core")
    enabled_tools: Optional[Set[str]] = Field(default=None)
    enable_skills: bool = Field(default=False)
    context_window: Optional[int] = Field(default=None)
    thinking_level: Optional[str] = Field(default=None)
    timeout_seconds: Optional[int] = Field(default=None, ge=10, le=3600)
    cleanup: CleanupMode = Field(default=CleanupMode.KEEP)
    specialization: Optional[str] = Field(default=None)
    auto_route: bool = Field(default=False)
    match_keywords: List[str] = Field(default_factory=list)
    match_examples: List[str] = Field(default_factory=list)
    routing_mode: RoutingMode = Field(default=RoutingMode.DIRECT)
    created_at: float = Field(default_factory=time.time)
    last_active_at: Optional[float] = Field(default=None)
    last_run_agent_id: Optional[str] = Field(default=None)
    last_run_state: Optional[str] = Field(default=None)

    def to_subagent_config(self) -> "SubagentConfig":
        """Convert this profile into a session-mode SubagentConfig."""
        return SubagentConfig(
            role=self.role,
            model=self.model,
            provider=self.provider,
            api_key=self.api_key,
            base_url=self.base_url,
            max_iterations=self.max_iterations,
            system_prompt=self.system_prompt,
            tool_profile=self.tool_profile,
            enabled_tools=self.enabled_tools,
            enable_skills=self.enable_skills,
            context_window=self.context_window,
            thinking_level=self.thinking_level,
            timeout_seconds=self.timeout_seconds,
            spawn_mode=SpawnMode.SESSION,
            cleanup=self.cleanup,
            agent_name=self.name,
            specialization=self.specialization,
            auto_route=self.auto_route,
            match_keywords=list(self.match_keywords),
            match_examples=list(self.match_examples),
            routing_mode=self.routing_mode,
        )

    @classmethod
    def from_subagent_config(
        cls,
        *,
        name: str,
        config: "SubagentConfig",
        created_at: float | None = None,
        last_active_at: float | None = None,
        last_run_agent_id: str | None = None,
        last_run_state: str | None = None,
    ) -> "PersistentSubagentProfile":
        """Create a persistent profile from a session-mode config."""
        return cls(
            name=name,
            role=config.role,
            model=config.model,
            provider=config.provider,
            api_key=config.api_key,
            base_url=config.base_url,
            max_iterations=config.max_iterations,
            system_prompt=config.system_prompt,
            tool_profile=config.tool_profile,
            enabled_tools=config.enabled_tools,
            enable_skills=config.enable_skills,
            context_window=config.context_window,
            thinking_level=config.thinking_level,
            timeout_seconds=config.timeout_seconds,
            cleanup=config.cleanup,
            specialization=config.specialization,
            auto_route=config.auto_route,
            match_keywords=list(config.match_keywords),
            match_examples=list(config.match_examples),
            routing_mode=config.routing_mode,
            created_at=created_at or time.time(),
            last_active_at=last_active_at,
            last_run_agent_id=last_run_agent_id,
            last_run_state=last_run_state,
        )


class SubagentRecord(BaseModel):
    """Tracks a single sub-agent's lifecycle."""

    agent_id: str = Field(
        default_factory=lambda: f"sub_{uuid4().hex[:10]}",
        description="Unique identifier for this sub-agent",
    )
    parent_id: Optional[str] = Field(
        default=None,
        description="Parent agent_id (None = spawned by root agent)",
    )
    depth: int = Field(
        default=0,
        description="Spawn depth (0 = root, 1 = first generation, etc.)",
    )
    label: str = Field(
        default="",
        description="Short human-readable label for this sub-agent",
    )
    task: str = Field(
        default="",
        description="The task/message sent to this sub-agent",
    )
    state: SubagentState = Field(
        default=SubagentState.PENDING,
        description="Current lifecycle state",
    )
    result: Optional[str] = Field(
        default=None,
        description="Result text when state is COMPLETED",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message when state is FAILED",
    )
    config: SubagentConfig = Field(
        default_factory=SubagentConfig,
        description="Sub-agent configuration",
    )
    session_key: str = Field(
        default="",
        description="Session key for this sub-agent (auto-generated)",
    )
    # Spawner context — used for push-based wake continuation delivery
    spawner_session_key: Optional[str] = Field(
        default=None,
        description=(
            "Session key of the entity that spawned this sub-agent "
            "(used for push-based result delivery)"
        ),
    )
    spawner_channel: Optional[str] = Field(
        default=None,
        description=(
            "Channel name (e.g. 'telegram:spoon_bot') of the spawner "
            "(used for push-based result delivery)"
        ),
    )
    spawner_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Original inbound metadata of the spawner, when available.",
    )
    spawner_reply_to: Optional[str] = Field(
        default=None,
        description="Original inbound message id that replies should target.",
    )
    # Effective model name (resolved at spawn time for display)
    model_name: Optional[str] = Field(
        default=None,
        description="Effective model name used by this sub-agent (for display)",
    )
    # Token usage
    token_usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token usage statistics for this run",
    )
    # Wake continuation — latest frozen output for re-invocation
    frozen_result_text: Optional[str] = Field(
        default=None,
        description=(
            "Latest frozen completion text. Updated each time the sub-agent "
            "completes (even if woken multiple times)."
        ),
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Unix timestamp when this record was created",
    )
    started_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp when execution started",
    )
    completed_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp when execution completed",
    )
    children: List[str] = Field(
        default_factory=list,
        description="List of child agent_ids spawned by this sub-agent",
    )
    # Spawn mode and cleanup (mirrored from config at spawn time for querying)
    spawn_mode: SpawnMode = Field(
        default=SpawnMode.RUN,
        description="Spawn mode ('run' or 'session') — mirrored from config at spawn time",
    )
    cleanup: CleanupMode = Field(
        default=CleanupMode.KEEP,
        description="Cleanup mode ('keep' or 'delete') — mirrored from config at spawn time",
    )
    agent_name: Optional[str] = Field(
        default=None,
        description="Named agent name for session-mode agents",
    )
    agent_dir: Optional[str] = Field(
        default=None,
        description="Absolute path to the persistent agent directory (session mode only)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Auto-generate session_key from agent_id if not set."""
        if not self.session_key:
            self.session_key = f"subagent-{self.agent_id}"


class SubagentResult(BaseModel):
    """Result delivered back to the parent agent via the results queue."""

    agent_id: str = Field(description="Sub-agent identifier")
    label: str = Field(description="Sub-agent label")
    state: SubagentState = Field(description="Final state")
    result: Optional[str] = Field(default=None, description="Output text if completed")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    elapsed_seconds: Optional[float] = Field(
        default=None,
        description="Wall-clock seconds from start to completion",
    )
    spawner_session_key: Optional[str] = Field(
        default=None,
        description="Session key of the spawner (for push-based routing)",
    )
    spawner_channel: Optional[str] = Field(
        default=None,
        description="Channel name of the spawner (for push-based routing)",
    )
    spawner_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Original inbound metadata of the spawner, when available.",
    )
    spawner_reply_to: Optional[str] = Field(
        default=None,
        description="Original inbound message id that replies should target.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Effective model used by this sub-agent",
    )
