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
    """

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
            "Extended thinking level for Claude models: "
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
    model_name: Optional[str] = Field(
        default=None,
        description="Effective model used by this sub-agent",
    )
