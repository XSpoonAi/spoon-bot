"""Data models for the skills system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SkillState(Enum):
    """Skill activation state."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"


class TriggerType(Enum):
    """Types of skill triggers."""
    KEYWORD = "keyword"
    PATTERN = "pattern"
    INTENT = "intent"


@dataclass
class SkillTrigger:
    """Defines when a skill should be activated."""
    type: TriggerType
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    intent_category: str | None = None
    priority: int = 50


@dataclass
class SkillMetadata:
    """Metadata parsed from SKILL.md frontmatter."""
    name: str
    description: str
    version: str = "1.0.0"
    author: str | None = None
    tags: list[str] = field(default_factory=list)
    triggers: list[SkillTrigger] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    composable: bool = False
    composes: list[str] = field(default_factory=list)
    persist_state: bool = False


@dataclass
class Skill:
    """Complete skill definition."""
    metadata: SkillMetadata
    instructions: str
    source_path: str | None = None
    loaded_at: datetime | None = None
    state: SkillState = SkillState.INACTIVE
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

    def get_prompt_injection(self) -> str:
        """Get the prompt content to inject when skill is active."""
        parts = [f"## Active Skill: {self.name}"]
        parts.append(f"Description: {self.description}")
        if self.context:
            parts.append(f"Context: {self.context}")
        parts.append("")
        parts.append(self.instructions)
        return "\n".join(parts)
