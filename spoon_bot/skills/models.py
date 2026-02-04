"""Data models for the skills system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class SkillState(str, Enum):
    """Skill activation state."""
    INACTIVE = "inactive"
    ACTIVE = "active"
    ERROR = "error"


class TriggerType(str, Enum):
    """Types of skill triggers."""
    KEYWORD = "keyword"
    PATTERN = "pattern"
    INTENT = "intent"


class SkillTrigger(BaseModel):
    """
    Defines when a skill should be activated.

    Attributes:
        type: Type of trigger (keyword, pattern, intent).
        keywords: Keywords that activate this trigger.
        patterns: Regex patterns that activate this trigger.
        intent_category: Intent category for LLM-based matching.
        priority: Trigger priority (0-100, higher is more important).
    """

    type: TriggerType = Field(default=TriggerType.KEYWORD)
    keywords: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    intent_category: str | None = Field(default=None)
    priority: int = Field(default=50, ge=0, le=100)

    @model_validator(mode="after")
    def validate_trigger_content(self) -> "SkillTrigger":
        """Ensure at least one trigger mechanism is defined."""
        if self.type == TriggerType.KEYWORD and not self.keywords:
            raise ValueError("Keyword trigger requires at least one keyword")
        elif self.type == TriggerType.PATTERN and not self.patterns:
            raise ValueError("Pattern trigger requires at least one pattern")
        elif self.type == TriggerType.INTENT and not self.intent_category:
            raise ValueError("Intent trigger requires intent_category")
        return self

    @field_validator("patterns")
    @classmethod
    def validate_patterns(cls, v: list[str]) -> list[str]:
        """Validate that patterns are valid regex."""
        import re
        for pattern in v:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        return v


class SkillMetadata(BaseModel):
    """
    Metadata parsed from SKILL.md frontmatter.

    Attributes:
        name: Unique skill name.
        description: Skill description.
        version: Skill version (semver format).
        author: Skill author.
        tags: Skill tags for categorization.
        triggers: Skill activation triggers.
        parameters: Skill parameters.
        prerequisites: Required skills that must be active first.
        composable: Whether this skill can be composed into others.
        composes: Skills this skill composes.
        persist_state: Whether to persist skill state across sessions.
    """

    name: str = Field(..., min_length=1, description="Unique skill name")
    description: str = Field(..., min_length=1, description="Skill description")
    version: str = Field(default="1.0.0", description="Skill version")
    author: str | None = Field(default=None, description="Skill author")
    tags: list[str] = Field(default_factory=list, description="Skill tags")
    triggers: list[SkillTrigger] = Field(default_factory=list, description="Activation triggers")
    parameters: list[dict[str, Any]] = Field(default_factory=list, description="Skill parameters")
    prerequisites: list[str] = Field(default_factory=list, description="Required skills")
    composable: bool = Field(default=False, description="Can be composed")
    composes: list[str] = Field(default_factory=list, description="Composed skills")
    persist_state: bool = Field(default=False, description="Persist state")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate version format (basic semver check)."""
        import re
        # Basic semver pattern
        if not re.match(r"^\d+\.\d+\.\d+(-[\w.]+)?(\+[\w.]+)?$", v):
            raise ValueError(
                f"Invalid version format: {v}. Expected semver (e.g., 1.0.0)"
            )
        return v


class Skill(BaseModel):
    """
    Complete skill definition.

    Attributes:
        metadata: Skill metadata from frontmatter.
        instructions: Skill instructions/prompt content.
        source_path: Path to the skill file.
        loaded_at: When the skill was loaded.
        state: Current activation state.
        context: Runtime context data.
    """

    metadata: SkillMetadata
    instructions: str = Field(..., min_length=1, description="Skill instructions")
    source_path: str | None = Field(default=None, description="Source file path")
    loaded_at: datetime | None = Field(default=None, description="Load timestamp")
    state: SkillState = Field(default=SkillState.INACTIVE, description="Activation state")
    context: dict[str, Any] = Field(default_factory=dict, description="Runtime context")

    # Allow mutation of state and context
    model_config = {"frozen": False}

    @property
    def name(self) -> str:
        """Get skill name."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """Get skill description."""
        return self.metadata.description

    def get_prompt_injection(self) -> str:
        """
        Get the prompt content to inject when skill is active.

        Returns:
            Formatted prompt content for the active skill.
        """
        parts = [f"## Active Skill: {self.name}"]
        parts.append(f"Description: {self.description}")
        if self.context:
            parts.append(f"Context: {self.context}")
        parts.append("")
        parts.append(self.instructions)
        return "\n".join(parts)

    def activate(self, context: dict[str, Any] | None = None) -> None:
        """
        Activate the skill.

        Args:
            context: Optional context data to set.
        """
        self.state = SkillState.ACTIVE
        if context:
            self.context.update(context)

    def deactivate(self) -> None:
        """Deactivate the skill and clear context."""
        self.state = SkillState.INACTIVE
        self.context.clear()

    def set_error(self, error_message: str) -> None:
        """
        Mark the skill as having an error.

        Args:
            error_message: Description of the error.
        """
        self.state = SkillState.ERROR
        self.context["error"] = error_message
