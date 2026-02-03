"""Skill manager for lifecycle and activation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger

from spoon_bot.skills.models import Skill, SkillState, TriggerType
from spoon_bot.skills.loader import SkillLoader


class SkillManager:
    """
    Central manager for skill lifecycle, discovery, and activation.

    Features:
    - Multi-path skill discovery
    - Keyword and pattern-based trigger matching
    - Skill activation and deactivation
    - Context injection for active skills
    """

    def __init__(
        self,
        skill_paths: list[Path | str] | None = None,
        auto_discover: bool = True,
    ):
        """
        Initialize skill manager.

        Args:
            skill_paths: Directories to search for skills.
            auto_discover: Whether to discover skills on init.
        """
        self._loader = SkillLoader(skill_paths)
        self._skills: dict[str, Skill] = {}
        self._active_skills: dict[str, Skill] = {}

        if auto_discover:
            self._discover_and_load()

    def _discover_and_load(self) -> None:
        """Discover and load all skills."""
        for skill in self._loader.load_all():
            self._skills[skill.name] = skill

    def add_skill_path(self, path: Path | str) -> None:
        """Add a skill search path and reload."""
        self._loader.add_path(path)
        self._discover_and_load()

    def list(self) -> list[str]:
        """List all available skill names."""
        return list(self._skills.keys())

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def match_triggers(self, text: str) -> list[Skill]:
        """
        Find skills whose triggers match the given text.

        Args:
            text: User input text.

        Returns:
            List of matching skills, sorted by priority.
        """
        matches = []
        text_lower = text.lower()

        for skill in self._skills.values():
            for trigger in skill.metadata.triggers:
                matched = False

                if trigger.type == TriggerType.KEYWORD:
                    # Keyword matching
                    for keyword in trigger.keywords:
                        if keyword.lower() in text_lower:
                            matched = True
                            break

                elif trigger.type == TriggerType.PATTERN:
                    # Regex pattern matching
                    for pattern in trigger.patterns:
                        if re.search(pattern, text, re.IGNORECASE):
                            matched = True
                            break

                if matched:
                    matches.append((trigger.priority, skill))
                    break  # Only count each skill once

        # Sort by priority (highest first)
        matches.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in matches]

    async def activate(
        self,
        name: str,
        context: dict[str, Any] | None = None,
    ) -> Skill:
        """
        Activate a skill.

        Args:
            name: Skill name to activate.
            context: Optional context data.

        Returns:
            Activated skill.

        Raises:
            ValueError: If skill not found.
        """
        skill = self._skills.get(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found")

        if name in self._active_skills:
            logger.debug(f"Skill '{name}' already active")
            return self._active_skills[name]

        # Check prerequisites
        for prereq in skill.metadata.prerequisites:
            if prereq not in self._active_skills:
                prereq_skill = self._skills.get(prereq)
                if prereq_skill:
                    await self.activate(prereq)
                else:
                    logger.warning(f"Prerequisite '{prereq}' not found for skill '{name}'")

        # Activate composed skills
        for composed in skill.metadata.composes:
            if composed not in self._active_skills:
                composed_skill = self._skills.get(composed)
                if composed_skill and composed_skill.metadata.composable:
                    await self.activate(composed)

        # Update state
        skill.state = SkillState.ACTIVE
        if context:
            skill.context.update(context)

        self._active_skills[name] = skill
        logger.info(f"Activated skill: {name}")
        return skill

    async def deactivate(self, name: str) -> bool:
        """
        Deactivate a skill.

        Args:
            name: Skill name to deactivate.

        Returns:
            True if skill was deactivated.
        """
        if name not in self._active_skills:
            return False

        skill = self._active_skills[name]
        skill.state = SkillState.INACTIVE
        skill.context.clear()
        del self._active_skills[name]

        logger.info(f"Deactivated skill: {name}")
        return True

    async def deactivate_all(self) -> int:
        """
        Deactivate all skills.

        Returns:
            Number of skills deactivated.
        """
        count = len(self._active_skills)
        for skill in list(self._active_skills.values()):
            skill.state = SkillState.INACTIVE
            skill.context.clear()
        self._active_skills.clear()
        return count

    def is_active(self, name: str) -> bool:
        """Check if a skill is active."""
        return name in self._active_skills

    def get_active_skill_names(self) -> list[str]:
        """Get names of all active skills."""
        return list(self._active_skills.keys())

    def get_active_context(self) -> str:
        """
        Get combined context injection for all active skills.

        Returns:
            Combined prompt content from active skills.
        """
        if not self._active_skills:
            return ""

        parts = ["# Active Skills\n"]
        for skill in self._active_skills.values():
            parts.append(skill.get_prompt_injection())
            parts.append("")

        return "\n".join(parts)

    def get_skills_summary(self) -> str:
        """
        Get a summary of all available skills for context.

        Returns:
            Short summary suitable for system prompt.
        """
        if not self._skills:
            return ""

        lines = ["## Available Skills\n"]
        for skill in self._skills.values():
            status = "[ACTIVE]" if skill.name in self._active_skills else ""
            lines.append(f"- **{skill.name}** {status}: {skill.description}")

        return "\n".join(lines)
