"""Skill loader for SKILL.md files."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from spoon_bot.skills.models import (
    Skill,
    SkillMetadata,
    SkillTrigger,
    TriggerType,
)


class SkillLoader:
    """
    Loads skills from SKILL.md files.

    SKILL.md Format:
    ---
    name: my-skill
    description: What this skill does
    version: 1.0.0
    triggers:
      - type: keyword
        keywords: [analyze, review]
        priority: 80
    ---

    # Skill Instructions

    Markdown content here...
    """

    FRONTMATTER_PATTERN = re.compile(
        r'^---\s*\n(.*?)\n---\s*\n(.*)$',
        re.DOTALL
    )

    def __init__(self, skill_paths: list[Path | str] | None = None):
        """
        Initialize skill loader.

        Args:
            skill_paths: List of directories to search for skills.
        """
        self._paths = [Path(p) for p in (skill_paths or [])]
        self._cache: dict[str, Skill] = {}

    def add_path(self, path: Path | str) -> None:
        """Add a skill search path."""
        self._paths.append(Path(path))

    def discover(self) -> list[Path]:
        """
        Discover all SKILL.md files in configured paths.

        Returns:
            List of paths to SKILL.md files.
        """
        skill_files = []

        for base_path in self._paths:
            if not base_path.exists():
                logger.warning(f"Skill path does not exist: {base_path}")
                continue

            # Find SKILL.md files (direct or in subdirectories)
            skill_files.extend(base_path.glob("**/SKILL.md"))

        logger.debug(f"Discovered {len(skill_files)} skill files")
        return skill_files

    def parse(self, file_path: Path) -> tuple[SkillMetadata, str]:
        """
        Parse a SKILL.md file into metadata and instructions.

        Args:
            file_path: Path to SKILL.md file.

        Returns:
            Tuple of (SkillMetadata, instructions).

        Raises:
            ValueError: If file format is invalid.
        """
        content = file_path.read_text(encoding="utf-8")

        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            raise ValueError(
                f"Invalid SKILL.md format in {file_path}: "
                "missing YAML frontmatter (must start with ---)"
            )

        yaml_content = match.group(1)
        instructions = match.group(2).strip()

        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")

        metadata = self._parse_metadata(data)
        return metadata, instructions

    def _parse_metadata(self, data: dict[str, Any]) -> SkillMetadata:
        """Parse metadata dictionary into SkillMetadata."""
        # Parse triggers
        triggers = []
        for trigger_data in data.get("triggers", []):
            trigger_type = TriggerType(trigger_data.get("type", "keyword"))
            trigger = SkillTrigger(
                type=trigger_type,
                keywords=trigger_data.get("keywords", []),
                patterns=trigger_data.get("patterns", []),
                intent_category=trigger_data.get("intent_category"),
                priority=trigger_data.get("priority", 50),
            )
            triggers.append(trigger)

        return SkillMetadata(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            author=data.get("author"),
            tags=data.get("tags", []),
            triggers=triggers,
            parameters=data.get("parameters", []),
            prerequisites=data.get("prerequisites", []),
            composable=data.get("composable", False),
            composes=data.get("composes", []),
            persist_state=data.get("persist_state", False),
        )

    def load(self, file_path: Path | str) -> Skill:
        """
        Load a skill from a SKILL.md file.

        Args:
            file_path: Path to SKILL.md file.

        Returns:
            Loaded Skill instance.
        """
        file_path = Path(file_path)
        metadata, instructions = self.parse(file_path)

        skill = Skill(
            metadata=metadata,
            instructions=instructions,
            source_path=str(file_path),
            loaded_at=datetime.now(),
        )

        self._cache[metadata.name] = skill
        logger.info(f"Loaded skill: {metadata.name}")
        return skill

    def load_all(self) -> list[Skill]:
        """
        Discover and load all skills.

        Returns:
            List of loaded Skill instances.
        """
        skills = []
        for file_path in self.discover():
            try:
                skill = self.load(file_path)
                skills.append(skill)
            except Exception as e:
                logger.error(f"Failed to load skill from {file_path}: {e}")

        return skills

    def get_cached(self, name: str) -> Skill | None:
        """Get a skill from cache."""
        return self._cache.get(name)
