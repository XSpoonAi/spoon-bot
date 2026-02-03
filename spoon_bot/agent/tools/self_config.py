"""Self-configuration tool for agent self-management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from spoon_bot.agent.tools.base import Tool


class SelfConfigTool(Tool):
    """
    Tool for agent self-configuration.

    Allows the agent to read and modify its own configuration.

    Destructive Operations:
        - set: Modifies configuration values. Critical keys (model, provider,
          max_iterations) affect agent behavior immediately.
        - reset: Removes ALL custom configuration and reverts to defaults.
          This action cannot be undone.
    """

    # Critical keys that warrant extra warnings when modified
    CRITICAL_KEYS = frozenset({"model", "provider", "max_iterations"})

    def __init__(self, config_path: Path | str | None = None):
        """
        Initialize self-config tool.

        Args:
            config_path: Path to config file.
        """
        self._config_path = Path(config_path) if config_path else (
            Path.home() / ".spoon-bot" / "config.json"
        )
        self._config_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "self_config"

    @property
    def description(self) -> str:
        return """Manage agent configuration. Actions:
- get <key>: Get a configuration value
- set <key> <value>: Set a configuration value
- list: List all configuration
- reset: Reset to defaults"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["get", "set", "list", "reset"],
                    "description": "Action to perform",
                },
                "key": {
                    "type": "string",
                    "description": "Configuration key (e.g., 'model', 'max_iterations')",
                },
                "value": {
                    "type": "string",
                    "description": "New value to set",
                },
            },
            "required": ["action"],
        }

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        if self._config_path.exists():
            try:
                return json.loads(self._config_path.read_text())
            except Exception:
                pass
        return self._get_defaults()

    def _save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to file."""
        self._config_path.write_text(json.dumps(config, indent=2))

    def _get_defaults(self) -> dict[str, Any]:
        """Get default configuration."""
        return {
            "model": "claude-sonnet-4-20250514",
            "max_iterations": 20,
            "shell_timeout": 60,
            "max_output": 10000,
            "provider": "anthropic",
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "list")
        key = kwargs.get("key")
        value = kwargs.get("value")

        config = self._load_config()

        if action == "get":
            if not key:
                return "Error: 'key' is required for 'get' action"
            if key in config:
                return f"{key} = {config[key]}"
            return f"Key '{key}' not found"

        elif action == "set":
            if not key:
                return "Error: 'key' is required for 'set' action"
            if value is None:
                return "Error: 'value' is required for 'set' action"

            # Type conversion
            old_value = config.get(key)
            if isinstance(old_value, int):
                try:
                    value = int(value)
                except ValueError:
                    return f"Error: '{key}' must be an integer"
            elif isinstance(old_value, bool):
                value = value.lower() in ("true", "yes", "1")

            config[key] = value
            self._save_config(config)
            logger.info(f"Config updated: {key} = {value}")

            # Add warning for critical configuration changes
            if key in self.CRITICAL_KEYS:
                old_display = old_value if old_value is not None else "(unset)"
                return (
                    f"Warning: Configuration change: {key} will be changed from "
                    f"{old_display} to {value}. This takes effect immediately."
                )
            return f"Updated {key} = {value}"

        elif action == "list":
            lines = ["Current configuration:"]
            for k, v in config.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        elif action == "reset":
            config = self._get_defaults()
            self._save_config(config)
            return (
                "Warning: This will reset ALL configuration to defaults. "
                "Configuration reset completed."
            )

        return f"Unknown action: {action}"


class MemoryManagementTool(Tool):
    """
    Tool for managing agent memory.

    Provides explicit control over long-term memory and notes.

    Destructive Operations:
        - forget: Permanently removes a memory entry. This action cannot be
          undone and the memory content will be lost.
    """

    def __init__(self, memory_store: Any = None):
        """
        Initialize memory tool.

        Args:
            memory_store: MemoryStore instance (optional, will use default).
        """
        self._memory_store = memory_store

    def set_memory_store(self, store: Any) -> None:
        """Set the memory store."""
        self._memory_store = store

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return """Manage agent memory. Actions:
- remember <content> [category]: Add a fact to long-term memory
- note <content>: Add a note to today's daily file
- search <query>: Search memory for matching content
- forget <content>: Remove a memory entry
- summary: Get memory summary"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["remember", "note", "search", "forget", "summary"],
                    "description": "Action to perform",
                },
                "content": {
                    "type": "string",
                    "description": "Content for remember/note/forget",
                },
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "category": {
                    "type": "string",
                    "description": "Category for remember (default: 'Facts')",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if not self._memory_store:
            return "Error: Memory store not initialized"

        action = kwargs.get("action", "summary")
        content = kwargs.get("content")
        query = kwargs.get("query")
        category = kwargs.get("category", "Facts")

        if action == "remember":
            if not content:
                return "Error: 'content' is required for 'remember' action"
            self._memory_store.add_memory(content, category)
            return f"Remembered: {content[:50]}..."

        elif action == "note":
            if not content:
                return "Error: 'content' is required for 'note' action"
            self._memory_store.add_daily_note(content)
            return f"Added note for today: {content[:50]}..."

        elif action == "search":
            if not query:
                return "Error: 'query' is required for 'search' action"
            results = self._memory_store.search(query)
            if results:
                return "Search results:\n" + "\n".join(results)
            return "No results found"

        elif action == "forget":
            if not content:
                return "Error: 'content' is required for 'forget' action"
            if self._memory_store.remove_memory(content):
                content_preview = content[:50] + "..." if len(content) > 50 else content
                return f"Warning: Memory removed permanently: {content_preview}"
            return "Memory not found"

        elif action == "summary":
            return self._memory_store.get_summary()

        return f"Unknown action: {action}"


class SelfUpgradeTool(Tool):
    """
    Tool for agent self-upgrading.

    Manages skills and checks for updates.
    """

    def __init__(self, workspace: Path | str | None = None):
        """
        Initialize self-upgrade tool.

        Args:
            workspace: Workspace directory path.
        """
        self._workspace = Path(workspace) if workspace else (
            Path.home() / ".spoon-bot" / "workspace"
        )

    @property
    def name(self) -> str:
        return "self_upgrade"

    @property
    def description(self) -> str:
        return """Manage agent capabilities and updates. Actions:
- check_update: Check for new versions
- list_skills: List installed skills
- skill_info <skill_name>: Get skill details"""

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["check_update", "list_skills", "skill_info"],
                    "description": "Action to perform",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Skill name for skill_info action",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "list_skills")
        skill_name = kwargs.get("skill_name")

        if action == "check_update":
            # In a real implementation, this would check PyPI or a registry
            return "spoon-bot v0.1.0 is the latest version"

        elif action == "list_skills":
            skills_dir = self._workspace / "skills"
            if not skills_dir.exists():
                return "No skills installed"

            skills = []
            for skill_path in skills_dir.iterdir():
                if skill_path.is_dir():
                    skill_md = skill_path / "SKILL.md"
                    if skill_md.exists():
                        skills.append(skill_path.name)

            if skills:
                return "Installed skills:\n" + "\n".join(f"  - {s}" for s in skills)
            return "No skills installed"

        elif action == "skill_info":
            if not skill_name:
                return "Error: 'skill_name' is required for 'skill_info' action"

            skill_dir = self._workspace / "skills" / skill_name
            skill_md = skill_dir / "SKILL.md"

            if not skill_md.exists():
                return f"Skill '{skill_name}' not found"

            content = skill_md.read_text(encoding="utf-8")
            # Return first 500 chars
            return f"Skill: {skill_name}\n\n{content[:500]}..."

        return f"Unknown action: {action}"
