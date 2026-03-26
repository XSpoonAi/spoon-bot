"""Self-configuration tool for agent self-management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from spoon_bot.agent.tools.base import Tool


# ---------------------------------------------------------------------------
# Dynamic tool activation
# ---------------------------------------------------------------------------


class ActivateToolTool(Tool):
    """
    Dynamically activate registered but inactive tools at runtime.

    The AI Agent autonomously decides which tools to activate based on
    the user's request. No hardcoded topic mapping — the agent reads the
    tool descriptions from ``list`` and activates what it needs.

    Actions:
        - activate: Activate one or more tools by name.
        - list:     Show all inactive tools with descriptions.
    """

    def __init__(
        self,
        activate_fn: Callable[[str], bool],
        list_inactive_fn: Callable[[], list[dict[str, str]]],
    ):
        self._activate = activate_fn
        self._list_inactive = list_inactive_fn

    @property
    def name(self) -> str:
        return "activate_tool"

    @property
    def description(self) -> str:
        return (
            "Dynamically load inactive tools at runtime. "
            "Use action='list' to see all available inactive tools and their "
            "descriptions, then action='activate' with tool_name to load the "
            "ones you need. You can activate multiple tools by calling this "
            "tool repeatedly. Always activate the right tools BEFORE answering "
            "domain-specific questions (e.g. crypto prices, blockchain ops, "
            "security checks)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["activate", "list"],
                    "description": (
                        "'activate' (load a tool by name), "
                        "'list' (show all inactive tools)"
                    ),
                },
                "tool_name": {
                    "type": "string",
                    "description": (
                        "Tool name to activate. You can also pass a "
                        "comma-separated list to activate multiple at once, "
                        "e.g. 'get_token_price,get_24h_stats'"
                    ),
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "list")
        tool_name = kwargs.get("tool_name")

        if action == "activate":
            if not tool_name:
                return "Error: 'tool_name' is required for 'activate' action."

            # Support comma-separated multi-activate
            names = [n.strip() for n in tool_name.split(",") if n.strip()]

            activated: list[str] = []
            already_active: list[str] = []
            not_found: list[str] = []

            for tn in names:
                ok = self._activate(tn)
                if ok:
                    activated.append(tn)
                else:
                    # Cannot distinguish "already active" from "not found"
                    # so just report it generically
                    already_active.append(tn)

            parts: list[str] = []
            if activated:
                parts.append(f"Activated: {', '.join(activated)}.")
            if already_active:
                parts.append(
                    f"Already active or not found: {', '.join(already_active)}."
                )
            parts.append("You can now use the activated tools.")
            return " ".join(parts)

        if action == "list":
            inactive = self._list_inactive()
            if not inactive:
                return "All tools are already active."
            lines = ["Available tools that can be activated:\n"]
            for t in inactive:
                lines.append(f"- **{t['name']}**: {t['description']}")
            return "\n".join(lines)

        return f"Unknown action: {action}"


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
            "model": "claude-sonnet-4.6",
            "max_iterations": 50,
            "shell_timeout": 3600,
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
            # Use semantic search if available, else fall back to text search
            if hasattr(self._memory_store, 'async_search'):
                raw_results = await self._memory_store.async_search(query, top_k=10)
                if raw_results:
                    lines = ["Search results (semantic):"]
                    for r in raw_results:
                        source = r.get("source", "unknown")
                        heading = r.get("heading", "")
                        score = r.get("score", 0)
                        content_text = r.get("content", "").strip()
                        if len(content_text) > 200:
                            content_text = content_text[:200] + "..."
                        label = heading if heading else source
                        lines.append(f"- [{label} | score={score:.3f}] {content_text}")
                    return "\n".join(lines)
                return "No results found"
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
    Tool for agent runtime management: reload skills, MCP servers, and status.

    This is a lean runtime tool — marketplace actions (search, install, remove)
    are provided by the ``skill-manager`` skill instead.
    """

    def __init__(self, workspace: Path | str | None = None):
        self._workspace = Path(workspace) if workspace else (
            Path.home() / ".spoon-bot" / "workspace"
        )
        # Injected by AgentLoop after construction.
        self._agent_loop: Any = None

    def set_agent_loop(self, agent_loop: Any) -> None:
        """Inject the owning AgentLoop so reload actions can call back."""
        self._agent_loop = agent_loop

    @property
    def name(self) -> str:
        return "self_upgrade"

    @property
    def description(self) -> str:
        return (
            "Agent runtime management — reload skills/MCP and check status.\n"
            "Actions:\n"
            "- reload_skills: Re-discover skills from disk and update the running agent\n"
            "- reload_mcp: Shutdown existing MCP connections and re-initialize them\n"
            "- reload_all: Reload both skills and MCP servers\n"
            "- list_skills: List currently loaded skills\n"
            "- status: Show current runtime status (model, tools, skills, session)"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "reload_skills", "reload_mcp", "reload_all",
                        "list_skills", "status",
                    ],
                    "description": "Action to perform",
                },
            },
            "required": ["action"],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _do_reload_skills(self) -> str:
        """Call agent_loop.reload_skills() and format the result.

        When new skills are added, their SKILL.md is included so the agent
        knows how to use them immediately.  Most skills are script-based —
        the agent should use the ``shell`` tool to run commands described in
        the SKILL.md rather than looking for a separate Python tool.
        """
        if not self._agent_loop:
            return "Warning: agent loop not available for reload."
        result = await self._agent_loop.reload_skills()
        after = result.get("after", [])
        added = result.get("added", [])
        removed = result.get("removed", [])
        parts = [f"Skills reloaded. Total: {len(after)}"]
        if added:
            parts.append(f"  Added: {', '.join(added)}")
        if removed:
            parts.append(f"  Removed: {', '.join(removed)}")

        # Include SKILL.md content for newly added skills so the agent
        # knows how to use them right away.
        if added:
            ws = str(self._workspace).replace("\\", "/")
            for skill_name in added:
                skill_dir = f"{ws}/skills/{skill_name}"
                parts.append(f"  Installed at: {skill_dir}")
                skill_md = self._read_skill_md(skill_name)
                if skill_md:
                    parts.append("")
                    parts.append(f"--- {skill_name} SKILL.md ---")
                    parts.append(skill_md)
                    parts.append(f"--- end {skill_name} ---")
            parts.append("")
            parts.append(
                f"NOTE: Skill scripts are at {ws}/skills/<skill_name>/scripts/. "
                "Use ABSOLUTE paths when running scripts via `shell`, e.g.: "
                f"bash {ws}/skills/{added[0]}/scripts/<script>.sh. "
                "Use the `shell` tool to run any commands or scripts. "
                "You do NOT need a special Python tool."
            )

        return "\n".join(parts)

    def _read_skill_md(self, skill_name: str) -> str | None:
        """Read the SKILL.md for a given skill from known skill paths."""
        # Check runtime workspace first, then bundled paths
        candidates = [self._workspace / "skills" / skill_name / "SKILL.md"]
        bundled = Path(__file__).resolve().parent.parent.parent / "workspace" / "skills"
        candidates.append(bundled / skill_name / "SKILL.md")

        for path in candidates:
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8")
                except Exception:
                    pass
        return None

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "list_skills")

        if action == "reload_skills":
            if not self._agent_loop:
                return "Error: agent loop not available for reload"
            try:
                return await self._do_reload_skills()
            except Exception as e:
                logger.error(f"reload_skills failed: {e}")
                return f"Error reloading skills: {e}"

        elif action == "reload_mcp":
            if not self._agent_loop:
                return "Error: agent loop not available for reload"
            try:
                result = await self._agent_loop.reload_mcp()
                before = result.get("before", [])
                after = result.get("after", [])
                added = set(after) - set(before)
                removed = set(before) - set(after)
                parts = [f"MCP servers reloaded. Total: {len(after)}"]
                if added:
                    parts.append(f"  Added: {', '.join(added)}")
                if removed:
                    parts.append(f"  Removed: {', '.join(removed)}")
                return "\n".join(parts)
            except Exception as e:
                logger.error(f"reload_mcp failed: {e}")
                return f"Error reloading MCP: {e}"

        elif action == "reload_all":
            if not self._agent_loop:
                return "Error: agent loop not available for reload"
            try:
                result = await self._agent_loop.reload_all()
                skills_r = result.get("skills", {})
                mcp_r = result.get("mcp", {})
                return (
                    f"Full reload complete.\n"
                    f"  Skills: {len(skills_r.get('after', []))} loaded\n"
                    f"  MCP servers: {len(mcp_r.get('after', []))} connected"
                )
            except Exception as e:
                logger.error(f"reload_all failed: {e}")
                return f"Error during full reload: {e}"

        elif action == "status":
            parts = ["Agent runtime status:"]
            if self._agent_loop:
                al = self._agent_loop
                parts.append(f"  Model: {al.model}")
                parts.append(f"  Provider: {al.provider}")
                parts.append(f"  Active tools: {len(al.tools.get_active_tools())}")
                parts.append(f"  Total tools: {len(al.tools._tools)}")
                parts.append(f"  MCP tools: {len(al._mcp_tools)}")
                skills = al.skills
                parts.append(
                    f"  Skills: {len(skills)} "
                    f"({', '.join(skills) if skills else 'none'})"
                )
                parts.append(f"  Session: {al.session_key}")
            else:
                parts.append("  Agent loop not available")
            return "\n".join(parts)

        elif action == "list_skills":
            if self._agent_loop and self._agent_loop.skills:
                skills = self._agent_loop.skills
                return "Loaded skills:\n" + "\n".join(f"  - {s}" for s in skills)

            skills_dir = self._workspace / "skills"
            if not skills_dir.exists():
                return "No skills installed"

            skills = [
                p.name
                for p in skills_dir.iterdir()
                if p.is_dir() and (p / "SKILL.md").exists()
            ]
            if skills:
                return "Installed skills:\n" + "\n".join(f"  - {s}" for s in skills)
            return "No skills installed"

        return f"Unknown action: {action!r}"
