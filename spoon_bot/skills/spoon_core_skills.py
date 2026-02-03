"""Skills integration using spoon-core's SkillManager.

This module provides a wrapper around spoon-core's SkillManager for spoon-bot.
It falls back to the local implementation if spoon-core is not available.

Usage:
    from spoon_bot.skills.spoon_core_skills import SpoonCoreSkillManager

    # Create manager with spoon-core if available
    manager = SpoonCoreSkillManager(
        skill_paths=["/path/to/skills"],
        llm=llm_instance,  # Optional: enables LLM intent matching
        scripts_enabled=True,
    )

    # List skills
    skill_names = manager.list()

    # Match triggers (keyword/pattern-based)
    matches = manager.match_triggers("analyze this data")

    # Match with LLM intent (if LLM provided)
    matches = await manager.match_intent("help me trade")

    # Activate skill
    skill = await manager.activate("research", context={"topic": "AI"})

    # Get active context for prompt injection
    context = manager.get_active_context()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# Flag to track which implementation is being used
_USING_SPOON_CORE = False


@runtime_checkable
class SkillProtocol(Protocol):
    """Protocol for skill objects to allow type checking across implementations."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def get_prompt_injection(self) -> str: ...


# Try to import spoon-core's SkillManager
try:
    from spoon_ai.skills import SkillManager as CoreSkillManager
    from spoon_ai.skills import Skill as CoreSkill
    _USING_SPOON_CORE = True
    logger.info("Using spoon-core SkillManager")
except ImportError:
    CoreSkillManager = None  # type: ignore
    CoreSkill = None  # type: ignore
    logger.info("spoon-core not available, using local SkillManager")


class SpoonCoreSkillManager:
    """
    Wrapper around spoon-core's SkillManager for spoon-bot.

    Falls back to local implementation if spoon-core is not installed.
    Provides a unified interface for skill management with support for:
    - Multi-path skill discovery
    - Keyword and pattern-based trigger matching
    - LLM-powered intent matching (when LLM provided)
    - Script execution (when using spoon-core)
    - State persistence (when using spoon-core)

    Attributes:
        using_spoon_core: Boolean indicating if spoon-core is being used.
    """

    def __init__(
        self,
        skill_paths: List[Path | str] | None = None,
        llm: Any = None,
        scripts_enabled: bool = True,
        auto_discover: bool = True,
    ):
        """
        Initialize the skill manager wrapper.

        Args:
            skill_paths: List of directories to search for skills.
            llm: LLM instance for intent-based matching. Should be compatible
                 with spoon-core's LLMManager interface. Ignored if spoon-core
                 not installed.
            scripts_enabled: Whether to enable script execution. Only applies
                            when using spoon-core.
            auto_discover: Whether to auto-discover skills on initialization.
        """
        self._skill_paths = skill_paths or []
        self._llm = llm
        self._scripts_enabled = scripts_enabled
        self._manager: Any = None

        # Convert paths to strings for spoon-core compatibility
        str_paths = [str(p) for p in self._skill_paths]

        if _USING_SPOON_CORE and CoreSkillManager is not None:
            # Use spoon-core's SkillManager
            self._manager = CoreSkillManager(
                skill_paths=str_paths if str_paths else None,
                llm=llm,
                auto_discover=auto_discover,
                scripts_enabled=scripts_enabled,
            )
            self._using_core = True
        else:
            # Fall back to local implementation
            from spoon_bot.skills.manager import SkillManager as LocalSkillManager

            # Convert to Path objects for local implementation
            path_list = [Path(p) for p in self._skill_paths] if self._skill_paths else None
            self._manager = LocalSkillManager(
                skill_paths=path_list,
                auto_discover=auto_discover,
            )
            self._using_core = False

            # Store LLM for potential future use
            self._local_llm = llm

    @property
    def using_spoon_core(self) -> bool:
        """Check if spoon-core's SkillManager is being used."""
        return self._using_core

    def list(self) -> List[str]:
        """
        List all available skill names.

        Returns:
            List of skill names.
        """
        return self._manager.list()

    def get(self, name: str) -> Optional[SkillProtocol]:
        """
        Get a skill by name.

        Args:
            name: Skill name to retrieve.

        Returns:
            Skill instance or None if not found.
        """
        return self._manager.get(name)

    def match_triggers(self, text: str) -> List[SkillProtocol]:
        """
        Find skills whose keyword/pattern triggers match the given text.

        This is a fast, synchronous method that does not use LLM.

        Args:
            text: User input text to match against.

        Returns:
            List of matching skills, sorted by priority.
        """
        return self._manager.match_triggers(text)

    async def match_intent(self, text: str) -> List[SkillProtocol]:
        """
        Find skills using LLM-powered intent analysis.

        Only available when:
        - spoon-core is installed
        - LLM was provided during initialization

        Args:
            text: User input text to analyze.

        Returns:
            List of matching skills based on intent. Returns empty list
            if LLM not available or spoon-core not installed.
        """
        if self._using_core and hasattr(self._manager, 'match_intent'):
            return await self._manager.match_intent(text)
        return []

    async def find_matching_skills(
        self,
        text: str,
        use_intent: bool = True,
    ) -> List[SkillProtocol]:
        """
        Find all matching skills using both trigger and intent matching.

        Combines keyword/pattern matching with LLM intent analysis.

        Args:
            text: User input text to match.
            use_intent: Whether to also use LLM intent matching.
                       Requires spoon-core and LLM to be available.

        Returns:
            Combined list of matching skills (deduplicated).
        """
        if self._using_core and hasattr(self._manager, 'find_matching_skills'):
            return await self._manager.find_matching_skills(text, use_intent=use_intent)

        # Fall back to trigger-only matching
        return self.match_triggers(text)

    async def activate(
        self,
        name: str,
        context: Dict[str, Any] | None = None,
    ) -> SkillProtocol:
        """
        Activate a skill by name.

        When using spoon-core, this also:
        - Checks prerequisites
        - Activates composed skills
        - Restores persisted state
        - Runs activation scripts

        Args:
            name: Skill name to activate.
            context: Optional context data for the skill.

        Returns:
            Activated skill instance.

        Raises:
            ValueError: If skill not found or prerequisites not met.
        """
        return await self._manager.activate(name, context=context)

    async def deactivate(self, name: str) -> bool:
        """
        Deactivate a skill by name.

        When using spoon-core, this also:
        - Runs deactivation scripts
        - Persists state if configured

        Args:
            name: Skill name to deactivate.

        Returns:
            True if skill was deactivated, False if not active.
        """
        return await self._manager.deactivate(name)

    async def deactivate_all(self) -> int:
        """
        Deactivate all active skills.

        Returns:
            Number of skills deactivated.
        """
        return await self._manager.deactivate_all()

    def is_active(self, name: str) -> bool:
        """
        Check if a skill is currently active.

        Args:
            name: Skill name to check.

        Returns:
            True if skill is active.
        """
        return self._manager.is_active(name)

    def get_active_skill_names(self) -> List[str]:
        """
        Get names of all currently active skills.

        Returns:
            List of active skill names.
        """
        return self._manager.get_active_skill_names()

    def get_active_context(self) -> str:
        """
        Get combined prompt injection context for all active skills.

        Returns:
            Formatted string containing instructions from all active skills,
            suitable for injection into the system prompt.
        """
        return self._manager.get_active_context()

    def get_active_tools(self) -> List[Any]:
        """
        Get all tools from active skills.

        Only available when using spoon-core. Includes both Python tools
        and script tools from active skills.

        Returns:
            List of tool instances. Returns empty list if not using spoon-core.
        """
        if self._using_core and hasattr(self._manager, 'get_active_tools'):
            return self._manager.get_active_tools()
        return []

    def add_skill_path(self, path: Path | str) -> None:
        """
        Add a directory to search for skills and reload.

        Args:
            path: Directory path to add.
        """
        self._manager.add_skill_path(str(path) if isinstance(path, Path) else path)

    # === Script execution (spoon-core only) ===

    async def execute_script(
        self,
        skill_name: str,
        script_name: str,
        input_text: Optional[str] = None,
    ) -> Any:
        """
        Execute a script from a skill.

        Only available when using spoon-core.

        Args:
            skill_name: Name of the skill containing the script.
            script_name: Name of the script to execute.
            input_text: Optional input to pass to the script.

        Returns:
            ScriptResult with execution details.

        Raises:
            ValueError: If skill or script not found.
            NotImplementedError: If not using spoon-core.
        """
        if not self._using_core:
            raise NotImplementedError(
                "Script execution requires spoon-core to be installed"
            )
        return await self._manager.execute_script(
            skill_name, script_name, input_text
        )

    def set_scripts_enabled(self, enabled: bool) -> None:
        """
        Enable or disable script execution globally.

        Only has effect when using spoon-core.

        Args:
            enabled: Whether to enable script execution.
        """
        if self._using_core and hasattr(self._manager, 'set_scripts_enabled'):
            self._manager.set_scripts_enabled(enabled)
        self._scripts_enabled = enabled

    def get_script_tools(self, skill_name: str) -> List[Any]:
        """
        Get script tools for a specific skill.

        Only available when using spoon-core.

        Args:
            skill_name: Name of the skill.

        Returns:
            List of ScriptTool instances. Returns empty list if not using spoon-core.
        """
        if self._using_core and hasattr(self._manager, 'get_script_tools'):
            return self._manager.get_script_tools(skill_name)
        return []

    # === Information methods ===

    def get_skill_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a skill.

        Args:
            name: Skill name.

        Returns:
            Dictionary with skill details or None if not found.
        """
        if self._using_core and hasattr(self._manager, 'get_skill_info'):
            return self._manager.get_skill_info(name)

        # Build info dict for local implementation
        skill = self._manager.get(name)
        if not skill:
            return None

        return {
            "name": skill.name,
            "description": skill.description,
            "version": getattr(skill.metadata, 'version', '1.0.0'),
            "is_active": self._manager.is_active(name),
            "source_path": getattr(skill, 'source_path', None),
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the skill system.

        Returns:
            Dictionary with skill system statistics.
        """
        if self._using_core and hasattr(self._manager, 'get_stats'):
            return self._manager.get_stats()

        # Build stats for local implementation
        return {
            "total_skills": len(self._manager.list()),
            "active_skills": len(self._manager.get_active_skill_names()),
            "active_skill_names": self._manager.get_active_skill_names(),
            "using_spoon_core": self._using_core,
            "scripts_enabled": self._scripts_enabled,
        }

    def get_skills_summary(self) -> str:
        """
        Get a summary of all available skills.

        Returns:
            Short summary suitable for system prompt.
        """
        if hasattr(self._manager, 'get_skills_summary'):
            return self._manager.get_skills_summary()

        # Build summary manually
        skills = self._manager.list()
        if not skills:
            return ""

        lines = ["## Available Skills\n"]
        for name in skills:
            skill = self._manager.get(name)
            if skill:
                status = "[ACTIVE]" if self._manager.is_active(name) else ""
                lines.append(f"- **{name}** {status}: {skill.description}")

        return "\n".join(lines)


def is_spoon_core_available() -> bool:
    """
    Check if spoon-core is available.

    Returns:
        True if spoon-core's SkillManager can be imported.
    """
    return _USING_SPOON_CORE
