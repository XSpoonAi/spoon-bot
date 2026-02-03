"""Skills system for spoon-bot.

This module provides skill management for spoon-bot, with optional integration
with spoon-core's more feature-rich SkillManager.

Usage:
    # Use the wrapper that prefers spoon-core when available
    from spoon_bot.skills import SpoonCoreSkillManager

    manager = SpoonCoreSkillManager(
        skill_paths=["/path/to/skills"],
        llm=llm_instance,  # Optional: enables LLM intent matching
        scripts_enabled=True,
    )

    # Or use the local implementation directly
    from spoon_bot.skills import SkillManager

    manager = SkillManager(skill_paths=[...])
"""

from spoon_bot.skills.models import Skill, SkillMetadata, SkillTrigger, SkillState, TriggerType
from spoon_bot.skills.loader import SkillLoader
from spoon_bot.skills.manager import SkillManager
from spoon_bot.skills.spoon_core_skills import SpoonCoreSkillManager, is_spoon_core_available

__all__ = [
    # Models
    "Skill",
    "SkillMetadata",
    "SkillTrigger",
    "SkillState",
    "TriggerType",
    # Loaders and managers
    "SkillLoader",
    "SkillManager",
    # spoon-core integration
    "SpoonCoreSkillManager",
    "is_spoon_core_available",
]
