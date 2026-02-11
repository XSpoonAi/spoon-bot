"""
Skills module - Uses spoon-core SDK directly.

All skill functionality is provided by spoon-core's SkillManager.
No local reimplementations - use spoon-core directly.
"""

try:
    from spoon_ai.skills import SkillManager
    from spoon_ai.skills.loader import SkillLoader
    from spoon_ai.skills.registry import SkillRegistry
    from spoon_ai.skills.executor import ScriptExecutor
    from spoon_ai.skills.script_tool import ScriptTool
    from spoon_ai.agents.skill_mixin import SkillEnabledMixin

    __all__ = [
        "SkillManager",
        "SkillLoader",
        "SkillRegistry",
        "ScriptExecutor",
        "ScriptTool",
        "SkillEnabledMixin",
    ]

except ImportError:
    raise ImportError(
        "spoon-bot requires spoon-core SDK for skill functionality. "
        "Install with: pip install spoon-ai"
    )
