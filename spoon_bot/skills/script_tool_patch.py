"""Patched ScriptTool that aligns tool-call schema with skill input_schema.

The upstream ``ScriptTool`` (from spoon-core) exposes a single optional
``input`` string parameter regardless of the skill's declared ``input_schema``.
This causes LLMs to omit the ``input`` key or pass natural-language text
instead of the JSON payload scripts expect via stdin.

This module provides :func:`patch_script_tool` which monkey-patches an
existing ``ScriptTool`` instance so that:

1. ``parameters`` reflects the skill's ``input_schema`` when available.
2. ``execute()`` serialises tool-call arguments to JSON before passing
   them to the script's stdin, preserving backward compatibility for
   legacy scripts that accept plain-text ``input``.
"""

from __future__ import annotations

import json
from typing import Any

from loguru import logger


def patch_script_tool(tool: Any) -> Any:
    """Patch a ScriptTool so its parameters match the skill input_schema.

    Args:
        tool: A ``ScriptTool`` instance (from spoon-core).

    Returns:
        The same tool instance, patched in-place.
    """
    # Only patch if the tool has script metadata with input_schema
    script_meta = getattr(tool, "_script", None) or getattr(tool, "script", None)
    if script_meta is None:
        return tool

    input_schema = getattr(script_meta, "input_schema", None)
    if not input_schema or not isinstance(input_schema, dict):
        return tool

    # Override the parameters property to return the real schema
    original_parameters = tool.parameters if hasattr(tool, "parameters") else {}
    tool._original_parameters = original_parameters
    tool._skill_input_schema = input_schema

    # Build OpenAI-compatible tool parameters from input_schema
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    patched_params = {
        "type": "object",
        "properties": properties,
    }
    if required:
        patched_params["required"] = required

    # Monkey-patch parameters
    type(tool).parameters = property(
        lambda self: self._skill_input_schema
        if hasattr(self, "_skill_input_schema")
        else self._original_parameters
    )

    # Wrap execute to serialise kwargs → JSON stdin
    original_execute = tool.execute

    async def _patched_execute(**kwargs: Any) -> str:
        """Convert tool-call kwargs to JSON string for script stdin."""
        # If called with structured kwargs matching the schema, serialise them
        if kwargs and "input" not in kwargs:
            json_input = json.dumps(kwargs, ensure_ascii=False)
            return await original_execute(input=json_input)

        # If 'input' is provided but isn't valid JSON, try wrapping it
        raw_input = kwargs.get("input", "")
        if raw_input:
            try:
                json.loads(raw_input)  # Already valid JSON
            except (json.JSONDecodeError, TypeError):
                # Try to build JSON from the raw text as a prompt
                if properties and "prompt" in properties:
                    json_input = json.dumps({"prompt": raw_input}, ensure_ascii=False)
                    return await original_execute(input=json_input)

        return await original_execute(**kwargs)

    tool.execute = _patched_execute
    logger.debug(f"Patched ScriptTool '{getattr(tool, 'name', '?')}' with skill input_schema")
    return tool


def patch_all_script_tools(skill_manager: Any) -> int:
    """Patch all ScriptTool instances registered in a SkillManager.

    Returns the number of tools patched.
    """
    count = 0
    try:
        from spoon_ai.skills.script_tool import ScriptTool
    except ImportError:
        return 0

    # Walk skills and their tools
    skill_names = skill_manager.list() if hasattr(skill_manager, "list") else []
    for name in skill_names:
        skill = skill_manager.get(name) if hasattr(skill_manager, "get") else None
        if skill is None:
            continue
        tools = getattr(skill, "tools", [])
        for tool in tools:
            if isinstance(tool, ScriptTool):
                patch_script_tool(tool)
                count += 1

    if count:
        logger.info(f"Patched {count} ScriptTool(s) with skill input_schema")
    return count
