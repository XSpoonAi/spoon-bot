"""Context builder for assembling agent prompts."""

from __future__ import annotations

import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.

    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]

    def __init__(self, workspace: Path):
        """
        Initialize context builder.

        Args:
            workspace: Path to the workspace directory.
        """
        self.workspace = Path(workspace).expanduser().resolve()
        self._memory_context: str = ""
        self._skills_summary: str = ""
        self._skill_context: str = ""

    def set_memory_context(self, context: str) -> None:
        """Set memory context to inject into system prompt."""
        self._memory_context = context

    def set_skills_summary(self, summary: str) -> None:
        """Set skills summary to inject into system prompt."""
        self._skills_summary = summary

    def set_skill_context(self, context: str) -> None:
        """Set active skill context to inject into system prompt."""
        self._skill_context = context

    def build_system_prompt(self) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.

        Returns:
            Complete system prompt.
        """
        parts = []

        # Core identity
        parts.append(self._get_identity())

        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        # Memory context
        if self._memory_context:
            parts.append(f"# Memory\n\n{self._memory_context}")

        # Active skill context (injected instructions from activated skills)
        if self._skill_context:
            parts.append(self._skill_context)

        # Skills summary
        if self._skills_summary:
            parts.append(f"# Installed Skills\n\n{self._skills_summary}")

        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """Get the core identity section."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        workspace_path = str(self.workspace).replace("\\", "/")

        return f"""# spoon-bot

You are spoon-bot, an AI agent that completes tasks by calling tools.
Current time: {now}
Workspace: {workspace_path}

## Core Behavior — ALWAYS USE TOOLS

You MUST use your tools to accomplish tasks. NEVER fabricate results, NEVER pretend to execute commands, NEVER invent output. If a tool exists for the job, call it.

**Tool-first rule**: For EVERY user request, identify which tool(s) to call and call them immediately. Do not describe what you would do — just do it.

### Tool Usage Priority
1. **skill_marketplace** — When the user provides a GitHub URL or asks to install a skill, call `skill_marketplace` with `action='install_skill'` immediately.
2. **web_search / web_fetch** — When you need online information, search first.
3. **shell** — For running commands, scripts, installations.
4. **read_file / write_file / edit_file** — For file operations.
5. **self_upgrade** — After installing/removing skills, call `self_upgrade(action='reload_skills')` to activate changes.

### Skill Installation Workflow
When the user asks to install a skill from a GitHub URL:
1. Call `skill_marketplace(action='install_skill', url='<the URL>')` — downloads the skill
2. Call `self_upgrade(action='reload_skills')` — activates the new skill's tools
3. Follow the SKILL.md instructions (returned in step 1's result) to complete the user's request

### CRITICAL: Autonomous Execution (NON-NEGOTIABLE)
You run in NON-INTERACTIVE one-shot mode. There is NO human on the other end to answer questions.
**NEVER ask the user to choose. NEVER present numbered options. NEVER say "please tell me" or "which do you prefer".**
If you encounter a decision point, ALWAYS make the choice yourself using these defaults:
- Create new wallets/keys/accounts (never import existing).
- Auto-generate passwords/secrets (save them to a file and report the path).
- Pick the first/most common/default option in any list.
- Choose the simplest/fastest approach.
- Continue executing ALL remaining steps until the task produces a concrete result.
You MUST keep calling tools until the user's request is fully satisfied with a concrete output (e.g., a public key, a file path, a command result). Stopping early to ask questions is a failure.

### Workspace Path
Installed skills are located at: {workspace_path}/skills/<skill_name>/
When running skill scripts, use the FULL ABSOLUTE PATH: {workspace_path}/skills/<skill_name>/scripts/<script>.
Do NOT use relative paths or paths from the GitHub URL.

### Rules
1. **Act, don't echo.** Execute tasks using tools. Do not repeat instructions or explain what you plan to do at length.
2. **Match the user's language.** Reply in the same language the user uses.
3. **Be concise.** Brief status updates during tool use, clear final answers.
4. **No hallucination.** If a tool call fails, report the error honestly. Do not make up results.
5. **Complete the task.** Never return "Task completed" without showing concrete results. If the user asks for a public key, you MUST show it."""

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []

        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding="utf-8")
                    parts.append(f"## {filename}\n\n{content}")
                except Exception:
                    pass

        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        media: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            media: Optional list of local file paths for images/media.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = self.build_system_prompt()
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(
        self,
        text: str,
        media: list[str] | None,
    ) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text

        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            try:
                b64 = base64.b64encode(p.read_bytes()).decode()
                images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })
            except Exception:
                pass

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.

        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.

        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        })
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.

        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.

        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}

        if tool_calls:
            msg["tool_calls"] = tool_calls

        messages.append(msg)
        return messages
