"""Context builder for assembling agent prompts."""

from __future__ import annotations

import base64
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.

    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """

    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md"]
    SANDBOX_WORKSPACE_ROOT = "/workspace"

    def __init__(self, workspace: Path, *, yolo_mode: bool = False):
        """
        Initialize context builder.

        Args:
            workspace: Path to the workspace directory.
            yolo_mode: When True, the agent operates directly in the user's
                       filesystem path without sandbox isolation.
        """
        self.workspace = Path(workspace).expanduser().resolve()
        self.yolo_mode = yolo_mode
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
        import sys
        import re
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")

        # Build workspace path in two forms:
        # - display_path: native OS form for readability
        # - shell_path: POSIX form safe for bash/Git-Bash on all platforms
        raw_path = str(self.workspace)
        display_path = raw_path.replace("\\", "/")

        if sys.platform == "win32":
            # Convert C:\path or C:/path -> /c/path (Git Bash POSIX format)
            def _to_posix(p: str) -> str:
                p = p.replace("\\", "/")
                return re.sub(r'^([A-Za-z]):', lambda m: f'/{m.group(1).lower()}', p)
            shell_path = _to_posix(raw_path)
        else:
            shell_path = display_path

        yolo_banner = ""
        if self.yolo_mode:
            yolo_banner = (
                "\n**YOLO MODE ACTIVE** — You are operating directly on the "
                "user's filesystem. All file reads, writes, and shell commands "
                "execute against this real directory tree. Proceed with care.\n"
            )

        return f"""# spoon-bot

You are spoon-bot, an AI agent that completes tasks by calling tools.
Current time: {now}
Workspace: {display_path}
{yolo_banner}
## Core Behavior — ALWAYS USE TOOLS

You MUST use your tools to accomplish tasks. NEVER fabricate results, NEVER pretend to execute commands, NEVER invent output. If a tool exists for the job, call it.

**Tool-first rule**: For EVERY user request, identify which tool(s) to call and call them immediately. Do not describe what you would do — just do it.

### Tool Usage Priority
1. **shell** — For running CLI commands (`cast`, `curl`, etc). ALWAYS prefer direct commands.
2. **grep** — Search for specific values (addresses, URLs, configs) without reading full files.
3. **read_file** — Read SKILL.md (use offset+limit for large files). Avoid reading entire reference docs.
4. **skill_marketplace** — Install/update skills from GitHub URLs.
5. **web_search / web_fetch** — When you need online information.
6. **self_upgrade** — After installing/removing skills, call `self_upgrade(action='reload_skills')`.

### CRITICAL: Autonomous Execution (NON-INTERACTIVE)
You run in NON-INTERACTIVE mode. There is NO human on the other end to answer questions.
**NEVER ask the user to choose. NEVER present numbered options. NEVER say "please tell me" or "which do you prefer".**
If you encounter a decision point, ALWAYS make the choice yourself using these defaults:
- Create new wallets/keys/accounts (never import existing).
- Auto-generate passwords/secrets (save them to a file and report the path).
- Pick the first/most common/default option in any list.
- Choose the simplest/fastest approach.

**EXCEPTION — Conversational messages:** If the user sends a casual or conversational message (e.g. "hi", "hello", "thanks", "how are you"), respond naturally in plain text. Do NOT invoke tools, run scripts, or trigger skill initializations. Background readiness checks and skill auto-tasks are SKIPPED for conversational messages.

**Stop on failure:** If a tool call or script returns an error, STOP and report the failure clearly. Do NOT automatically continue with follow-up steps from the same workflow — wait for the user to explicitly ask you to retry or continue.

### Workspace Path
The shell tool already runs commands with the workspace as the current directory — do NOT prepend `cd {shell_path}` to your commands.
Installed skills are located at: {shell_path}/skills/<skill_name>/
When running skill scripts via the shell tool, always use the POSIX absolute path: {shell_path}/skills/<skill_name>/scripts/<script>
Do NOT use relative paths, Windows backslash paths, or paths from the GitHub URL.

### Rules
1. **Latest message first.** The user's most recent message always takes priority over any background skill initialization or standing auto-task instructions.
2. **Act, don't echo.** Execute tasks using tools. Do not repeat instructions or explain what you plan to do at length.
3. **Match the user's language.** Reply in the same language the user uses.
4. **Be concise.** Brief status updates during tool use, clear final answers.
5. **No hallucination.** If a tool call fails, report the error honestly. Do not make up results.
6. **Complete explicit tasks.** Never return "Task completed" without showing concrete results. If the user asks for a public key, you MUST show it."""

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
        skipped: list[str] = []
        for path in media:
            p = self._resolve_media_path(path)
            mime, _ = mimetypes.guess_type(path)
            if p is None or not p.is_file():
                skipped.append(f"{path}: missing file")
                continue
            if not mime or not mime.startswith("image/"):
                skipped.append(f"{path}: unsupported mime {mime or 'unknown'}")
                continue
            try:
                b64 = base64.b64encode(p.read_bytes()).decode()
                images.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"}
                })
            except Exception as exc:
                skipped.append(f"{path}: {exc}")

        if skipped:
            logger.warning(
                "Skipped media inputs while building multimodal content: {}",
                "; ".join(skipped),
            )

        if not images:
            return text
        return images + [{"type": "text", "text": text}]

    def _resolve_media_path(self, path: str) -> Path | None:
        """Resolve runtime media refs, including /workspace aliases and workspace-relative paths."""
        candidate = str(path or "").strip()
        if not candidate:
            return None

        sandbox_root = self.SANDBOX_WORKSPACE_ROOT.rstrip("/")
        workspace_root_str = self.workspace.as_posix().rstrip("/")
        try:
            if candidate.startswith("/"):
                normalized = Path(candidate).as_posix()
                if normalized == sandbox_root or normalized.startswith(sandbox_root + "/"):
                    relative = normalized[len(sandbox_root):].lstrip("/")
                    resolved = (self.workspace / relative).resolve(strict=True)
                elif normalized == workspace_root_str or normalized.startswith(workspace_root_str + "/"):
                    relative = normalized[len(workspace_root_str):].lstrip("/")
                    resolved = (self.workspace / relative).resolve(strict=True)
                else:
                    resolved = Path(candidate).expanduser().resolve(strict=True)
            else:
                resolved = (self.workspace / candidate).resolve(strict=True)
        except (FileNotFoundError, OSError):
            return None

        return resolved if resolved.is_file() else None

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
