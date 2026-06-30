"""Context builder for assembling agent prompts."""

from __future__ import annotations

import base64
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


def current_datetime_context() -> dict[str, str]:
    """Return the current local and UTC time facts for prompt context."""
    local_now = datetime.now().astimezone()
    utc_now = local_now.astimezone(timezone.utc)
    offset = local_now.strftime("%z")
    offset_text = f"{offset[:3]}:{offset[3:]}" if offset else "unknown"
    return {
        "date": local_now.strftime("%Y-%m-%d"),
        "time": local_now.strftime("%H:%M:%S"),
        "weekday": local_now.strftime("%A"),
        "timezone": f"{local_now.tzname() or 'local'} (UTC{offset_text})",
        "iso": local_now.isoformat(timespec="seconds"),
        "utc_iso": utc_now.isoformat(timespec="seconds"),
    }


def format_current_datetime_context(*, bracketed: bool = False) -> str:
    """Format current date/time facts for system or per-turn prompts."""
    facts = current_datetime_context()
    if bracketed:
        return "\n".join([
            f"[CURRENT DATE]: {facts['date']}",
            f"[CURRENT TIME]: {facts['time']}",
            f"[CURRENT WEEKDAY]: {facts['weekday']}",
            f"[CURRENT TIMEZONE]: {facts['timezone']}",
            f"[CURRENT ISO TIMESTAMP]: {facts['iso']}",
            f"[CURRENT UTC TIMESTAMP]: {facts['utc_iso']}",
            (
                "[TEMPORAL GROUNDING]: Treat the current date/year above as "
                "authoritative for relative dates such as today, yesterday, "
                "this year, and current. Do not replace it with training-data "
                "or search-result years."
            ),
        ])
    return "\n".join([
        f"Current date: {facts['date']}",
        f"Current time: {facts['time']}",
        f"Current weekday: {facts['weekday']}",
        f"Current timezone: {facts['timezone']}",
        f"Current ISO timestamp: {facts['iso']}",
        f"Current UTC timestamp: {facts['utc_iso']}",
        (
            "Temporal grounding: current date/year above are authoritative for "
            "relative dates such as today, yesterday, this year, and current; "
            "do not replace them with training-data or search-result years."
        ),
    ])


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
        """Set skill context to inject into system prompt."""
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
        now_context = format_current_datetime_context()

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
{now_context}
Workspace: {display_path}
{yolo_banner}
## Core Behavior — ALWAYS USE TOOLS

You MUST use your tools to accomplish tasks. NEVER fabricate results, NEVER pretend to execute commands, NEVER invent output. If a tool exists for the job, call it.

**Tool-first rule**: For EVERY user request, identify which tool(s) to call and call them immediately. Do not describe what you would do — just do it. If the safe next step is a clarification because the request does not identify which external workflow/tool/skill to use, answer with that clarification instead of calling a side-effecting tool.

**Live-state rule**: Treat memory, recent replies, and conversation history as stale hints. When the user asks for the current state of the workspace, files, installed skills, external systems, accounts, balances, jobs, or prior tool-backed artifacts, verify with the appropriate tool before answering. Do not answer live-state questions solely from memory or an earlier assistant reply.

**Prior-action dispute rule**: If the user says you forgot, contradicts your account of what happened, asks what you just did, or references earlier tool-backed actions/results, first use `search_history` with `scope='current'` to recover exact prior user/tool facts before agreeing, denying, or correcting the record. Long-term memory is not the session transcript. External/live-state tools may describe current state, but they do not prove what happened earlier unless the matching tool call/result appears in current-session history.

**Current-request rule**: The latest user request is already in the active context. Do not use `search_history` to rediscover the current/latest request, find examples/templates, or begin a new build/coding/execution task. Use `search_history` only for explicit earlier-conversation facts, prior tool results, or compacted-session recovery.

**Explicit tool boundary**: When the newest request explicitly says to use a named tool, MCP, connector, or integration, use that named capability only if it is actually registered and available. If it is not available, report that limitation; do not satisfy the request by substituting a different tool, web/API call, shell command, or ad-hoc script unless the user explicitly allowed an alternative.

### Tool Selection
- Use the tool whose contract matches the evidence needed for the newest request.
- Use `search_history` for exact same-session user/tool facts.
- Use `shell` for documented CLI commands and direct local/runtime checks.
- Use `grep` to search for specific values without reading full files.
- Use `read_file` for SKILL.md and focused file inspection.
- Use `skill_marketplace` to install/update skills from GitHub URLs.
- Use `web_search` / `web_fetch` when online information is required.
- Use `self_upgrade(action='reload_skills')` after installing or removing skills.

### CRITICAL: Autonomous Execution (NON-INTERACTIVE)
You run in NON-INTERACTIVE mode by default.
**Do not ask the user to choose when the request, active skill contract, or tool schema selects one safe workflow.**
When the latest request contains multiple ordered goals or follow-on actions, treat those listed actions as one current-turn workflow. Do not pause between listed stages to ask for feedback or whether to proceed; continue until the requested workflow is complete or tool evidence shows a concrete blocker.
If you encounter a decision point inside a selected workflow, make the choice yourself using these defaults:
- Auto-generate passwords/secrets (save them to a file and report the path).
- Pick the first/most common/default option in any list.
- Choose the simplest/fastest approach.
- If an optional enhancement requires missing input, skip that enhancement and continue with the core task.
- If a requested action is blocked by current state (for example zero balance), report the concrete blocker and stop; do not ask for confirmation to retry the impossible action.

**Clarification boundary:** Ask one concise clarification when proceeding would choose between multiple plausible external workflows, accounts, tools, skills, networks, or paid/irreversible side effects and the latest user request does not select one or explicitly ask for all of them. Name the concrete choices if you know them. In that case, do not run the side-effecting action first.

**EXCEPTION — Conversational messages:** If the user sends a casual or conversational message (e.g. "hi", "hello", "thanks", "how are you"), respond naturally in plain text. Do NOT invoke tools, run scripts, or trigger skill initializations. Background readiness checks and skill auto-tasks are SKIPPED for conversational messages.
Short confirmations or imperative follow-ups are not casual chatter when there is a recent task, active skill, interrupted request, or other session task anchor; resolve them against that task context instead of returning a generic welcome.

**Error handling:** If a tool call returns an error, evaluate whether it is recoverable. For transient or expected errors (e.g., waiting for state changes, retryable network issues), continue the workflow. For unrecoverable errors (e.g., missing credentials, invalid configuration), report the failure clearly and stop.

**Unsupported irreversible actions:** If no available tool or installed skill can safely perform an irreversible action (for example, transferring funds, swapping assets, changing credentials, or deleting data), report the limitation and stop; do not create ad-hoc scripts to bypass a missing safe tool, do not search for or expose secrets, and do not claim the action was completed.

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
6. **Keep scratchpad private.** Do not include internal planning notes, self-instructions, or reasoning-process narration in the final user-facing answer. Start with the result or answer.
7. **Human-facing final answer.** Convert tool JSON, command transcripts, and structured outputs into concise normal prose. Do not paste raw tool payloads unless the user explicitly asks for raw logs or JSON.
8. **Complete explicit tasks.** Never return "Task completed" without showing concrete results. If the user asks for a public key, you MUST show it."""

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
