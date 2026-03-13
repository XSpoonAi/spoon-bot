"""LLM-facing tool definitions for sub-agent spawning and management."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.subagent.models import SubagentConfig

if TYPE_CHECKING:
    from spoon_bot.subagent.manager import SubagentManager

# Maximum characters of a sub-agent result surfaced to the parent LLM.
_RESULT_TRUNCATE_LEN = 3000


class SubagentTool(Tool):
    """Spawn and manage sub-agents from within the LLM reasoning loop.

    Sub-agents are independent AgentLoop instances that run concurrently
    in the background. Use this tool to decompose complex tasks into
    parallel subtasks, then collect their results.

    Actions
    -------
    spawn   — Create and start a sub-agent with a specific task.
    status  — List all sub-agents with state, model, and pending descendants.
    results — Collect completed sub-agent results (optionally with timeout).
    wait    — Block until a result arrives (configurable timeout).
    cancel  — Cancel a sub-agent and all its descendants (cascade kill).
    kill    — Alias for cancel (cascade).
    steer   — Redirect a running sub-agent with a new message.
    info    — Show detailed metadata for a specific sub-agent.
    """

    def __init__(
        self,
        manager: SubagentManager | None = None,
        parent_agent_id: str | None = None,
    ) -> None:
        self._manager = manager
        self._parent_agent_id = parent_agent_id

    def set_manager(self, manager: SubagentManager) -> None:
        """Inject the SubagentManager after tool registration."""
        self._manager = manager

    # ------------------------------------------------------------------
    # Tool ABC
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return (
            "Spawn and manage sub-agents to handle subtasks concurrently.\n"
            "Use 'spawn' to start a sub-agent, 'results'/'wait' to collect "
            "output, 'status' to monitor, 'steer' to redirect a running agent, "
            "'info' for details, and 'cancel'/'kill' to stop.\n\n"
            "Actions:\n"
            "  spawn   — Start a sub-agent (supports model, tool_profile, "
            "thinking, timeout_seconds)\n"
            "  status  — Show all sub-agents, states, models, and pending "
            "descendants\n"
            "  results — Collect any completed sub-agent results\n"
            "  wait    — Wait up to timeout seconds for a result\n"
            "  cancel  — Cascade-cancel a sub-agent and its descendants\n"
            "  kill    — Alias for cancel (cascade)\n"
            "  steer   — Redirect a running sub-agent with new instructions\n"
            "  info    — Show detailed metadata for a specific sub-agent"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "spawn", "status", "results", "wait",
                        "cancel", "kill", "steer", "info",
                    ],
                    "description": "Action to perform",
                },
                "task": {
                    "type": "string",
                    "description": "Task description (required for 'spawn')",
                },
                "label": {
                    "type": "string",
                    "description": "Short label to identify this sub-agent",
                },
                "agent_id": {
                    "type": "string",
                    "description": (
                        "Sub-agent ID (for 'cancel', 'kill', 'wait', "
                        "'steer', 'info')"
                    ),
                },
                "model": {
                    "type": "string",
                    "description": "Override LLM model for the sub-agent",
                },
                "tool_profile": {
                    "type": "string",
                    "description": (
                        "Tool profile for the sub-agent "
                        "(core, coding, research, full)"
                    ),
                },
                "thinking": {
                    "type": "string",
                    "description": (
                        "Enable extended thinking for Claude models. "
                        "Values: 'basic' or 'extended'."
                    ),
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": (
                        "Hard wall-clock timeout in seconds for the sub-agent "
                        "(10-3600). None = no timeout."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": (
                        "Seconds to wait for results "
                        "(for 'results' and 'wait'; default 0 = non-blocking)"
                    ),
                },
                "message": {
                    "type": "string",
                    "description": "New instruction message (for 'steer')",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if not self._manager:
            return (
                "Error: Sub-agent system is not initialized. "
                "SubagentManager was not injected into this tool."
            )

        action = kwargs.get("action", "status")

        if action == "spawn":
            return await self._handle_spawn(kwargs)
        elif action == "status":
            return self._handle_status()
        elif action == "results":
            return await self._handle_results(kwargs)
        elif action == "wait":
            return await self._handle_wait(kwargs)
        elif action in ("cancel", "kill"):
            return await self._handle_cancel(kwargs)
        elif action == "steer":
            return await self._handle_steer(kwargs)
        elif action == "info":
            return await self._handle_info(kwargs)
        else:
            return f"Unknown action: {action!r}"

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    async def _handle_spawn(self, kwargs: dict[str, Any]) -> str:
        task = kwargs.get("task")
        if not task:
            return "Error: 'task' is required for the 'spawn' action."

        config = SubagentConfig()
        if kwargs.get("model"):
            config.model = kwargs["model"]
        if kwargs.get("tool_profile"):
            config.tool_profile = kwargs["tool_profile"]
        if kwargs.get("thinking"):
            config.thinking_level = kwargs["thinking"]
        if kwargs.get("timeout_seconds"):
            try:
                config.timeout_seconds = int(kwargs["timeout_seconds"])
            except (ValueError, TypeError):
                pass

        try:
            record = await self._manager.spawn(
                task=task,
                label=kwargs.get("label", ""),
                parent_id=self._parent_agent_id,
                config=config,
            )
            model_str = f" | model: {record.model_name}" if record.model_name else ""
            thinking_str = (
                f" | thinking: {config.thinking_level}" if config.thinking_level else ""
            )
            timeout_str = (
                f" | timeout: {config.timeout_seconds}s" if config.timeout_seconds else ""
            )
            return (
                f"Sub-agent spawned successfully.\n"
                f"  ID:      {record.agent_id}\n"
                f"  Label:   {record.label}\n"
                f"  Depth:   {record.depth}{model_str}{thinking_str}{timeout_str}\n\n"
                f"Use spawn(action='results') or spawn(action='wait', timeout=30) "
                f"to retrieve its output when ready. "
                f"Results will also be pushed automatically when the sub-agent completes."
            )
        except ValueError as exc:
            logger.warning(f"Spawn rejected: {exc}")
            return f"Error: {exc}"

    def _handle_status(self) -> str:
        summary = self._manager.get_status_summary(
            parent_id=self._parent_agent_id
        )
        if summary["total"] == 0:
            return "No sub-agents have been spawned."

        lines = [f"Sub-agents ({summary['total']} total):"]

        if summary["active"]:
            lines.append("\nActive:")
            for a in summary["active"]:
                elapsed_str = (
                    f" — {a['elapsed_seconds']}s"
                    if a.get("elapsed_seconds") is not None else ""
                )
                model_str = f" [{a['model']}]" if a.get("model") else ""
                desc_str = (
                    f" (waiting on {a['pending_descendants']} descendants)"
                    if a.get("pending_descendants", 0) > 0 else ""
                )
                lines.append(
                    f"  [{a['agent_id']}] {a['label']}: "
                    f"{a['state']}{elapsed_str}{model_str}{desc_str}"
                )

        if summary["recent"]:
            lines.append("\nRecent:")
            for a in summary["recent"]:
                elapsed_str = (
                    f" — {a['elapsed_seconds']}s"
                    if a.get("elapsed_seconds") is not None else ""
                )
                model_str = f" [{a['model']}]" if a.get("model") else ""
                lines.append(
                    f"  [{a['agent_id']}] {a['label']}: "
                    f"{a['state']}{elapsed_str}{model_str}"
                )

        return "\n".join(lines)

    async def _handle_results(self, kwargs: dict[str, Any]) -> str:
        timeout = float(kwargs.get("timeout", 0.0))
        results = await self._manager.collect_results(timeout=timeout)
        if not results:
            return (
                "No completed results available yet. "
                "Use spawn(action='wait', timeout=30) to block until one arrives."
            )
        return self._format_results(results)

    async def _handle_wait(self, kwargs: dict[str, Any]) -> str:
        timeout = float(kwargs.get("timeout", 30.0))
        agent_id = kwargs.get("agent_id")

        results = await self._manager.collect_results(timeout=timeout)
        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]

        if not results:
            return (
                f"No results received within {timeout}s. "
                f"Sub-agents may still be running — try again later."
            )
        return self._format_results(results)

    async def _handle_cancel(self, kwargs: dict[str, Any]) -> str:
        agent_id = kwargs.get("agent_id")
        if agent_id:
            # Count descendants before cancelling for the message
            record = self._manager.registry.get(agent_id)
            descendants = (
                self._manager.registry.get_descendants(agent_id)
                if record else []
            )
            success = await self._manager.cancel(agent_id, cascade=True)
            if success:
                desc_count = len(descendants)
                desc_str = f" (+ {desc_count} descendants)" if desc_count else ""
                return f"Cancellation requested for sub-agent {agent_id}{desc_str}."
            return (
                f"Could not cancel sub-agent {agent_id}: "
                f"it may already be finished or not found."
            )
        else:
            count = await self._manager.cancel_all(
                parent_id=self._parent_agent_id
            )
            return f"Cancellation requested for {count} sub-agent(s) (cascade)."

    async def _handle_steer(self, kwargs: dict[str, Any]) -> str:
        agent_id = kwargs.get("agent_id")
        message = kwargs.get("message", "")

        if not agent_id:
            return "Error: 'agent_id' is required for the 'steer' action."
        if not message:
            return "Error: 'message' is required for the 'steer' action."

        result = await self._manager.steer(agent_id, message)
        status = result.get("status")
        msg = result.get("message", "")

        if status == "accepted":
            return f"Steer accepted for sub-agent {agent_id}. {msg}"
        elif status == "rate_limited":
            return f"Rate limited: {msg}"
        elif status == "done":
            return f"Cannot steer: {msg}"
        else:
            return f"Steer failed: {msg}"

    async def _handle_info(self, kwargs: dict[str, Any]) -> str:
        agent_id = kwargs.get("agent_id")
        if not agent_id:
            return "Error: 'agent_id' is required for the 'info' action."

        info = await self._manager.get_info(agent_id)
        if info is None:
            return f"Sub-agent {agent_id!r} not found."

        lines = [f"Sub-agent {agent_id} info:"]
        lines.append(f"  Label:        {info['label']}")
        lines.append(f"  State:        {info['state']}")
        lines.append(f"  Task:         {info['task'][:100]}")
        lines.append(f"  Depth:        {info['depth']}")
        if info.get("model"):
            lines.append(f"  Model:        {info['model']}")
        lines.append(f"  Tool profile: {info['tool_profile']}")
        if info.get("thinking_level"):
            lines.append(f"  Thinking:     {info['thinking_level']}")
        if info.get("elapsed_seconds") is not None:
            lines.append(f"  Elapsed:      {info['elapsed_seconds']}s")
        if info.get("pending_descendants", 0) > 0:
            lines.append(f"  Pending descendants: {info['pending_descendants']}")
        if info.get("children"):
            lines.append(f"  Children:     {', '.join(info['children'])}")
        if info.get("token_usage"):
            tu = info["token_usage"]
            lines.append(
                f"  Tokens:       {tu.get('total_tokens', 0)} "
                f"(in {tu.get('input_tokens', 0)} / out {tu.get('output_tokens', 0)})"
            )
        if info.get("result_preview"):
            lines.append(f"  Result preview: {info['result_preview']}")
        if info.get("error"):
            lines.append(f"  Error:        {info['error']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(results: list) -> str:
        lines: list[str] = []
        for r in results:
            elapsed_str = (
                f" in {r.elapsed_seconds}s" if r.elapsed_seconds is not None else ""
            )
            model_str = f" | {r.model_name}" if getattr(r, "model_name", None) else ""
            header = (
                f"--- [{r.agent_id}] {r.label} "
                f"({r.state.value}{elapsed_str}{model_str}) ---"
            )
            content = r.result or r.error or "(no output)"
            if len(content) > _RESULT_TRUNCATE_LEN:
                content = (
                    content[:_RESULT_TRUNCATE_LEN]
                    + f"\n... [truncated — {len(content) - _RESULT_TRUNCATE_LEN} chars omitted]"
                )
            lines.append(f"\n{header}\n{content}")
        return "\n".join(lines)
