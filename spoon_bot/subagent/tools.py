"""LLM-facing tool definitions for sub-agent spawning and management.

Design notes
------------
The ``SubagentTool`` is the LLM-facing interface for the sub-agent system.
It mirrors two key patterns from the reference repos:

* **opencode** ``task.ts`` / ``task.txt``:
  - The tool description dynamically lists all available agent roles
    (mirrors the ``{agents}`` placeholder in task.txt).
  - A ``role`` parameter lets the LLM pick a specialised agent type
    (mirrors ``subagent_type`` in task.ts).
  - When a role is specified, its ``system_prompt``, ``tool_profile``,
    and ``max_iterations`` are loaded from the catalog automatically.

* **openclaw** ``sessions-spawn-tool.ts`` / ``subagents-tool.ts``:
  - ``label`` and ``model`` parameters on spawn (mirrors sessions_spawn).
  - ``list_roles`` action so the LLM can query available roles at runtime.
  - ``status`` / ``cancel`` / ``steer`` management actions (mirrors subagents tool).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from loguru import logger

from spoon_bot.agent.tools.base import Tool
from spoon_bot.subagent.catalog import AGENT_CATALOG, format_roles_for_prompt, get_role
from spoon_bot.subagent.models import (
    CleanupMode,
    RoutingMode,
    SpawnMode,
    SubagentConfig,
    normalize_thinking_level,
)

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
        self._spawner_context_bound = False
        self._spawner_session_key: str | None = None
        self._spawner_channel: str | None = None
        self._spawner_metadata: dict[str, Any] = {}
        self._spawner_reply_to: str | None = None

    def set_manager(self, manager: SubagentManager) -> None:
        """Inject the SubagentManager after tool registration."""
        self._manager = manager

    def set_spawner_context(
        self,
        *,
        session_key: str | None,
        channel: str | None,
        metadata: dict[str, Any] | None = None,
        reply_to: str | None = None,
    ) -> None:
        """Bind this tool instance to the session/channel that owns it."""
        self._spawner_context_bound = True
        self._spawner_session_key = session_key
        self._spawner_channel = channel
        self._spawner_metadata = dict(metadata or {})
        self._spawner_reply_to = reply_to

    def _effective_spawner_session_key(self) -> str | None:
        if self._spawner_context_bound:
            return self._spawner_session_key
        if self._manager is None:
            return None
        return getattr(self._manager, "_current_spawner_session", None)

    def _effective_spawner_channel(self) -> str | None:
        if self._spawner_context_bound:
            return self._spawner_channel
        if self._manager is None:
            return None
        return getattr(self._manager, "_current_spawner_channel", None)

    def _effective_spawner_metadata(self) -> dict[str, Any]:
        if self._spawner_context_bound:
            return dict(self._spawner_metadata)
        if self._manager is None:
            return {}
        return dict(getattr(self._manager, "_current_spawner_metadata", {}))

    def _effective_spawner_reply_to(self) -> str | None:
        if self._spawner_context_bound:
            return self._spawner_reply_to
        if self._manager is None:
            return None
        return getattr(self._manager, "_current_spawner_reply_to", None)

    # ------------------------------------------------------------------
    # Tool ABC
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        # Dynamically inject the role catalog — mirrors opencode task.txt's {agents} block.
        roles_block = format_roles_for_prompt()
        return (
            "Spawn and manage specialised sub-agents to handle subtasks.\n\n"
            "## Orchestration (LLM decides the order autonomously)\n"
            "For complex tasks (e.g. 'build a user management system'), decompose the\n"
            "work and delegate to specialised sub-agents using the `role` parameter.\n"
            "You decide which roles to use and in what order — there is no fixed pipeline.\n\n"
            "## Available agent roles (use role=<name> when spawning)\n"
            f"{roles_block}\n\n"
            "## Actions\n"
            "  spawn      — Start a sub-agent (use `role` to pick a specialised type)\n"
            "  list_roles — Show all available roles with descriptions\n"
            "  resume     — Re-invoke a persistent session-mode agent with a new task\n"
            "  status     — Show all sub-agents, states, models, and pending descendants\n"
            "  results    — Collect any completed sub-agent results\n"
            "  wait       — Wait up to timeout seconds for a result\n"
            "  cancel     — Cascade-cancel a sub-agent and its descendants\n"
            "  kill       — Alias for cancel (cascade)\n"
            "  steer      — Redirect a running sub-agent with new instructions\n"
            "  info       — Show detailed metadata for a specific sub-agent\n\n"
            "## Orchestration example\n"
            "  spawn(action='spawn', role='planner', task='Analyse requirements for: ...')\n"
            "  spawn(action='wait', timeout=120)   # wait for planner\n"
            "  spawn(action='spawn', role='backend', task='Implement API based on plan: ...')\n"
            "  spawn(action='spawn', role='frontend', task='Build UI based on plan: ...')\n"
            "  spawn(action='wait', timeout=180)   # wait for both\n\n"
            "Nested orchestration is opt-in: set `allow_subagents=true` only for\n"
            "explicit orchestrator-style child agents.\n\n"
            "Tip: launch multiple independent sub-agents in parallel (one tool call each)\n"
            "to maximise throughput. Wait for results before spawning dependents."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        # Build the role enum dynamically from the catalog
        role_names = list(AGENT_CATALOG.keys())
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "spawn", "list_roles", "resume", "status", "results",
                        "wait", "cancel", "kill", "steer", "info",
                    ],
                    "description": (
                        "Action to perform. Use 'list_roles' to see all available "
                        "specialised agent roles before spawning."
                    ),
                },
                # ---- role parameter (mirrors opencode's subagent_type) ----
                "role": {
                    "type": "string",
                    "enum": role_names,
                    "description": (
                        "Specialised agent role to use for this sub-agent. "
                        "When set, the role's system_prompt, tool_profile, and "
                        "max_iterations are loaded automatically from the catalog. "
                        f"Available roles: {', '.join(role_names)}. "
                        "Use 'list_roles' action to see full descriptions."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Task description for the sub-agent (required for 'spawn'). "
                        "Be specific and include all context the sub-agent needs — "
                        "it starts with a fresh context unless task_id is provided."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Short (3-5 word) label to identify this sub-agent",
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "Resume a previous sub-agent session by its agent_id. "
                        "The sub-agent will continue with its previous conversation "
                        "context instead of starting fresh. "
                        "Mirrors opencode task.ts task_id parameter."
                    ),
                },
                "agent_id": {
                    "type": "string",
                    "description": (
                        "Sub-agent ID (for 'cancel', 'kill', 'wait', "
                        "'steer', 'info')"
                    ),
                },
                "run_id": {
                    "type": "string",
                    "description": (
                        "Execution run ID (for 'results' and 'wait'). "
                        "Use this to wait for the current run of a reused session "
                        "without picking up older buffered results."
                    ),
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Override LLM model for the sub-agent "
                        "(inherits from parent if not set)"
                    ),
                },
                "tool_profile": {
                    "type": "string",
                    "description": (
                        "Override tool profile for the sub-agent "
                        "(core, coding, research, full). "
                        "Ignored when 'role' is set unless explicitly provided."
                    ),
                },
                "thinking": {
                    "type": "string",
                    "description": (
                        "Thinking mode for supported models. "
                        "Values: 'off', 'basic', or 'extended'."
                    ),
                },
                "enable_skills": {
                    "type": "boolean",
                    "description": (
                        "Enable the child skill system explicitly. "
                        "Omit this field to inherit the parent setting."
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
                "spawn_mode": {
                    "type": "string",
                    "enum": ["run", "session"],
                    "description": (
                        "Spawn mode: 'run' (ephemeral, default) or "
                        "'session' (persistent named agent, never auto-archived). "
                        "Session agents retain conversation context between invocations."
                    ),
                },
                "cleanup": {
                    "type": "string",
                    "enum": ["keep", "delete"],
                    "description": (
                        "What to do with frozen_result_text after announcing to spawner: "
                        "'keep' (default) or 'delete' (clear to free memory)."
                    ),
                },
                "agent_name": {
                    "type": "string",
                    "description": (
                        "Name for a persistent session-mode agent "
                        "(required when spawn_mode='session'). "
                        "Use only letters, digits, hyphens, and underscores."
                    ),
                },
                "specialization": {
                    "type": "string",
                    "description": (
                        "Short description of what this persistent specialist should "
                        "handle (for auto-routing and user visibility)."
                    ),
                },
                "auto_route": {
                    "type": "boolean",
                    "description": (
                        "Whether matching top-level user requests should be routed "
                        "to this persistent specialist automatically."
                    ),
                },
                "match_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Keywords or phrases that strongly indicate this specialist "
                        "should handle the task."
                    ),
                },
                "match_examples": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional example tasks for this specialist's routing profile."
                    ),
                },
                "routing_mode": {
                    "type": "string",
                    "enum": ["direct", "orchestrated"],
                    "description": (
                        "How this specialist should be used when auto-routed. "
                        "'direct' routes the request straight to the specialist. "
                        "'orchestrated' lets the specialist coordinate nested workers "
                        "before returning a final answer."
                    ),
                },
                "allow_subagents": {
                    "type": "boolean",
                    "description": (
                        "Allow this child to spawn nested sub-agents of its own. "
                        "Keep false for normal workers; enable only for explicit "
                        "orchestrators."
                    ),
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
        elif action == "list_roles":
            return self._handle_list_roles()
        elif action == "resume":
            return await self._handle_resume(kwargs)
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

    @staticmethod
    def _handle_list_roles() -> str:
        """Return a formatted list of all available agent roles.

        Mirrors openclaw's subagents tool 'list' action and opencode's
        dynamic {agents} block in task.txt — gives the LLM a runtime
        view of what specialised roles are available.
        """
        lines = ["Available agent roles:\n"]
        for role in AGENT_CATALOG.values():
            lines.append(f"  {role.name}")
            lines.append(f"    Description:  {role.description}")
            lines.append(f"    Tool profile: {role.tool_profile}")
            lines.append(f"    Max steps:    {role.max_iterations}")
            lines.append("")
        lines.append(
            "Usage: spawn(action='spawn', role='<name>', task='<detailed task>')"
        )
        return "\n".join(lines)

    async def _handle_spawn(self, kwargs: dict[str, Any]) -> str:
        task = kwargs.get("task")
        if not task:
            return "Error: 'task' is required for the 'spawn' action."
        task_id = kwargs.get("task_id")

        config = SubagentConfig()

        # ----------------------------------------------------------------
        # Role-based configuration (mirrors opencode task.ts subagent_type)
        # When a role is specified, load its catalog defaults first, then
        # allow explicit kwargs to override individual fields.
        # ----------------------------------------------------------------
        role_name = kwargs.get("role")
        role_applied: str | None = None
        if role_name:
            agent_role = get_role(role_name)
            if agent_role is None:
                available = ", ".join(AGENT_CATALOG.keys())
                return (
                    f"Error: unknown role {role_name!r}. "
                    f"Available roles: {available}. "
                    f"Use spawn(action='list_roles') to see descriptions."
                )
            # Apply role defaults
            config.role = agent_role.name
            config.system_prompt = agent_role.system_prompt
            config.tool_profile = agent_role.tool_profile
            config.max_iterations = agent_role.max_iterations
            if agent_role.thinking_level:
                config.thinking_level = agent_role.thinking_level
            role_applied = agent_role.name
            logger.info(
                f"Applying role '{role_name}': "
                f"profile={agent_role.tool_profile}, "
                f"max_iter={agent_role.max_iterations}"
            )

        # Explicit kwargs override role defaults (caller can fine-tune)
        if kwargs.get("model"):
            config.model = kwargs["model"]
        # Only override tool_profile if explicitly provided AND no role set
        # (or caller explicitly wants to override the role's profile)
        if kwargs.get("tool_profile") and not role_name:
            config.tool_profile = kwargs["tool_profile"]
        elif kwargs.get("tool_profile") and role_name:
            # Explicit override even when role is set
            config.tool_profile = kwargs["tool_profile"]
        if kwargs.get("thinking") is not None:
            try:
                config.thinking_level = normalize_thinking_level(kwargs["thinking"])
            except ValueError as exc:
                return f"Error: {exc}"
        if kwargs.get("enable_skills") is not None:
            config.enable_skills = bool(kwargs["enable_skills"])
        if kwargs.get("timeout_seconds"):
            try:
                config.timeout_seconds = int(kwargs["timeout_seconds"])
            except (ValueError, TypeError):
                pass
        if kwargs.get("spawn_mode"):
            try:
                config.spawn_mode = SpawnMode(kwargs["spawn_mode"])
            except ValueError:
                return f"Error: invalid spawn_mode {kwargs['spawn_mode']!r}. Use 'run' or 'session'."
        if kwargs.get("cleanup"):
            try:
                config.cleanup = CleanupMode(kwargs["cleanup"])
            except ValueError:
                return f"Error: invalid cleanup {kwargs['cleanup']!r}. Use 'keep' or 'delete'."
        if kwargs.get("agent_name"):
            config.agent_name = kwargs["agent_name"]
        if kwargs.get("specialization"):
            config.specialization = str(kwargs["specialization"]).strip()
        if kwargs.get("auto_route") is not None:
            config.auto_route = bool(kwargs["auto_route"])
        if kwargs.get("match_keywords") is not None:
            raw_keywords = kwargs["match_keywords"]
            if isinstance(raw_keywords, list):
                config.match_keywords = [
                    str(item).strip() for item in raw_keywords if str(item).strip()
                ]
            else:
                return "Error: 'match_keywords' must be an array of strings."
        if kwargs.get("match_examples") is not None:
            raw_examples = kwargs["match_examples"]
            if isinstance(raw_examples, list):
                config.match_examples = [
                    str(item).strip() for item in raw_examples if str(item).strip()
                ]
            else:
                return "Error: 'match_examples' must be an array of strings."
        if kwargs.get("routing_mode"):
            try:
                config.routing_mode = RoutingMode(kwargs["routing_mode"])
            except ValueError:
                return (
                    f"Error: invalid routing_mode {kwargs['routing_mode']!r}. "
                    "Use 'direct' or 'orchestrated'."
                )
        if kwargs.get("allow_subagents") is not None:
            config.allow_subagents = bool(kwargs["allow_subagents"])

        has_explicit_config_override = any([
            role_name is not None,
            kwargs.get("model") is not None,
            kwargs.get("tool_profile") is not None,
            kwargs.get("thinking") is not None,
            kwargs.get("enable_skills") is not None,
            kwargs.get("timeout_seconds") is not None,
            kwargs.get("spawn_mode") is not None,
            kwargs.get("cleanup") is not None,
            kwargs.get("agent_name") is not None,
            kwargs.get("specialization") is not None,
            kwargs.get("auto_route") is not None,
            kwargs.get("match_keywords") is not None,
            kwargs.get("match_examples") is not None,
            kwargs.get("routing_mode") is not None,
            kwargs.get("allow_subagents") is not None,
        ])
        config_for_call = config if (not task_id or has_explicit_config_override) else None

        # Auto-generate label from role if not provided (mirrors openclaw label)
        label = kwargs.get("label", "")
        if not label and role_applied:
            # Derive a short label: "<role>: first 40 chars of task"
            task_snippet = task[:40].rstrip() + ("…" if len(task) > 40 else "")
            label = f"{role_applied}: {task_snippet}"

        try:
            spawn_kwargs = {
                "task": task,
                "label": label,
                "parent_id": self._parent_agent_id,
                "config": config_for_call,
                "spawner_session_key": self._effective_spawner_session_key(),
                "spawner_channel": self._effective_spawner_channel(),
                "spawner_metadata": self._effective_spawner_metadata(),
                "spawner_reply_to": self._effective_spawner_reply_to(),
            }
            if task_id:
                record = await self._manager.resume_task(
                    task_id=task_id,
                    **spawn_kwargs,
                )
                intro = "Sub-agent session resumed successfully."
            else:
                record = await self._manager.spawn(**spawn_kwargs)
                intro = "Sub-agent spawned successfully."
            model_str = f" | model: {record.model_name}" if record.model_name else ""
            thinking_str = (
                f" | thinking: {record.config.thinking_level}" if record.config.thinking_level else ""
            )
            if record.config.enable_skills is None:
                skills_str = " | skills: inherit"
            else:
                skills_str = (
                    f" | skills: {'on' if record.config.enable_skills else 'off'}"
                )
            timeout_str = (
                f" | timeout: {config.timeout_seconds}s" if config.timeout_seconds else ""
            )
            role_str = f" | role: {role_applied}" if role_applied else ""
            mode_str = (
                f" | mode: {record.spawn_mode.value}"
                + (f" [{record.agent_name}]" if record.agent_name else "")
            )
            route_str = ""
            if getattr(record.config, "auto_route", False):
                route_str = " | auto-route: on"
            nested_str = (
                " | nested: on"
                if (
                    getattr(record.config, "allow_subagents", False)
                    or getattr(record.config, "routing_mode", RoutingMode.DIRECT) == RoutingMode.ORCHESTRATED
                )
                else ""
            )
            return (
                f"{intro}\n"
                f"  ID:      {record.agent_id}\n"
                f"  Run ID:  {record.run_id}\n"
                f"  Task ID: {record.agent_id}\n"
                f"  Label:   {record.label}\n"
                f"  Depth:   {record.depth}"
                f"{role_str}{model_str}{thinking_str}{skills_str}{timeout_str}{mode_str}{route_str}{nested_str}\n\n"
                f"Use spawn(action='results') or spawn(action='wait', timeout=30) "
                f"to retrieve its output when ready. "
                f"Results will also be pushed automatically when the sub-agent completes.\n"
                f"The result will include a task_id you can pass to future spawn calls "
                f"to resume this sub-agent's session."
            )
        except ValueError as exc:
            logger.warning(f"Spawn rejected: {exc}")
            return f"Error: {exc}"

    async def _handle_resume(self, kwargs: dict[str, Any]) -> str:
        agent_name = kwargs.get("agent_name")
        task = kwargs.get("task")
        if not agent_name:
            return "Error: 'agent_name' is required for the 'resume' action."
        if not task:
            return "Error: 'task' is required for the 'resume' action."

        try:
            record = await self._manager.resume_agent(
                agent_name=agent_name,
                task=task,
                parent_id=self._parent_agent_id,
                spawner_session_key=self._effective_spawner_session_key(),
                spawner_channel=self._effective_spawner_channel(),
                spawner_metadata=self._effective_spawner_metadata(),
                spawner_reply_to=self._effective_spawner_reply_to(),
            )
            model_str = f" | model: {record.model_name}" if record.model_name else ""
            return (
                f"Session agent resumed successfully.\n"
                f"  ID:    {record.agent_id}\n"
                f"  Run ID: {record.run_id}\n"
                f"  Task ID: {record.agent_id}\n"
                f"  Name:  {agent_name}\n"
                f"  Label: {record.label}{model_str}\n\n"
                f"Session history is preserved. Use spawn(action='wait', timeout=30) "
                f"to retrieve its output."
            )
        except ValueError as exc:
            logger.warning(f"Resume rejected: {exc}")
            return f"Error: {exc}"

    def _handle_status(self) -> str:
        summary = self._manager.get_status_summary(
            parent_id=self._parent_agent_id,
            spawner_session_key=self._effective_spawner_session_key(),
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
                mode_str = (
                    f" [session:{a['agent_name']}]" if a.get("spawn_mode") == "session"
                    else ""
                )
                lines.append(
                    f"  [{a['agent_id']}/{a['run_id']}] {a['label']}: "
                    f"{a['state']}{elapsed_str}{model_str}{mode_str}{desc_str}"
                )

        if summary["recent"]:
            lines.append("\nRecent:")
            for a in summary["recent"]:
                elapsed_str = (
                    f" — {a['elapsed_seconds']}s"
                    if a.get("elapsed_seconds") is not None else ""
                )
                model_str = f" [{a['model']}]" if a.get("model") else ""
                mode_str = (
                    f" [session:{a['agent_name']}]" if a.get("spawn_mode") == "session"
                    else ""
                )
                lines.append(
                    f"  [{a['agent_id']}/{a['run_id']}] {a['label']}: "
                    f"{a['state']}{elapsed_str}{model_str}{mode_str}"
                )

        return "\n".join(lines)

    async def _handle_results(self, kwargs: dict[str, Any]) -> str:
        timeout = float(kwargs.get("timeout", 0.0))
        run_id = kwargs.get("run_id")
        results = await self._manager.collect_results(
            timeout=timeout,
            spawner_session_key=self._effective_spawner_session_key(),
            run_id=run_id,
        )
        if not results:
            return (
                "No completed results available yet. "
                "Use spawn(action='wait', timeout=30) to block until one arrives."
            )
        return self._format_results(results)

    async def _handle_wait(self, kwargs: dict[str, Any]) -> str:
        timeout = float(kwargs.get("timeout", 30.0))
        agent_id = kwargs.get("agent_id")
        run_id = kwargs.get("run_id")

        results = await self._manager.collect_results(
            timeout=timeout,
            spawner_session_key=self._effective_spawner_session_key(),
            agent_id=agent_id,
            run_id=run_id,
        )

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
            info = await self._manager.get_info(
                agent_id,
                spawner_session_key=self._effective_spawner_session_key(),
            )
            record = self._manager.registry.get(agent_id) if info is not None else None
            descendants = (
                self._manager.registry.get_descendants(agent_id)
                if record else []
            )
            success = await self._manager.cancel(
                agent_id,
                cascade=True,
                spawner_session_key=self._effective_spawner_session_key(),
            )
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
                parent_id=self._parent_agent_id,
                spawner_session_key=self._effective_spawner_session_key(),
            )
            return f"Cancellation requested for {count} sub-agent(s) (cascade)."

    async def _handle_steer(self, kwargs: dict[str, Any]) -> str:
        agent_id = kwargs.get("agent_id")
        message = kwargs.get("message", "")

        if not agent_id:
            return "Error: 'agent_id' is required for the 'steer' action."
        if not message:
            return "Error: 'message' is required for the 'steer' action."

        result = await self._manager.steer(
            agent_id,
            message,
            spawner_session_key=self._effective_spawner_session_key(),
        )
        status = result.get("status")
        msg = result.get("message", "")

        if status == "accepted":
            run_id = result.get("run_id")
            run_hint = f" New run: {run_id}." if run_id else ""
            return f"Steer accepted for sub-agent {agent_id}.{run_hint} {msg}".strip()
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

        info = await self._manager.get_info(
            agent_id,
            spawner_session_key=self._effective_spawner_session_key(),
        )
        if info is None:
            return f"Sub-agent {agent_id!r} not found."

        lines = [f"Sub-agent {agent_id} info:"]
        lines.append(f"  Run ID:       {info['run_id']}")
        lines.append(f"  Label:        {info['label']}")
        lines.append(f"  State:        {info['state']}")
        lines.append(f"  Task:         {info['task'][:100]}")
        lines.append(f"  Depth:        {info['depth']}")
        if info.get("model"):
            lines.append(f"  Model:        {info['model']}")
        lines.append(f"  Tool profile: {info['tool_profile']}")
        if info.get("specialization"):
            lines.append(f"  Specialization: {info['specialization']}")
        if info.get("auto_route"):
            lines.append("  Auto-route:    on")
        if info.get("routing_mode"):
            lines.append(f"  Routing mode:  {info['routing_mode']}")
        if info.get("allow_subagents"):
            lines.append("  Nested spawn:  on")
        if "effective_enable_skills" in info:
            inherited = " (inherited)" if info.get("enable_skills") is None else ""
            lines.append(
                f"  Skills:       {'on' if info['effective_enable_skills'] else 'off'}{inherited}"
            )
        if info.get("match_keywords"):
            lines.append(f"  Keywords:      {', '.join(info['match_keywords'])}")
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
                f"--- [{r.agent_id}/{r.run_id}] {r.label} "
                f"({r.state.value}{elapsed_str}{model_str}) ---"
            )
            content = r.result or r.error or "(no output)"
            if len(content) > _RESULT_TRUNCATE_LEN:
                content = (
                    content[:_RESULT_TRUNCATE_LEN]
                    + f"\n... [truncated — {len(content) - _RESULT_TRUNCATE_LEN} chars omitted]"
                )
            lines.append(
                f"\n{header}\ntask_id: {r.agent_id}\nrun_id: {r.run_id}\n{content}"
            )
        return "\n".join(lines)
