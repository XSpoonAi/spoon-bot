"""Sub-agent manager — spawn, run, collect results, cleanup."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import Any, Callable, Awaitable, Optional, TYPE_CHECKING
from uuid import uuid4

from loguru import logger

from spoon_bot.agent.tools.registry import TOOL_PROFILES
from spoon_bot.subagent.models import (
    CleanupMode,
    PersistentSubagentProfile,
    RoutingMode,
    SpawnMode,
    SubagentConfig,
    SubagentRecord,
    SubagentResult,
    SubagentState,
    TokenUsage,
    normalize_thinking_level,
)
from spoon_bot.subagent.registry import SubagentRegistry
from spoon_bot.subagent.persistence import AgentDirectory, SubagentRunsFile, SubagentSweeper
from spoon_bot.bus.events import SubagentEvent

if TYPE_CHECKING:
    from spoon_bot.session.manager import SessionManager
    from spoon_bot.bus.queue import MessageBus

# Announce retry settings
_ANNOUNCE_MAX_RETRIES = 3
_ANNOUNCE_BASE_DELAY = 1.0  # seconds
# Steer rate limit
_STEER_MIN_INTERVAL = 2.0  # seconds between steers to the same agent
# Result truncation for wake messages
_WAKE_RESULT_TRUNCATE = 2000


class SubagentManager:
    """Manages sub-agent lifecycle.

    One SubagentManager instance is created per root AgentLoop. It:
    - Maintains a SubagentRegistry for lifecycle tracking
    - Runs each sub-agent as an independent asyncio.Task
    - Delivers results via asyncio.Queue (pull) AND bus injection (push/wake)
    - Shares the parent's SessionManager (sub-agents get unique session keys)
    - Enforces depth, children-per-agent, and total concurrency limits
    - Emits SubagentEvents for channel integrations (Discord, Telegram)
    """

    def __init__(
        self,
        *,
        session_manager: SessionManager,
        workspace: Path,
        max_depth: int = 2,
        max_children_per_agent: int = 5,
        max_total_subagents: int = 20,
        # Parent agent config for inheritance
        parent_model: str | None = None,
        parent_provider: str | None = None,
        parent_api_key: str | None = None,
        parent_base_url: str | None = None,
        parent_enable_skills: bool = True,
        default_model: str | None = None,
        default_tool_profile: str = "core",
        # Persistence settings
        persist_runs: bool = True,
        persist_file: str = "subagents/runs.json",
        archive_after_minutes: int = 60,
        sweeper_interval_seconds: int = 60,
        max_persistent_agents: int = 10,
    ) -> None:
        # Build persistence layer
        runs_file: SubagentRunsFile | None = None
        if persist_runs:
            persist_path = Path(persist_file)
            if not persist_path.is_absolute():
                persist_path = workspace / persist_path
            runs_file = SubagentRunsFile(persist_path)

        self.registry = SubagentRegistry(runs_file=runs_file)
        self._persistent_profiles: dict[str, PersistentSubagentProfile] = {
            profile.name: profile
            for profile in AgentDirectory.list_profiles(workspace)
        }

        # Background sweeper (started lazily in start_sweeper())
        self._sweeper: SubagentSweeper | None = (
            SubagentSweeper(
                registry=self.registry,
                session_manager=session_manager,
                archive_after_minutes=archive_after_minutes,
                interval_seconds=sweeper_interval_seconds,
            )
            if persist_runs
            else None
        )
        self.session_manager = session_manager
        self.workspace = workspace
        self.max_depth = max_depth
        self.max_children_per_agent = max_children_per_agent
        self.max_total_subagents = max_total_subagents
        self.max_persistent_agents = max_persistent_agents

        # Parent config used for inheritance
        self._parent_model = parent_model
        self._parent_provider = parent_provider
        self._parent_api_key = parent_api_key
        self._parent_base_url = parent_base_url
        self._parent_enable_skills = parent_enable_skills
        self._default_model = default_model
        self._default_tool_profile = default_tool_profile

        # Pull-based result delivery
        self._results: asyncio.Queue[SubagentResult] = asyncio.Queue()
        self._buffered_results: list[SubagentResult] = []
        self._results_lock = asyncio.Lock()

        # Push-based delivery — reference to the message bus
        self._bus: MessageBus | None = None

        # Current spawner context (updated per-process call from AgentLoop)
        self._current_spawner_session: str | None = None
        self._current_spawner_channel: str | None = None
        self._current_spawner_metadata: dict[str, Any] = {}
        self._current_spawner_reply_to: str | None = None

        # Live asyncio.Task handles — agent_id → Task
        self._tasks: dict[str, asyncio.Task[None]] = {}

        # Steer support — pending steer messages and timestamps
        self._steer_requests: dict[str, dict[str, Any]] = {}
        self._steer_timestamps: dict[str, float] = {}

        # Lifecycle event listeners
        self._event_listeners: list[Callable[[SubagentEvent], Awaitable[None] | None]] = []
        self._persistent_session_managers: dict[str, SessionManager] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_bus(self, bus: MessageBus) -> None:
        """Attach the MessageBus for push-based wake delivery."""
        self._bus = bus

    def set_spawner_context(
        self,
        *,
        session_key: str | None,
        channel: str | None,
        metadata: dict[str, Any] | None = None,
        reply_to: str | None = None,
    ) -> None:
        """Update the current spawner context.

        Called by AgentLoop at the start of each process() / process_with_thinking()
        so that any sub-agents spawned during that call know who to deliver results to.
        """
        self._current_spawner_session = session_key
        self._current_spawner_channel = channel
        self._current_spawner_metadata = dict(metadata or {})
        self._current_spawner_reply_to = reply_to

    def add_event_listener(
        self,
        listener: Callable[[SubagentEvent], Awaitable[None] | None],
    ) -> None:
        """Register a lifecycle event listener (e.g. for Discord notifications)."""
        self._event_listeners.append(listener)

    def _resolve_spawner_context(
        self,
        *,
        spawner_session_key: str | None = None,
        spawner_channel: str | None = None,
        spawner_metadata: dict[str, Any] | None = None,
        spawner_reply_to: str | None = None,
    ) -> tuple[str | None, str | None, dict[str, Any], str | None]:
        """Resolve explicit spawner context over manager-level defaults."""
        return (
            spawner_session_key
            if spawner_session_key is not None
            else self._current_spawner_session,
            spawner_channel
            if spawner_channel is not None
            else self._current_spawner_channel,
            dict(spawner_metadata)
            if spawner_metadata is not None
            else dict(self._current_spawner_metadata),
            spawner_reply_to
            if spawner_reply_to is not None
            else self._current_spawner_reply_to,
        )

    @staticmethod
    def _config_field_explicit(
        cfg: SubagentConfig,
        field_name: str,
    ) -> bool:
        """Return True when the caller explicitly set *field_name*."""
        return field_name in getattr(cfg, "model_fields_set", set())

    def _apply_default_config(
        self,
        cfg: SubagentConfig | None,
    ) -> SubagentConfig:
        """Apply manager-level subagent defaults without overriding explicit fields."""
        effective = cfg.model_copy(deep=True) if cfg is not None else SubagentConfig()
        default_tool_profile = str(self._default_tool_profile or "").strip()
        base_tool_profile = SubagentConfig.model_fields["tool_profile"].default

        if (
            self._default_model
            and not self._config_field_explicit(effective, "model")
            and not effective.model
        ):
            effective.model = self._default_model

        if (
            default_tool_profile
            and not self._config_field_explicit(effective, "tool_profile")
            and effective.tool_profile == base_tool_profile
        ):
            effective.tool_profile = default_tool_profile

        effective.thinking_level = normalize_thinking_level(effective.thinking_level)

        return effective

    @staticmethod
    def _config_allows_nested_subagents(cfg: SubagentConfig) -> bool:
        """Return True when the config allows nested child spawning."""
        return bool(
            cfg.allow_subagents or cfg.routing_mode == RoutingMode.ORCHESTRATED
        )

    def _resolve_effective_enable_skills(self, cfg: SubagentConfig) -> bool:
        """Resolve the effective skills toggle for a child agent."""
        if cfg.enable_skills is None:
            return self._parent_enable_skills
        return bool(cfg.enable_skills)

    def _record_visible_to_requester(
        self,
        record: SubagentRecord,
        *,
        spawner_session_key: str | None = None,
    ) -> bool:
        """Return True when *record* belongs to the requester's subagent lineage."""
        if spawner_session_key is None:
            return True

        current: SubagentRecord | None = record
        visited: set[str] = set()
        while current is not None and current.agent_id not in visited:
            if current.spawner_session_key == spawner_session_key:
                return True
            visited.add(current.agent_id)
            if not current.parent_id:
                break
            current = self.registry.get(current.parent_id)
        return False

    def _filter_records_for_requester(
        self,
        records: list[SubagentRecord],
        *,
        spawner_session_key: str | None = None,
    ) -> list[SubagentRecord]:
        """Filter records down to the requester-visible lineage."""
        if spawner_session_key is None:
            return list(records)
        return [
            record
            for record in records
            if self._record_visible_to_requester(
                record,
                spawner_session_key=spawner_session_key,
            )
        ]

    def _get_scoped_record(
        self,
        agent_id: str,
        *,
        spawner_session_key: str | None = None,
    ) -> SubagentRecord | None:
        """Return *agent_id* only when it is visible to the requester."""
        record = self.registry.get(agent_id)
        if record is None:
            return None
        if not self._record_visible_to_requester(
            record,
            spawner_session_key=spawner_session_key,
        ):
            return None
        return record

    @staticmethod
    def _result_matches_scope(
        result: SubagentResult,
        *,
        spawner_session_key: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        """Return True when *result* belongs to the requested requester scope."""
        if agent_id is not None and result.agent_id != agent_id:
            return False
        if run_id is not None and result.run_id != run_id:
            return False
        if (
            spawner_session_key is not None
            and result.spawner_session_key != spawner_session_key
        ):
            return False
        return True

    @staticmethod
    def _new_run_id() -> str:
        """Return a fresh run identifier for one execution attempt."""
        return f"run_{uuid4().hex[:12]}"

    @staticmethod
    def _resolve_child_enabled_tools(cfg: SubagentConfig) -> set[str] | None:
        """Resolve effective child tools, enabling ``spawn`` only for orchestrators."""
        if cfg.enabled_tools is not None:
            enabled = set(cfg.enabled_tools)
            if SubagentManager._config_allows_nested_subagents(cfg):
                enabled.add("spawn")
            else:
                enabled.discard("spawn")
            return enabled

        if not SubagentManager._config_allows_nested_subagents(cfg):
            return None

        enabled = set(TOOL_PROFILES.get(cfg.tool_profile, frozenset()))
        enabled.add("spawn")
        return enabled

    def _start_background_run(
        self,
        record: SubagentRecord,
        *,
        task_override: str | None = None,
        is_restart: bool = False,
    ) -> None:
        """Launch a tracked asyncio task for the given sub-agent record."""
        bg_task = asyncio.create_task(
            self._run_subagent(
                record,
                task_override=task_override,
                is_restart=is_restart,
            ),
            name=f"subagent-{record.agent_id}",
        )
        self._tasks[record.agent_id] = bg_task

    def _persistent_agent_names(self) -> set[str]:
        """Return all known persistent agent names from disk and registry."""
        names = set(self._persistent_profiles.keys())
        names.update(
            self._record_agent_name(record)
            for record in self.registry.list_all()
            if self._record_spawn_mode(record) == SpawnMode.SESSION and self._record_agent_name(record)
        )
        return names

    def _get_persistent_profile(self, name: str) -> PersistentSubagentProfile | None:
        """Return a persistent subagent profile by name."""
        return self._persistent_profiles.get(name)

    def _resolve_agent_directory(
        self,
        record: SubagentRecord,
    ) -> AgentDirectory | None:
        """Return the persistent agent directory for a session-mode record."""
        if self._record_spawn_mode(record) != SpawnMode.SESSION:
            return None

        agent_name = self._record_agent_name(record)
        if not agent_name:
            return None

        agent_dir = AgentDirectory(self.workspace, agent_name)
        agent_dir.ensure()
        expected_root = str(agent_dir.root)
        if record.agent_dir != expected_root:
            self.registry.update_fields(record.agent_id, agent_dir=expected_root)
            record.agent_dir = expected_root
            self._persist_session_agent_state(record.agent_id)
        return agent_dir

    def _session_manager_for_record(self, record: SubagentRecord) -> SessionManager:
        """Return the session manager that should back *record* transcripts."""
        agent_dir = self._resolve_agent_directory(record)
        if agent_dir is None:
            return self.session_manager

        agent_name = self._record_agent_name(record)
        if not agent_name:
            return self.session_manager

        session_manager = self._persistent_session_managers.get(agent_name)
        if session_manager is None:
            session_manager = agent_dir.build_session_manager()
            self._persistent_session_managers[agent_name] = session_manager

        self._migrate_persistent_session_if_needed(record, session_manager)
        return session_manager

    def _migrate_persistent_session_if_needed(
        self,
        record: SubagentRecord,
        target_manager: SessionManager,
    ) -> None:
        """Move a legacy shared session transcript into the agent-scoped store."""
        if target_manager is self.session_manager:
            return
        if self._record_spawn_mode(record) != SpawnMode.SESSION:
            return
        if not record.session_key:
            return

        existing = target_manager.get(record.session_key)
        if existing is not None:
            return

        legacy_session = self.session_manager.get(record.session_key)
        if legacy_session is None:
            return

        target_manager.save(legacy_session)
        logger.info(
            f"Migrated persistent agent session {record.session_key!r} "
            f"to {target_manager.sessions_dir}"
        )

        try:
            self.session_manager.archive(record.session_key)
        except Exception as exc:
            logger.warning(
                f"Failed to archive legacy session {record.session_key!r} "
                f"after migration: {exc}"
            )

    def _save_persistent_profile(self, profile: PersistentSubagentProfile) -> None:
        """Persist and cache a persistent subagent profile."""
        agent_dir = AgentDirectory(self.workspace, profile.name)
        agent_dir.ensure()
        agent_dir.save_profile_json(profile)
        self._persistent_profiles[profile.name] = profile

    def _discover_profile_session_key(
        self,
        profile: PersistentSubagentProfile,
    ) -> str | None:
        """Recover a persistent session key from agent-scoped transcripts."""
        if profile.session_key:
            return profile.session_key

        agent_dir = AgentDirectory(self.workspace, profile.name)
        if not agent_dir.exists():
            if profile.last_run_agent_id:
                legacy_session_key = f"subagent-{profile.last_run_agent_id}"
                if self.session_manager.get(legacy_session_key) is not None:
                    profile.session_key = legacy_session_key
                    self._save_persistent_profile(profile)
                    return legacy_session_key
            return None

        session_manager = self._persistent_session_managers.get(profile.name)
        owns_manager = session_manager is None
        if session_manager is None:
            session_manager = agent_dir.build_session_manager()
        try:
            if profile.last_run_agent_id:
                legacy_session_key = f"subagent-{profile.last_run_agent_id}"
                if (
                    session_manager.get(legacy_session_key) is not None
                    or self.session_manager.get(legacy_session_key) is not None
                ):
                    profile.session_key = legacy_session_key
                    self._save_persistent_profile(profile)
                    return legacy_session_key

            session_keys = session_manager.list_sessions()
            if len(session_keys) == 1:
                profile.session_key = session_keys[0]
                self._save_persistent_profile(profile)
                return session_keys[0]
        finally:
            if owns_manager:
                session_manager.close()

        return None

    @staticmethod
    def _record_agent_name(record: SubagentRecord) -> str | None:
        """Return the effective persistent agent name for a record."""
        return record.agent_name or record.config.agent_name

    @staticmethod
    def _record_spawn_mode(record: SubagentRecord) -> SpawnMode:
        """Return the effective spawn mode for a record."""
        return record.config.spawn_mode or record.spawn_mode

    def _allocate_persistent_agent_name(self, preferred_name: str) -> str:
        """Return a unique persistent agent name derived from *preferred_name*."""
        base = re.sub(r"[^a-z0-9_-]+", "-", preferred_name.lower()).strip("-_")
        if not base:
            base = "subagent"
        existing = self._persistent_agent_names()
        if base not in existing:
            return base
        suffix = 2
        while f"{base}-{suffix}" in existing:
            suffix += 1
        return f"{base}-{suffix}"

    @classmethod
    def _infer_persistent_subagent_profile(cls, description: str) -> dict[str, Any]:
        """Infer a persistent subagent profile from a natural-language description."""
        raw = description.strip().strip("。.!? ")
        normalized = cls._normalize_text(raw)
        keywords = [raw]
        match_examples = [raw]
        tool_profile = "coding"
        suggested_name = "subagent"

        if any(token in normalized for token in ["新闻", "news", "headline", "热点"]):
            suggested_name = "news-subagent"
            tool_profile = "research"
            keywords.extend([
                "今天的新闻",
                "今日新闻",
                "新闻摘要",
                "科技热点",
                "国际热点",
                "daily news",
                "news summary",
            ])
            system_prompt = (
                "You are a persistent news sub-agent. Your responsibility is to "
                "summarize current news and produce concise, structured digests. "
                "Prioritize fresh information, especially technology and international topics "
                "when the request asks for them."
            )
        elif any(token in normalized for token in ["认证", "登录", "password", "authentication", "login"]):
            suggested_name = "auth-subagent"
            tool_profile = "coding"
            keywords.extend([
                "登录失败",
                "密码重置",
                "authentication",
                "login issue",
                "password reset",
            ])
            system_prompt = (
                "You are a persistent authentication sub-agent. Handle login, "
                "password reset, and account recovery work directly."
            )
        elif any(token in normalized for token in ["docker", "部署", "deploy", "kubernetes", "ci/cd"]):
            suggested_name = "deploy-subagent"
            tool_profile = "coding"
            keywords.extend([
                "部署",
                "Docker",
                "CI/CD",
                "deployment",
            ])
            system_prompt = (
                "You are a persistent deployment sub-agent. Handle Docker, CI/CD, "
                "and deployment configuration requests."
            )
        else:
            ascii_tokens = re.findall(r"[a-z0-9_]{3,}", normalized)
            if ascii_tokens:
                suggested_name = "-".join(ascii_tokens[:2]) + "-subagent"
            system_prompt = (
                "You are a persistent sub-agent dedicated to a specific class of tasks. "
                f"Your specialization is: {raw}. Handle future matching requests directly."
            )

        deduped_keywords: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            stripped = keyword.strip()
            if not stripped:
                continue
            marker = stripped.lower()
            if marker in seen:
                continue
            seen.add(marker)
            deduped_keywords.append(stripped)

        return {
            "suggested_name": suggested_name,
            "specialization": raw,
            "tool_profile": tool_profile,
            "system_prompt": system_prompt,
            "match_keywords": deduped_keywords,
            "match_examples": match_examples,
        }

    @classmethod
    def _infer_persistent_subagent_profile(cls, description: str) -> dict[str, Any]:
        """Infer a persistent subagent profile from natural-language intent.

        This duplicate definition intentionally overrides the older heuristic
        above. It uses ASCII-safe unicode escapes for Chinese trigger words so
        the matching logic remains stable regardless of console encoding.
        """
        raw = description.strip().strip("。.!? ")
        normalized = cls._normalize_text(raw)
        keywords = [raw]
        match_examples = [raw]
        tool_profile = "coding"
        suggested_name = "subagent"

        news_terms = [
            "news",
            "headline",
            "daily news",
            "news summary",
            "today's news",
            "today news",
            "\u65b0\u95fb",
            "\u70ed\u70b9",
        ]
        auth_terms = [
            "authentication",
            "login",
            "password",
            "\u8ba4\u8bc1",
            "\u767b\u5f55",
            "\u5bc6\u7801",
        ]
        literature_terms = [
            "literature",
            "paper",
            "papers",
            "research paper",
            "academic research",
            "scholar",
            "\u6587\u732e",
            "\u8bba\u6587",
            "\u5b66\u672f",
            "\u7814\u7a76\u8d44\u6599",
        ]
        deploy_terms = [
            "docker",
            "deploy",
            "deployment",
            "kubernetes",
            "ci/cd",
            "\u90e8\u7f72",
        ]

        if any(term in normalized for term in news_terms):
            suggested_name = "news-subagent"
            tool_profile = "research"
            keywords.extend([
                "\u4eca\u5929\u7684\u65b0\u95fb",
                "\u4eca\u65e5\u65b0\u95fb",
                "\u65b0\u95fb\u6458\u8981",
                "\u79d1\u6280\u70ed\u70b9",
                "\u56fd\u9645\u70ed\u70b9",
                "daily news",
                "news summary",
            ])
            system_prompt = (
                "You are a persistent news sub-agent. Your responsibility is to "
                "summarize current news and produce concise, structured digests. "
                "Prioritize fresh information, especially technology and "
                "international topics when the request asks for them."
            )
        elif any(term in normalized for term in auth_terms):
            suggested_name = "auth-subagent"
            tool_profile = "coding"
            keywords.extend([
                "\u767b\u5f55\u5931\u8d25",
                "\u5bc6\u7801\u91cd\u7f6e",
                "authentication",
                "login issue",
                "password reset",
            ])
            system_prompt = (
                "You are a persistent authentication sub-agent. Handle login, "
                "password reset, and account recovery work directly."
            )
        elif any(term in normalized for term in literature_terms):
            suggested_name = "academic-research-subagent"
            tool_profile = "research"
            keywords.extend([
                "\u6587\u732e\u67e5\u8be2",
                "\u8bba\u6587\u68c0\u7d22",
                "\u8bba\u6587\u6458\u8981",
                "\u7814\u7a76\u8d44\u6599",
                "\u5b66\u672f\u7814\u7a76",
                "\u6587\u732e\u7efc\u8ff0",
                "\u671f\u520a\u8bba\u6587",
                "\u5b66\u672f\u6570\u636e\u5e93",
                "\u8bba\u6587\u603b\u7ed3",
                "paper",
                "papers",
                "research",
                "literature",
                "summary",
                "literature search",
                "paper search",
                "paper summary",
            ])
            system_prompt = (
                "You are a persistent academic research sub-agent. Handle "
                "literature search, paper discovery, paper summaries, "
                "research background synthesis, and reference collection."
            )
        elif any(term in normalized for term in deploy_terms):
            suggested_name = "deploy-subagent"
            tool_profile = "coding"
            keywords.extend([
                "\u90e8\u7f72",
                "Docker",
                "CI/CD",
                "deployment",
            ])
            system_prompt = (
                "You are a persistent deployment sub-agent. Handle Docker, "
                "CI/CD, and deployment configuration requests."
            )
        else:
            ascii_tokens = re.findall(r"[a-z0-9_]{3,}", normalized)
            if ascii_tokens:
                suggested_name = "-".join(ascii_tokens[:2]) + "-subagent"
            system_prompt = (
                "You are a persistent sub-agent dedicated to a specific class "
                f"of tasks. Your specialization is: {raw}. Handle future "
                "matching requests directly."
            )

        deduped_keywords: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            stripped = keyword.strip()
            if not stripped:
                continue
            marker = stripped.lower()
            if marker in seen:
                continue
            seen.add(marker)
            deduped_keywords.append(stripped)

        return {
            "suggested_name": suggested_name,
            "specialization": raw,
            "tool_profile": tool_profile,
            "system_prompt": system_prompt,
            "match_keywords": deduped_keywords,
            "match_examples": match_examples,
        }

    @classmethod
    def parse_persistent_subagent_request(cls, message: str) -> dict[str, Any] | None:
        """Parse a natural-language request to create a persistent subagent."""
        text = cls._strip_sender_prefix(message)
        patterns = [
            re.compile(
                r"^(?:请|帮我|麻烦你|请你)?创建(?:一个|个)?(?:专门的|持久的)?\s*(?:subagent|sub agent|子代理)\s*(?:来|用于|负责|专门处理|专门做)?(?P<body>.+)$",
                re.IGNORECASE,
            ),
            re.compile(
                r"^(?:please\s+)?create\s+(?:me\s+)?(?:a\s+)?(?:persistent\s+)?(?:subagent|sub agent)\s+(?:to|for)\s+(?P<body>.+)$",
                re.IGNORECASE,
            ),
        ]
        for pattern in patterns:
            match = pattern.match(text)
            if not match:
                continue
            body = match.group("body").strip().strip("。.!? ")
            if not body:
                return None
            profile = cls._infer_persistent_subagent_profile(body)
            profile["source_message"] = text
            return profile
        return None

    @staticmethod
    def _strip_sender_prefix(text: str) -> str:
        """Remove leading chat-style sender prefixes like ``[Name]:``."""
        return re.sub(r"^\s*\[[^\]]+\]:\s*", "", text, count=1).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", SubagentManager._strip_sender_prefix(text).lower()).strip()

    @classmethod
    def _tokenize_text(cls, text: str) -> set[str]:
        normalized = cls._normalize_text(text)
        tokens = {
            token
            for token in re.findall(r"[a-z0-9_]{3,}", normalized)
        }
        for segment in re.findall(r"[\u4e00-\u9fff]+", normalized):
            if len(segment) <= 4:
                tokens.add(segment)
            for n in (2, 3, 4):
                if len(segment) < n:
                    continue
                for idx in range(len(segment) - n + 1):
                    tokens.add(segment[idx:idx + n])
        return tokens

    @classmethod
    def _score_specialist_match(
        cls,
        *,
        message: str,
        profile: PersistentSubagentProfile,
    ) -> tuple[int, bool, list[str]]:
        """Score how strongly a user message matches a persistent subagent profile."""
        normalized = cls._normalize_text(message)
        message_tokens = cls._tokenize_text(message)
        score = 0
        strong_signal = False
        reasons: list[str] = []

        agent_name = profile.name.strip().lower()
        if agent_name and agent_name in normalized:
            score += 12
            strong_signal = True
            reasons.append(f"name:{agent_name}")

        for keyword in profile.match_keywords:
            kw = keyword.strip().lower()
            if not kw:
                continue
            if kw in normalized:
                score += 6
                strong_signal = True
                reasons.append(f"keyword:{kw}")
                continue

            kw_tokens = cls._tokenize_text(kw)
            if not kw_tokens:
                continue
            overlap = kw_tokens & message_tokens
            contains_cjk = bool(re.search(r"[\u4e00-\u9fff]", kw))
            required_overlap = 1 if contains_cjk else (2 if len(kw_tokens) > 1 else 1)
            if len(overlap) >= required_overlap:
                score += 5 if contains_cjk else 4
                strong_signal = True
                reasons.append(f"keyword_tokens:{kw}")

        specialization = (profile.specialization or "").strip()
        if specialization:
            spec_tokens = cls._tokenize_text(specialization)
            overlap = sorted(spec_tokens & message_tokens)
            if overlap:
                score += min(5, len(overlap))
                reasons.append(f"specialization:{','.join(overlap[:4])}")

        for example in profile.match_examples:
            overlap = cls._tokenize_text(example) & message_tokens
            contains_cjk = bool(re.search(r"[\u4e00-\u9fff]", example))
            if len(overlap) >= (1 if contains_cjk else 2):
                score += 3 if contains_cjk else 2
                reasons.append("example")
                break

        return score, strong_signal, reasons

    def _persist_session_agent_state(self, agent_id: str) -> None:
        """Persist agent.json metadata for a session-mode agent if present."""
        record = self.registry.get(agent_id)
        agent_name = self._record_agent_name(record) if record else None
        if not record or self._record_spawn_mode(record) != SpawnMode.SESSION or not agent_name:
            return
        try:
            base_profile = self._get_persistent_profile(agent_name)
            if base_profile is None:
                profile = PersistentSubagentProfile.from_subagent_config(
                    name=agent_name,
                    config=record.config,
                    created_at=record.created_at,
                    session_key=record.session_key,
                )
            else:
                profile = base_profile.model_copy(deep=True)
                updated_cfg = record.config
                profile.role = updated_cfg.role
                profile.model = record.model_name or updated_cfg.model or profile.model
                profile.provider = updated_cfg.provider or profile.provider
                profile.api_key = updated_cfg.api_key or profile.api_key
                profile.base_url = updated_cfg.base_url or profile.base_url
                profile.max_iterations = updated_cfg.max_iterations
                profile.system_prompt = updated_cfg.system_prompt or profile.system_prompt
                profile.tool_profile = updated_cfg.tool_profile
                profile.enabled_tools = updated_cfg.enabled_tools
                profile.enable_skills = updated_cfg.enable_skills
                profile.context_window = updated_cfg.context_window
                profile.thinking_level = updated_cfg.thinking_level
                profile.timeout_seconds = updated_cfg.timeout_seconds
                profile.cleanup = updated_cfg.cleanup
                profile.specialization = updated_cfg.specialization or profile.specialization
                profile.auto_route = updated_cfg.auto_route
                profile.match_keywords = list(updated_cfg.match_keywords)
                profile.match_examples = list(updated_cfg.match_examples)
                profile.routing_mode = updated_cfg.routing_mode
                profile.allow_subagents = updated_cfg.allow_subagents
                profile.session_key = record.session_key
            profile.last_active_at = record.completed_at or record.started_at or record.created_at
            profile.last_run_agent_id = record.agent_id
            profile.last_run_state = record.state.value
            self._save_persistent_profile(profile)
        except Exception as exc:
            logger.debug(f"Failed to update agent.json for {agent_name!r}: {exc}")

    @staticmethod
    def _merge_resume_config(
        base: SubagentConfig,
        override: SubagentConfig | None,
    ) -> SubagentConfig:
        """Merge explicit override fields onto an existing sub-agent config."""
        effective = base.model_copy(deep=True)
        if override is None:
            return effective
        for key, value in override.model_dump(exclude_unset=True).items():
            setattr(effective, key, value)
        return effective

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def spawn(
        self,
        task: str,
        *,
        label: str = "",
        parent_id: str | None = None,
        config: SubagentConfig | None = None,
        spawner_session_key: str | None = None,
        spawner_channel: str | None = None,
        spawner_metadata: dict[str, Any] | None = None,
        spawner_reply_to: str | None = None,
        allow_existing_profile_name: bool = False,
        session_key_override: str | None = None,
    ) -> SubagentRecord:
        """Spawn a new sub-agent to work on *task*.

        Validates all limits, registers the sub-agent, then starts it as
        a background asyncio.Task.

        Args:
            task:      The message/instruction for the sub-agent.
            label:     Short human-readable label (truncated from task if empty).
            parent_id: ID of the parent sub-agent (None = spawned by root).
            config:    Optional config overrides.

        Returns:
            The newly created SubagentRecord.

        Raises:
            ValueError: If any limit is exceeded or config is invalid.
        """
        cfg = self._apply_default_config(config)

        # --- Depth check ---
        parent_depth = 0
        parent_record: SubagentRecord | None = None
        if parent_id:
            parent_record = self.registry.get(parent_id)
            if parent_record:
                parent_depth = parent_record.depth
                if not self._config_allows_nested_subagents(parent_record.config):
                    raise ValueError(
                        "Nested sub-agent spawning is disabled for this parent. "
                        "Enable allow_subagents=true only for explicit orchestrator agents."
                    )
        child_depth = parent_depth + 1

        if child_depth > self.max_depth:
            raise ValueError(
                f"Max spawn depth ({self.max_depth}) exceeded. "
                f"Parent depth={parent_depth}, requested child depth={child_depth}."
            )

        # --- Children-per-agent limit ---
        if parent_id:
            active_children = self.registry.count_active_children(parent_id)
            if active_children >= self.max_children_per_agent:
                raise ValueError(
                    f"Max children per agent ({self.max_children_per_agent}) "
                    f"already reached for parent {parent_id}."
                )

        # --- Total sub-agent limit ---
        total_active = self.registry.count_active_total()
        if total_active >= self.max_total_subagents:
            raise ValueError(
                f"Max total active sub-agents ({self.max_total_subagents}) reached."
            )

        # --- Session-mode validations ---
        agent_dir_path: str | None = None
        if cfg.spawn_mode == SpawnMode.SESSION:
            if not cfg.agent_name:
                raise ValueError(
                    "agent_name is required when spawn_mode='session'."
                )
            agent_name = cfg.agent_name
            # Validate name: alphanumeric, hyphens, underscores only
            import re
            if not re.match(r"^[a-zA-Z0-9_-]+$", agent_name):
                raise ValueError(
                    f"agent_name {agent_name!r} is invalid. "
                    "Use only letters, digits, hyphens, and underscores."
                )
            existing_names = set(self._persistent_profiles.keys())
            if agent_name in existing_names and not allow_existing_profile_name:
                raise ValueError(
                    f"Persistent agent {agent_name!r} already exists. "
                    "Use resume_agent / spawn(action='resume') to reuse it."
                )
            if agent_name not in existing_names and len(existing_names) >= self.max_persistent_agents:
                raise ValueError(
                    f"Max persistent agents ({self.max_persistent_agents}) reached."
                )
            # Set up agent directory
            agent_dir = AgentDirectory(self.workspace, agent_name)
            agent_dir.ensure()
            agent_dir_path = str(agent_dir.root)

        # --- Resolve effective model name ---
        effective_model = cfg.model or self._parent_model
        (
            resolved_spawner_session,
            resolved_spawner_channel,
            resolved_spawner_metadata,
            resolved_spawner_reply_to,
        ) = self._resolve_spawner_context(
            spawner_session_key=spawner_session_key,
            spawner_channel=spawner_channel,
            spawner_metadata=spawner_metadata,
            spawner_reply_to=spawner_reply_to,
        )

        # --- Create and register record ---
        record = SubagentRecord(
            parent_id=parent_id,
            depth=child_depth,
            label=label or task[:60],
            task=task,
            config=cfg,
            session_key=session_key_override or "",
            model_name=effective_model,
            spawner_session_key=resolved_spawner_session,
            spawner_channel=resolved_spawner_channel,
            spawner_metadata=resolved_spawner_metadata,
            spawner_reply_to=resolved_spawner_reply_to,
            spawn_mode=cfg.spawn_mode,
            cleanup=cfg.cleanup,
            agent_name=cfg.agent_name,
            agent_dir=agent_dir_path,
        )

        self.registry.register(record)
        self._persist_session_agent_state(record.agent_id)
        logger.info(
            f"Sub-agent {record.agent_id!r} registered: "
            f"run_id={record.run_id}, depth={record.depth}, "
            f"label={record.label!r}, model={effective_model!r}"
        )

        # Emit lifecycle event
        await self._emit_event(SubagentEvent(
            event_type="spawning",
            agent_id=record.agent_id,
            label=record.label,
            parent_id=record.parent_id,
            depth=record.depth,
            model_name=effective_model,
            spawner_session_key=record.spawner_session_key,
            spawner_channel=record.spawner_channel,
            metadata={"run_id": record.run_id},
        ))

        # --- Start background execution ---
        self._start_background_run(record)

        return record

    async def collect_results(
        self,
        timeout: float = 0.0,
        *,
        spawner_session_key: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[SubagentResult]:
        """Collect scoped results without draining unrelated sub-agent outputs.

        Args:
            timeout: If > 0, wait up to this many seconds for at least one
                     result before returning. If 0, return immediately.

        Returns:
            List of SubagentResult (may be empty).
        """
        results: list[SubagentResult] = []
        deadline = time.monotonic() + timeout if timeout > 0 else None

        while True:
            async with self._results_lock:
                retained: list[SubagentResult] = []
                for buffered in self._buffered_results:
                    if self._result_matches_scope(
                        buffered,
                        spawner_session_key=spawner_session_key,
                        agent_id=agent_id,
                        run_id=run_id,
                    ):
                        results.append(buffered)
                    else:
                        retained.append(buffered)
                self._buffered_results = retained

            try:
                if results:
                    result = self._results.get_nowait()
                elif deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return results
                    result = await asyncio.wait_for(self._results.get(), timeout=remaining)
                else:
                    result = self._results.get_nowait()
            except asyncio.QueueEmpty:
                return results
            except asyncio.TimeoutError:
                return results

            if self._result_matches_scope(
                result,
                spawner_session_key=spawner_session_key,
                agent_id=agent_id,
                run_id=run_id,
            ):
                results.append(result)
            else:
                async with self._results_lock:
                    self._buffered_results.append(result)

    async def cancel(
        self,
        agent_id: str,
        *,
        cascade: bool = True,
        spawner_session_key: str | None = None,
    ) -> bool:
        """Cancel a running or pending sub-agent.

        Args:
            agent_id: The agent to cancel.
            cascade:  If True (default), also cancel all descendants.

        Returns:
            True if the task was found and cancellation was requested.
        """
        record = self._get_scoped_record(
            agent_id,
            spawner_session_key=spawner_session_key,
        )
        if record is None:
            return False

        found = False
        if cascade:
            # Cancel descendants first (deepest first)
            descendants = self.registry.get_descendants(agent_id)
            for desc in reversed(descendants):
                task = self._tasks.get(desc.agent_id)
                if task and not task.done():
                    task.cancel()
                    logger.info(f"Cascade cancel: sub-agent {desc.agent_id!r}")
                    found = True

        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Cancellation requested for sub-agent {agent_id}")
            found = True
        return found

    async def cancel_all(
        self,
        parent_id: str | None = None,
        *,
        spawner_session_key: str | None = None,
    ) -> int:
        """Cancel all active sub-agents, optionally filtered by parent.

        Always uses cascade kill.

        Returns:
            Number of top-level sub-agents whose cancellation was requested.
        """
        if parent_id:
            records = self.registry.list_by_parent(parent_id)
            records = self._filter_records_for_requester(
                records,
                spawner_session_key=spawner_session_key,
            )
        else:
            visible_records = self._filter_records_for_requester(
                self.registry.list_all(),
                spawner_session_key=spawner_session_key,
            )
            visible_ids = {record.agent_id for record in visible_records}
            records = [
                record
                for record in visible_records
                if record.parent_id not in visible_ids
            ]

        count = 0
        for record in records:
            if record.state in (SubagentState.PENDING, SubagentState.RUNNING):
                if await self.cancel(
                    record.agent_id,
                    cascade=True,
                    spawner_session_key=spawner_session_key,
                ):
                    count += 1
        return count

    async def steer(
        self,
        agent_id: str,
        new_message: str,
        *,
        spawner_session_key: str | None = None,
    ) -> dict[str, Any]:
        """Redirect a running sub-agent with a new message.

        The current run is aborted and the sub-agent is restarted with
        *new_message* appended to the existing session history.

        Rate limited: at most one steer per _STEER_MIN_INTERVAL seconds.

        Returns:
            Dict with keys: status, agent_id, label, message.
        """
        record = self._get_scoped_record(
            agent_id,
            spawner_session_key=spawner_session_key,
        )
        if record is None:
            return {"status": "not_found", "agent_id": agent_id, "message": "Agent not found."}

        if record.state not in (SubagentState.PENDING, SubagentState.RUNNING):
            return {
                "status": "done",
                "agent_id": agent_id,
                "label": record.label,
                "message": f"Agent is already in terminal state: {record.state.value}",
            }

        # Rate limiting
        last_steer = self._steer_timestamps.get(agent_id, 0.0)
        elapsed_since = time.time() - last_steer
        if elapsed_since < _STEER_MIN_INTERVAL:
            wait = round(_STEER_MIN_INTERVAL - elapsed_since, 1)
            return {
                "status": "rate_limited",
                "agent_id": agent_id,
                "label": record.label,
                "message": f"Rate limited. Try again in {wait}s.",
            }

        # Queue the steer request (processed in _run_subagent's CancelledError handler)
        next_run_id = self._new_run_id()
        self._steer_requests[agent_id] = {
            "message": new_message,
            "run_id": next_run_id,
        }
        self._steer_timestamps[agent_id] = time.time()

        # Cancel current task (triggers the steer restart)
        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()
            logger.info(
                f"Steer requested for sub-agent {agent_id!r}: "
                f"{new_message[:60]!r}"
            )
            return {
                "status": "accepted",
                "agent_id": agent_id,
                "label": record.label,
                "run_id": next_run_id,
                "message": (
                    f"Steer accepted. Sub-agent {agent_id} will be redirected "
                    f"to run {next_run_id}."
                ),
            }
        else:
            # Task is done — start a fresh run
            self._steer_requests.pop(agent_id, None)
            return {
                "status": "done",
                "agent_id": agent_id,
                "label": record.label,
                "message": "Agent already finished. Use spawn to create a new one.",
            }

    async def get_info(
        self,
        agent_id: str,
        *,
        spawner_session_key: str | None = None,
    ) -> dict[str, Any] | None:
        """Return detailed metadata for a sub-agent.

        Returns None if the agent is not found.
        """
        record = self._get_scoped_record(
            agent_id,
            spawner_session_key=spawner_session_key,
        )
        if record is None:
            return None

        now = time.time()
        elapsed: float | None = None
        if record.started_at:
            end = record.completed_at or now
            elapsed = round(end - record.started_at, 2)

        pending_desc = self.registry.count_pending_descendants(agent_id)

        return {
            "agent_id": record.agent_id,
            "run_id": record.run_id,
            "label": record.label,
            "state": record.state.value,
            "task": record.task,
            "depth": record.depth,
            "parent_id": record.parent_id,
            "session_key": record.session_key,
            "model": record.model_name,
            "tool_profile": record.config.tool_profile,
            "specialization": record.config.specialization,
            "auto_route": record.config.auto_route,
            "match_keywords": list(record.config.match_keywords),
            "match_examples": list(record.config.match_examples),
            "routing_mode": record.config.routing_mode.value,
            "allow_subagents": self._config_allows_nested_subagents(record.config),
            "enable_skills": record.config.enable_skills,
            "effective_enable_skills": self._resolve_effective_enable_skills(record.config),
            "max_iterations": record.config.max_iterations,
            "thinking_level": normalize_thinking_level(record.config.thinking_level),
            "timeout_seconds": record.config.timeout_seconds,
            "children": record.children,
            "pending_descendants": pending_desc,
            "created_at": record.created_at,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
            "elapsed_seconds": elapsed,
            "result_preview": (record.result or "")[:300] if record.result else None,
            "error": record.error,
            "token_usage": (
                record.token_usage.model_dump() if record.token_usage else None
            ),
        }

    async def resume_agent(
        self,
        agent_name: str,
        task: str,
        *,
        label: str | None = None,
        config: SubagentConfig | None = None,
        parent_id: str | None = None,
        spawner_session_key: str | None = None,
        spawner_channel: str | None = None,
        spawner_metadata: dict[str, Any] | None = None,
        spawner_reply_to: str | None = None,
    ) -> SubagentRecord:
        """Re-invoke a persistent session-mode agent with a new task.

        Finds the most recent record for *agent_name*, verifies it is a
        session-mode agent in a terminal state, then restarts it using
        the existing session_key (preserving conversation context).

        Args:
            agent_name: The name of the persistent agent to resume.
            task:       New task/instruction to send to the agent.
            config:     Optional config overrides (defaults to original config).

        Returns:
            The updated SubagentRecord.

        Raises:
            ValueError: If the agent is not found, not a session-mode agent,
                        or is currently running.
        """
        # Find record by agent_name
        matching = [
            r for r in self.registry.list_all()
            if self._record_agent_name(r) == agent_name and self._record_spawn_mode(r) == SpawnMode.SESSION
        ]
        if not matching:
            profile = self._get_persistent_profile(agent_name)
            if profile is not None:
                effective_config = self._apply_default_config(
                    self._merge_resume_config(profile.to_subagent_config(), config)
                )
                effective_config.spawn_mode = SpawnMode.SESSION
                effective_config.agent_name = agent_name
                return await self.spawn(
                    task=task,
                    label=label or task[:60],
                    config=effective_config,
                    parent_id=parent_id,
                    spawner_session_key=spawner_session_key,
                    spawner_channel=spawner_channel,
                    spawner_metadata=spawner_metadata,
                    spawner_reply_to=spawner_reply_to,
                    allow_existing_profile_name=True,
                    session_key_override=self._discover_profile_session_key(profile),
                )
            raise ValueError(
                f"No session-mode agent named {agent_name!r} found. "
                "Create it first before resuming it."
            )
        # Take the most recently created
        record = max(matching, key=lambda r: r.created_at)

        if record.state in (SubagentState.PENDING, SubagentState.RUNNING):
            raise ValueError(
                f"Agent {agent_name!r} ({record.agent_id}) is currently "
                f"{record.state.value}. Wait for it to complete first."
            )

        effective_config = self._apply_default_config(
            self._merge_resume_config(record.config, config)
        )
        effective_config.spawn_mode = SpawnMode.SESSION
        effective_config.agent_name = agent_name
        effective_model = effective_config.model or self._parent_model
        next_run_id = self._new_run_id()
        (
            resolved_spawner_session,
            resolved_spawner_channel,
            resolved_spawner_metadata,
            resolved_spawner_reply_to,
        ) = self._resolve_spawner_context(
            spawner_session_key=spawner_session_key,
            spawner_channel=spawner_channel,
            spawner_metadata=spawner_metadata,
            spawner_reply_to=spawner_reply_to,
        )
        label = label or task[:60]
        self.registry.prepare_for_resume(
            record.agent_id,
            task=task,
            label=label,
            run_id=next_run_id,
            parent_id=parent_id,
            spawner_session_key=resolved_spawner_session,
            spawner_channel=resolved_spawner_channel,
            spawner_metadata=resolved_spawner_metadata,
            spawner_reply_to=resolved_spawner_reply_to,
            model_name=effective_model,
            config=effective_config,
        )
        self._persist_session_agent_state(record.agent_id)

        logger.info(
            f"Resuming session agent {agent_name!r} ({record.agent_id}): "
            f"run_id={record.run_id}, task={task[:60]!r}"
        )

        # Launch background task (reuses existing session_key = preserved context)
        # is_restart=False so the PENDING→RUNNING transition fires properly
        await self._emit_event(SubagentEvent(
            event_type="spawning",
            agent_id=record.agent_id,
            label=record.label,
            parent_id=record.parent_id,
            depth=record.depth,
            model_name=record.model_name,
            spawner_session_key=record.spawner_session_key,
            spawner_channel=record.spawner_channel,
            metadata={"run_id": record.run_id},
        ))
        self._start_background_run(record)

        return record

    def find_best_auto_route_specialist(self, task: str) -> dict[str, Any] | None:
        """Return the best persistent subagent profile for *task*, or None."""
        candidates: list[dict[str, Any]] = []
        profiles: dict[str, PersistentSubagentProfile] = dict(self._persistent_profiles)
        for record in self.registry.list_all():
            agent_name = self._record_agent_name(record)
            if (
                agent_name
                and agent_name not in profiles
                and self._record_spawn_mode(record) == SpawnMode.SESSION
                and record.config.auto_route
            ):
                profiles[agent_name] = PersistentSubagentProfile.from_subagent_config(
                    name=agent_name,
                    config=record.config,
                    created_at=record.created_at,
                    last_active_at=record.completed_at or record.started_at or record.created_at,
                    last_run_agent_id=record.agent_id,
                    last_run_state=record.state.value,
                    session_key=record.session_key,
                )

        for profile in profiles.values():
            if not profile.auto_route:
                continue
            score, strong_signal, reasons = self._score_specialist_match(
                message=task,
                profile=profile,
            )
            if score <= 0:
                continue
            candidates.append({
                "profile": profile,
                "agent_name": profile.name,
                "score": score,
                "strong_signal": strong_signal,
                "reasons": reasons,
            })

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item["score"],
                len(item["reasons"]),
                item["profile"].created_at,
            ),
            reverse=True,
        )
        best = candidates[0]
        if not best["strong_signal"] or best["score"] < 7:
            return None
        if len(candidates) > 1 and candidates[1]["score"] >= best["score"] - 2:
            logger.info(
                "Persistent specialist routing skipped due to ambiguous match: "
                f"{best['agent_name']} score={best['score']} vs "
                f"{candidates[1]['agent_name']} score={candidates[1]['score']}"
            )
            return None

        return {
            "agent_name": best["agent_name"],
            "score": best["score"],
            "reasons": best["reasons"],
            "specialization": best["profile"].specialization,
            "profile": best["profile"],
        }

    def create_persistent_subagent(
        self,
        *,
        description: str,
        agent_name: str | None = None,
        config: SubagentConfig | None = None,
        label: str | None = None,
    ) -> PersistentSubagentProfile:
        """Create a persistent subagent profile without immediately running it."""
        profile = self._infer_persistent_subagent_profile(description)
        cfg = (config.model_copy(deep=True) if config else SubagentConfig())
        cfg.thinking_level = normalize_thinking_level(cfg.thinking_level)
        cfg.spawn_mode = SpawnMode.SESSION
        cfg.auto_route = True if config is None else cfg.auto_route
        cfg.specialization = cfg.specialization or profile["specialization"]
        if config is None or cfg.tool_profile == "core":
            cfg.tool_profile = profile["tool_profile"]
        cfg.system_prompt = cfg.system_prompt or profile["system_prompt"]
        if not cfg.match_keywords:
            cfg.match_keywords = list(profile["match_keywords"])
        if not cfg.match_examples:
            cfg.match_examples = list(profile["match_examples"])

        existing_names = self._persistent_agent_names()
        if agent_name:
            normalized_name = agent_name.strip()
            if normalized_name and normalized_name not in existing_names and len(existing_names) >= self.max_persistent_agents:
                raise ValueError(
                    f"Max persistent agents ({self.max_persistent_agents}) reached."
                )
        elif len(existing_names) >= self.max_persistent_agents:
            raise ValueError(
                f"Max persistent agents ({self.max_persistent_agents}) reached."
            )

        chosen_name = self._allocate_persistent_agent_name(
            agent_name or profile["suggested_name"]
        )
        cfg.agent_name = chosen_name

        persistent_profile = PersistentSubagentProfile.from_subagent_config(
            name=chosen_name,
            config=cfg,
        )
        self._save_persistent_profile(persistent_profile)
        logger.info(
            f"Persistent subagent created: {chosen_name!r} "
            f"(tool_profile={cfg.tool_profile}, auto_route={cfg.auto_route})"
        )
        return persistent_profile

    async def dispatch_persistent_subagent(
        self,
        *,
        agent_name: str,
        task: str,
        spawner_session_key: str | None = None,
        spawner_channel: str | None = None,
        spawner_metadata: dict[str, Any] | None = None,
        spawner_reply_to: str | None = None,
    ) -> SubagentRecord:
        """Start or resume a persistent subagent session from its profile."""
        profile = self._get_persistent_profile(agent_name)
        if profile is None:
            raise ValueError(f"No persistent subagent profile named {agent_name!r} found.")

        dispatch_task = task
        if profile.routing_mode == RoutingMode.ORCHESTRATED:
            dispatch_task = (
                "Handle this request as the owning orchestrator for your specialist area. "
                "You may decompose the work into nested sub-agents when useful, then "
                "return one final synthesized answer.\n\n"
                f"User request:\n{task}"
            )

        matching = [
            r for r in self.registry.list_all()
            if self._record_agent_name(r) == agent_name and self._record_spawn_mode(r) == SpawnMode.SESSION
        ]
        if matching:
            latest = max(matching, key=lambda r: r.created_at)
            if latest.state in (SubagentState.PENDING, SubagentState.RUNNING):
                raise ValueError(
                    f"Persistent subagent {agent_name!r} is already running. "
                    "Wait for it to finish before dispatching another task."
                )
        return await self.resume_agent(
            agent_name=agent_name,
            task=dispatch_task,
            label=task[:60],
            spawner_session_key=spawner_session_key,
            spawner_channel=spawner_channel,
            spawner_metadata=spawner_metadata,
            spawner_reply_to=spawner_reply_to,
        )

    async def resume_task(
        self,
        task_id: str,
        task: str,
        *,
        config: SubagentConfig | None = None,
        parent_id: str | None = None,
        spawner_session_key: str | None = None,
        spawner_channel: str | None = None,
        spawner_metadata: dict[str, Any] | None = None,
        spawner_reply_to: str | None = None,
    ) -> SubagentRecord:
        """Resume an existing sub-agent session by task_id / agent_id."""
        record = self.registry.get(task_id)
        if record is None:
            raise ValueError(f"Sub-agent task_id {task_id!r} was not found.")

        if record.state in (SubagentState.PENDING, SubagentState.RUNNING):
            raise ValueError(
                f"Sub-agent {task_id!r} is currently {record.state.value}. "
                "Wait for it to complete before resuming it."
            )

        effective_config = self._apply_default_config(
            self._merge_resume_config(record.config, config)
        )
        effective_config.spawn_mode = self._record_spawn_mode(record)
        if record.agent_name and not effective_config.agent_name:
            effective_config.agent_name = record.agent_name
        effective_model = effective_config.model or self._parent_model
        next_run_id = self._new_run_id()
        (
            resolved_spawner_session,
            resolved_spawner_channel,
            resolved_spawner_metadata,
            resolved_spawner_reply_to,
        ) = self._resolve_spawner_context(
            spawner_session_key=spawner_session_key,
            spawner_channel=spawner_channel,
            spawner_metadata=spawner_metadata,
            spawner_reply_to=spawner_reply_to,
        )
        label = task[:60]
        self.registry.prepare_for_resume(
            record.agent_id,
            task=task,
            label=label,
            run_id=next_run_id,
            parent_id=parent_id,
            spawner_session_key=resolved_spawner_session,
            spawner_channel=resolved_spawner_channel,
            spawner_metadata=resolved_spawner_metadata,
            spawner_reply_to=resolved_spawner_reply_to,
            model_name=effective_model,
            config=effective_config,
        )
        self._persist_session_agent_state(record.agent_id)

        logger.info(
            f"Resuming sub-agent task {task_id!r}: "
            f"run_id={record.run_id}, task={task[:60]!r}"
        )

        await self._emit_event(SubagentEvent(
            event_type="spawning",
            agent_id=record.agent_id,
            label=record.label,
            parent_id=record.parent_id,
            depth=record.depth,
            model_name=record.model_name,
            spawner_session_key=record.spawner_session_key,
            spawner_channel=record.spawner_channel,
            metadata={"run_id": record.run_id},
        ))
        self._start_background_run(record)

        return record

    def list_persistent_agents(self) -> list[dict[str, Any]]:
        """Return metadata for all session-mode agents (active and past)."""
        result: list[dict[str, Any]] = []
        runtime_by_name: dict[str, SubagentRecord] = {}
        for record in self.registry.list_all():
            if self._record_spawn_mode(record) != SpawnMode.SESSION or not self._record_agent_name(record):
                continue
            name = self._record_agent_name(record)
            existing = runtime_by_name.get(name)
            if existing is None or record.created_at > existing.created_at:
                runtime_by_name[name] = record

        for name, profile in sorted(self._persistent_profiles.items()):
            runtime_record = runtime_by_name.get(name)
            result.append({
                "agent_id": runtime_record.agent_id if runtime_record else profile.last_run_agent_id,
                "agent_name": name,
                "label": runtime_record.label if runtime_record else f"persistent:{name}",
                "state": runtime_record.state.value if runtime_record else (profile.last_run_state or "idle"),
                "model": runtime_record.model_name if runtime_record else profile.model,
                "agent_dir": str(AgentDirectory(self.workspace, name).root),
                "specialization": profile.specialization,
                "auto_route": profile.auto_route,
                "allow_subagents": self._config_allows_nested_subagents(
                    profile.to_subagent_config()
                ),
                "match_keywords": list(profile.match_keywords),
                "match_examples": list(profile.match_examples),
                "routing_mode": profile.routing_mode.value,
                "created_at": profile.created_at,
                "last_active_at": runtime_record.completed_at if runtime_record and runtime_record.completed_at else profile.last_active_at or profile.created_at,
            })
        return result

    async def start_sweeper(self) -> None:
        """Start the background archive sweeper (call after the event loop is running)."""
        if self._sweeper is not None:
            await self._sweeper.start()

    async def stop_sweeper(self) -> None:
        """Stop the background archive sweeper."""
        if self._sweeper is not None:
            await self._sweeper.stop()

    async def cleanup(self) -> None:
        """Shut down all sub-agents, stop the sweeper, and clean up sessions."""
        # Stop sweeper first so it doesn't race with cleanup
        await self.stop_sweeper()

        cancelled = await self.cancel_all()
        if cancelled:
            logger.info(f"Cancelled {cancelled} sub-agent(s) during cleanup")

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        for record in self.registry.list_all():
            # Preserve session data for persistent (session-mode) agents
            if self._record_spawn_mode(record) == SpawnMode.SESSION:
                continue
            try:
                self.session_manager.delete(record.session_key)
            except Exception:
                pass

        for session_manager in self._persistent_session_managers.values():
            try:
                session_manager.close()
            except Exception:
                pass
        self._persistent_session_managers.clear()

        self._tasks.clear()
        logger.info("SubagentManager cleanup complete")

    def get_status_summary(
        self,
        parent_id: str | None = None,
        *,
        spawner_session_key: str | None = None,
    ) -> dict[str, Any]:
        """Return a rich summary dict of sub-agent statuses.

        Separates active (pending/running) from recent (terminal) records.
        Includes model name and pending descendant counts.
        """
        records = (
            self.registry.list_by_parent(parent_id)
            if parent_id
            else self.registry.list_all()
        )
        records = self._filter_records_for_requester(
            records,
            spawner_session_key=spawner_session_key,
        )

        active: list[dict[str, Any]] = []
        recent: list[dict[str, Any]] = []

        now = time.time()
        for r in records:
            elapsed: float | None = None
            if r.started_at:
                end = r.completed_at or now
                elapsed = round(end - r.started_at, 1)

            pending_desc = self.registry.count_pending_descendants(r.agent_id)

            entry = {
                "agent_id": r.agent_id,
                "run_id": r.run_id,
                "label": r.label,
                "state": r.state.value,
                "depth": r.depth,
                "model": r.model_name,
                "elapsed_seconds": elapsed,
                "pending_descendants": pending_desc,
                "spawn_mode": r.spawn_mode.value,
                "agent_name": r.agent_name,
                "allow_subagents": self._config_allows_nested_subagents(r.config),
            }

            if r.state in (SubagentState.PENDING, SubagentState.RUNNING):
                active.append(entry)
            else:
                recent.append(entry)

        return {
            "total": len(records),
            "active": active,
            "recent": recent,
            # Legacy field for backward compat with existing status handler
            "by_state": {
                e["state"]: [e2 for e2 in records
                              if e2.state.value == e["state"]]
                for e in (active + recent)
            },
        }

    # ------------------------------------------------------------------
    # Internal: background execution
    # ------------------------------------------------------------------

    async def _run_subagent(
        self,
        record: SubagentRecord,
        task_override: str | None = None,
        is_restart: bool = False,
    ) -> None:
        """Background coroutine: create AgentLoop → run task → deliver result."""
        from spoon_bot.agent.loop import AgentLoop

        task_text = task_override or record.task
        started_at = time.time()

        if not is_restart:
            self.registry.transition(
                record.agent_id,
                SubagentState.RUNNING,
                started_at=started_at,
            )
        else:
            self.registry.update_fields(
                record.agent_id,
                state=SubagentState.RUNNING,
                started_at=started_at,
            )
        self._persist_session_agent_state(record.agent_id)
        await self._emit_event(SubagentEvent(
            event_type="started",
            agent_id=record.agent_id,
            label=record.label,
            parent_id=record.parent_id,
            depth=record.depth,
            model_name=record.model_name,
            spawner_session_key=record.spawner_session_key,
            spawner_channel=record.spawner_channel,
            metadata={"run_id": record.run_id, "restarted": is_restart},
        ))

        logger.info(
            f"Sub-agent {record.agent_id!r} starting run {record.run_id!r}: "
            f"{task_text[:80]!r}"
        )

        child_agent: AgentLoop | None = None
        _steered = False

        try:
            cfg = self._apply_default_config(record.config)
            effective_model = cfg.model or self._parent_model
            effective_enabled_tools = self._resolve_child_enabled_tools(cfg)
            effective_tool_profile = None if effective_enabled_tools is not None else cfg.tool_profile
            child_session_manager = self._session_manager_for_record(record)

            child_agent = AgentLoop(
                workspace=self.workspace,
                model=effective_model,
                provider=cfg.provider or self._parent_provider,
                api_key=cfg.api_key or self._parent_api_key,
                base_url=cfg.base_url or self._parent_base_url,
                max_iterations=cfg.max_iterations,
                session_key=record.session_key,
                system_prompt=self._compose_system_prompt(record, cfg),
                enable_skills=self._resolve_effective_enable_skills(cfg),
                enabled_tools=effective_enabled_tools,
                tool_profile=effective_tool_profile,
                context_window=cfg.context_window,
                auto_commit=False,
                session_manager=child_session_manager,
                subagent_manager=self,
            )

            await child_agent.initialize()

            # Inject SubagentTool with reference back to this manager
            from spoon_bot.subagent.tools import SubagentTool
            spawn_tool = child_agent.tools.get("spawn")
            if spawn_tool and isinstance(spawn_tool, SubagentTool):
                spawn_tool.set_manager(self)
                spawn_tool._parent_agent_id = record.agent_id
                spawn_tool.set_spawner_context(
                    session_key=record.session_key,
                    channel=record.spawner_channel,
                    metadata=dict(record.spawner_metadata),
                    reply_to=record.spawner_reply_to,
                )

            # Run the task — with optional timeout and thinking
            if cfg.timeout_seconds:
                run_coro = self._run_process(child_agent, task_text, cfg)
                result_text = await asyncio.wait_for(
                    run_coro, timeout=float(cfg.timeout_seconds)
                )
            else:
                result_text = await self._run_process(child_agent, task_text, cfg)

            # Extract token usage if available
            token_usage = self._extract_token_usage(child_agent)

            await child_agent.cleanup()

            now = time.time()
            elapsed = round(now - (record.started_at or record.created_at), 2)

            self.registry.transition(
                record.agent_id,
                SubagentState.COMPLETED,
                result=result_text,
                completed_at=now,
                frozen_result_text=result_text,
                token_usage=token_usage,
            )
            self._persist_session_agent_state(record.agent_id)

            result_obj = SubagentResult(
                agent_id=record.agent_id,
                run_id=record.run_id,
                label=record.label,
                state=SubagentState.COMPLETED,
                result=result_text,
                elapsed_seconds=elapsed,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
                spawner_metadata=dict(record.spawner_metadata),
                spawner_reply_to=record.spawner_reply_to,
                model_name=record.model_name,
            )
            await self._results.put(result_obj)

            logger.info(
                f"Sub-agent {record.agent_id!r} run {record.run_id!r} "
                f"completed in {elapsed}s"
            )

            # Emit lifecycle event
            await self._emit_event(SubagentEvent(
                event_type="completed",
                agent_id=record.agent_id,
                label=record.label,
                parent_id=record.parent_id,
                depth=record.depth,
                model_name=record.model_name,
                result=result_text,
                elapsed_seconds=elapsed,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
                metadata={"run_id": record.run_id},
            ))

            # Push-based delivery: announce result to spawner via bus
            await self._announce_result(result_obj)

        except asyncio.TimeoutError:
            if child_agent:
                try:
                    await child_agent.cleanup()
                except Exception:
                    pass
            error_msg = f"Sub-agent timed out after {cfg.timeout_seconds}s"
            logger.warning(f"Sub-agent {record.agent_id!r}: {error_msg}")
            self.registry.transition(
                record.agent_id,
                SubagentState.FAILED,
                error=error_msg,
                completed_at=time.time(),
            )
            self._persist_session_agent_state(record.agent_id)
            result_obj = SubagentResult(
                agent_id=record.agent_id,
                run_id=record.run_id,
                label=record.label,
                state=SubagentState.FAILED,
                error=error_msg,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
                spawner_metadata=dict(record.spawner_metadata),
                spawner_reply_to=record.spawner_reply_to,
            )
            await self._results.put(result_obj)
            await self._announce_result(result_obj)
            await self._emit_event(SubagentEvent(
                event_type="failed",
                agent_id=record.agent_id,
                label=record.label,
                parent_id=record.parent_id,
                depth=record.depth,
                error=error_msg,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
                metadata={"run_id": record.run_id},
            ))

        except asyncio.CancelledError:
            if child_agent:
                try:
                    await child_agent.cleanup()
                except Exception:
                    pass

            # Check if this is a steer request (not a user cancellation)
            steer_request = self._steer_requests.pop(record.agent_id, None)
            if steer_request is not None:
                # Steer: restart with the new message, preserve session history
                _steered = True
                next_message = str(steer_request.get("message", "")).strip() or record.task
                next_run_id = str(steer_request.get("run_id", "")).strip() or self._new_run_id()
                self.registry.update_fields(
                    record.agent_id,
                    run_id=next_run_id,
                    task=next_message,
                    result=None,
                    error=None,
                    started_at=None,
                    completed_at=None,
                    token_usage=None,
                    frozen_result_text=None,
                )
                self._persist_session_agent_state(record.agent_id)
                logger.info(
                    f"Sub-agent {record.agent_id!r} steered to run {record.run_id!r}"
                )
                await self._emit_event(SubagentEvent(
                    event_type="steered",
                    agent_id=record.agent_id,
                    label=record.label,
                    parent_id=record.parent_id,
                    depth=record.depth,
                    model_name=record.model_name,
                    spawner_session_key=record.spawner_session_key,
                    spawner_channel=record.spawner_channel,
                    metadata={"run_id": record.run_id, "task": next_message},
                ))
                self._start_background_run(
                    record,
                    task_override=next_message,
                    is_restart=True,
                )
            else:
                # Regular cancellation
                self.registry.transition(
                    record.agent_id,
                    SubagentState.CANCELLED,
                    completed_at=time.time(),
                )
                self._persist_session_agent_state(record.agent_id)
                result_obj = SubagentResult(
                    agent_id=record.agent_id,
                    run_id=record.run_id,
                    label=record.label,
                    state=SubagentState.CANCELLED,
                    spawner_session_key=record.spawner_session_key,
                    spawner_channel=record.spawner_channel,
                    spawner_metadata=dict(record.spawner_metadata),
                    spawner_reply_to=record.spawner_reply_to,
                )
                await self._results.put(result_obj)
                logger.info(f"Sub-agent {record.agent_id!r} was cancelled")
                await self._emit_event(SubagentEvent(
                    event_type="cancelled",
                    agent_id=record.agent_id,
                    label=record.label,
                    parent_id=record.parent_id,
                    depth=record.depth,
                    spawner_session_key=record.spawner_session_key,
                    spawner_channel=record.spawner_channel,
                    metadata={"run_id": record.run_id},
                ))

        except Exception as exc:
            if child_agent:
                try:
                    await child_agent.cleanup()
                except Exception:
                    pass
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"Sub-agent {record.agent_id!r} failed: {error_msg}")
            self.registry.transition(
                record.agent_id,
                SubagentState.FAILED,
                error=error_msg,
                completed_at=time.time(),
            )
            self._persist_session_agent_state(record.agent_id)
            result_obj = SubagentResult(
                agent_id=record.agent_id,
                run_id=record.run_id,
                label=record.label,
                state=SubagentState.FAILED,
                error=error_msg,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
                spawner_metadata=dict(record.spawner_metadata),
                spawner_reply_to=record.spawner_reply_to,
            )
            await self._results.put(result_obj)
            await self._announce_result(result_obj)
            await self._emit_event(SubagentEvent(
                event_type="failed",
                agent_id=record.agent_id,
                label=record.label,
                parent_id=record.parent_id,
                depth=record.depth,
                error=error_msg,
                spawner_session_key=record.spawner_session_key,
                spawner_channel=record.spawner_channel,
                metadata={"run_id": record.run_id},
            ))

        finally:
            if not _steered:
                self._tasks.pop(record.agent_id, None)

    @staticmethod
    async def _run_process(
        child_agent: Any,
        task_text: str,
        cfg: SubagentConfig,
    ) -> str:
        """Run the child agent process with optional extended thinking."""
        thinking_level = normalize_thinking_level(cfg.thinking_level)
        if thinking_level and thinking_level != "off":
            try:
                result_text, _ = await child_agent.process_with_thinking(
                    task_text,
                    thinking_level=thinking_level,
                )
            except TypeError as exc:
                if "thinking" not in str(exc):
                    raise
                logger.warning(
                    "Child agent runtime does not support the requested thinking mode; "
                    "falling back to standard processing."
                )
                result_text = await child_agent.process(task_text)
        else:
            result_text = await child_agent.process(task_text)
        return result_text or ""

    @staticmethod
    def _extract_token_usage(child_agent: Any) -> Optional[TokenUsage]:
        """Extract token usage from a child AgentLoop if available."""
        try:
            usage = child_agent.get_usage()
            if not usage:
                return None
            return TokenUsage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                cache_read_tokens=usage.get("cache_read_tokens", 0),
                cache_write_tokens=usage.get("cache_write_tokens", 0),
            )
        except Exception:
            return None

    async def _announce_result(self, result: SubagentResult) -> None:
        """Push result to spawner channel via bus injection (with retry/backoff).

        Falls back silently if no bus is set or channel cannot be reached.
        """
        if not self._bus:
            return
        if not result.spawner_session_key or not result.spawner_channel:
            return

        # Build wake message
        state = result.state.value
        content_raw = result.result or result.error or "(no output)"
        content = content_raw[:_WAKE_RESULT_TRUNCATE]
        if len(content_raw) > _WAKE_RESULT_TRUNCATE:
            content += f"\n... [{len(content_raw) - _WAKE_RESULT_TRUNCATE} chars omitted]"

        elapsed_str = (
            f" in {result.elapsed_seconds}s" if result.elapsed_seconds else ""
        )
        wake_content = (
            f"[Sub-agent Completed] '{result.label}' ({result.agent_id}) "
            f"has {state}{elapsed_str}.\n"
            f"task_id: {result.agent_id}\n\n"
            f"run_id: {result.run_id}\n\n"
            f"{content}"
        )

        from spoon_bot.bus.events import InboundMessage
        wake_metadata = dict(result.spawner_metadata)
        wake_metadata.update({
            "is_subagent_wake": True,
            "subagent_id": result.agent_id,
            "subagent_run_id": result.run_id,
        })
        if result.spawner_reply_to is not None:
            wake_metadata.setdefault("reply_to", result.spawner_reply_to)
        wake_msg = InboundMessage(
            content=wake_content,
            channel=result.spawner_channel,
            session_key=result.spawner_session_key,
            sender_id="subagent_system",
            metadata=wake_metadata,
        )

        for attempt in range(_ANNOUNCE_MAX_RETRIES):
            try:
                published = await self._bus.publish(wake_msg)
                if published:
                    logger.info(
                        f"Wake announced for sub-agent {result.agent_id!r} "
                        f"→ {result.spawner_channel} / {result.spawner_session_key}"
                    )
                    # Clear frozen result after announce if cleanup mode is DELETE
                    record = self.registry.get(result.agent_id)
                    if record and record.cleanup == CleanupMode.DELETE:
                        self.registry.update_fields(
                            result.agent_id,
                            frozen_result_text=None,
                        )
                    return
                # Queue full — retry after backoff
                delay = _ANNOUNCE_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"Bus queue full, retrying announce for {result.agent_id!r} "
                    f"in {delay}s (attempt {attempt + 1}/{_ANNOUNCE_MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
            except Exception as exc:
                logger.error(
                    f"Announce failed for {result.agent_id!r} "
                    f"(attempt {attempt + 1}): {exc}"
                )
                if attempt < _ANNOUNCE_MAX_RETRIES - 1:
                    await asyncio.sleep(_ANNOUNCE_BASE_DELAY * (2 ** attempt))

        logger.error(
            f"Failed to announce sub-agent {result.agent_id!r} result "
            f"after {_ANNOUNCE_MAX_RETRIES} attempts."
        )

    async def _emit_event(self, event: SubagentEvent) -> None:
        """Notify all registered lifecycle event listeners."""
        for listener in self._event_listeners:
            try:
                result = listener(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.debug(f"Subagent event listener error: {exc}")

    def _compose_system_prompt(
        self,
        record: SubagentRecord,
        cfg: SubagentConfig | None = None,
    ) -> str:
        """Stack caller-provided prompt text above the runtime subagent contract."""
        runtime_prompt = self._build_system_prompt(record).strip()
        effective_cfg = cfg or record.config
        custom_prompt = (effective_cfg.system_prompt or "").strip()
        if not custom_prompt:
            return runtime_prompt
        return f"{custom_prompt}\n\n{runtime_prompt}"

    def _build_system_prompt(self, record: SubagentRecord) -> str:
        """Build a focused system prompt for the sub-agent."""
        can_spawn_children = (
            self._config_allows_nested_subagents(record.config)
            and record.depth < self.max_depth
        )
        lines = [
            "# Subagent Context",
            "",
            f"You are a sub-agent session `{record.agent_id}` running execution `{record.run_id}`.",
            f"Depth: {record.depth}/{self.max_depth}",
            f"Task: {record.task}",
            "",
            "## Rules",
            "1. Stay focused on this task only.",
            "2. Return one concise final summary with concrete findings or changes.",
            "3. Your result is auto-announced back to the requester; do not talk to the user directly.",
            "4. Do not busy-poll for child status or send heartbeat-style updates.",
            "5. Re-read only targeted context when prior tool output is truncated or compacted.",
        ]
        if record.config.specialization:
            lines.extend([
                "",
                "## Specialization",
                record.config.specialization,
                "Treat matching requests as your primary responsibility.",
            ])
        if record.config.routing_mode == RoutingMode.ORCHESTRATED:
            lines.extend([
                "",
                "## Routing Mode",
                "This specialist is configured for orchestrated routing.",
                "When a matching request arrives, you may coordinate nested workers and then synthesize the final result.",
            ])
        if can_spawn_children:
            lines.extend([
                "",
                "## Orchestration",
                "You are explicitly allowed to spawn nested sub-agents for parallel work.",
                "Use the `spawn` tool only when delegation materially helps.",
                "Child results are pushed back automatically; wait for expected completions before finalizing.",
            ])
        else:
            lines.extend([
                "",
                "## Execution Mode",
                "You are a leaf worker for this run.",
                "Do not spawn further sub-agents from this session.",
            ])
        return "\n".join(lines)
