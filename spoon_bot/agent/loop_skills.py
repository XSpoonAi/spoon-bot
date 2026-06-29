"""Skill discovery and prompt helpers for AgentLoop.

ponytail: no service layer, just moved methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from spoon_bot.agent.context import format_current_datetime_context
from spoon_bot.agent.execution_ledger import (
    load_recent_execution_ledger_context,
)
from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.request_hints import (
    build_request_execution_hints,
    format_current_session_fact_check_context,
    format_exact_shell_command_context,
    format_explicit_tool_request_context,
    format_explicit_request_urls_context,
    format_explicit_request_values_context,
    request_is_bare_continuation,
    request_is_plain_continuation_only,
)

if TYPE_CHECKING:
    pass

AgentLoop: Any = None

from spoon_bot.agent.loop_state import _MISSING

_EXTERNAL_SIDE_EFFECT_BOUNDARY = (
    "[EXTERNAL SIDE-EFFECT BOUNDARY]: For external systems, account/wallet "
    "state, remote jobs, approvals, registrations, entries, submissions, "
    "trades, or any action that spends credits/tokens/funds or changes remote "
    "state, execute only the side-effect sequence that the newest user request "
    "or active SKILL.md contract actually selected. If the newest request gives "
    "an ordered workflow, complete that workflow in order until its first real "
    "terminal outcome or concrete blocker. Do not branch into alternative "
    "external actions, restart the same completed side effect, or repeat a paid "
    "submission just because tools remain available. Do not batch multiple "
    "alternative tool attempts in one assistant step; run one attempt, inspect "
    "its result, and then either answer with the blocker or proceed only when "
    "the newest request or active SKILL.md contract selects the next attempt. "
    "If the user gives an exact command for "
    "a replay, simulation, dry-run, or no-op check, run that command exactly "
    "as written; never remove protective wrappers such as echo/printf or "
    "dry-run/no-op flags, and never convert a simulated command into a live "
    "side-effecting command.\n"
)
_EXACT_COMMAND_BOUNDARY = (
    "[EXACT COMMAND BOUNDARY]: If the newest user request provides exact shell "
    "commands to run, run those commands exactly, in order, unless a command is "
    "unsafe or cannot be executed. Do not replace an exact requested command with "
    "a nearby or suggested command. If an exact command fails, report that command's "
    "failure directly instead of switching to a different command.\n"
)
_USER_FACING_OUTPUT_BOUNDARY = (
    "[USER-FACING OUTPUT BOUNDARY]: Final answers are for the user, not for the "
    "runtime. Match the newest user's natural language unless the user explicitly "
    "requested another language or a raw machine-readable format. A short "
    "continuation message still defines the output language; do not inherit the "
    "language of older requests or mostly-English tool evidence. Convert tool "
    "JSON, tool arguments, command transcripts, and structured outputs into concise "
    "human-readable results. Do not paste raw JSON/tool transcripts/internal "
    "planning unless the newest user explicitly asks for raw logs or JSON.\n"
)
_BOUNDED_CONTINUATION_BOUNDARY = (
    "[BOUNDED CONTINUATION REQUEST]: When the newest user message is only a "
    "short continuation of earlier work, resume at most one verifier-visible "
    "unit only if same-session evidence shows a single unfinished workflow and "
    "a clear next documented step. Use prior request facts as checkpoints, not "
    "completion evidence or renewed permission for older counts/repetition "
    "targets. Prior user request text is intentionally not repeated for plain "
    "continuations; use assistant/tool state evidence and live state instead. "
    "If the evidence is completed, stale, read-only, or ambiguous, ask a "
    "concise clarification or report current status instead of starting a new "
    "external side effect.\n"
)


class LoopSkillsMixin:
    def _workspace_posix_path(self) -> str:
        """Return the workspace path in POSIX form for shell commands."""
        import re as _re
        import sys

        raw = str(self.workspace).replace("\\", "/")
        if sys.platform == "win32":
            raw = _re.sub(r"^([A-Za-z]):", lambda m: f"/{m.group(1).lower()}", raw)
        return raw

    # Directories in the workspace root that are never skills
    _WORKSPACE_INFRA_DIRS = frozenset(
        {
            "skills",
            "logs",
            "sessions",
            "memory",
            "channels",
            "wallet",
            ".git",
        }
    )

    # ------------------------------------------------------------------
    # Conditional activation helpers
    # ------------------------------------------------------------------

    def record_touched_paths(self, *paths: str | Path) -> None:
        """Register file paths the agent has interacted with.

        Called by file-oriented tools (read_file, write_file, edit_file, etc.)
        so that path-conditional skills can be activated dynamically.

        Also performs **dynamic discovery**: walks up from the touched path
        looking for directories containing ``SKILL.md`` and adds them to
        ``_skill_paths`` if not already known (inspired by Claude Code's
        ``discoverSkillDirsForPaths``).
        """
        for p in paths:
            resolved = Path(p)
            if not resolved.is_absolute():
                resolved = self.workspace / resolved
            try:
                resolved = resolved.resolve()
            except OSError:
                pass

            try:
                rel = str(resolved.relative_to(self.workspace))
            except (ValueError, OSError):
                rel = str(p)
            self._touched_paths.add(rel.replace("\\", "/"))

            self._discover_skills_near(resolved)

    def _discover_skills_near(self, file_path: Path) -> None:
        """Walk up from *file_path* looking for ``skills/`` directories.

        When a directory containing ``SKILL.md`` files is found and is not
        already in ``_skill_paths``, it is added dynamically.  This mirrors
        Claude Code's ``discoverSkillDirsForPaths`` which walks up from every
        file operation to find project-level ``.claude/skills/`` directories.

        Only walks up to ``self.workspace`` (never above).
        """
        known = {str(Path(sp).resolve()) for sp in self._skill_paths}
        current = file_path.parent if file_path.is_file() else file_path

        try:
            ws_resolved = self.workspace.resolve()
        except OSError:
            return

        while True:
            try:
                if not current.is_relative_to(ws_resolved):
                    break
            except (TypeError, ValueError):
                break

            skills_dir = current / "skills"
            if skills_dir.is_dir() and str(skills_dir.resolve()) not in known:
                has_skill = any(
                    (child / "SKILL.md").exists()
                    for child in skills_dir.iterdir()
                    if child.is_dir()
                )
                if has_skill:
                    self._skill_paths.append(skills_dir)
                    known.add(str(skills_dir.resolve()))
                    logger.info(f"Dynamic skill discovery: found {skills_dir}")

            if current == ws_resolved:
                break
            current = current.parent

    @staticmethod
    def _skill_paths_match(
        patterns: list[str],
        touched: set[str],
        workspace: Path | None = None,
    ) -> bool:
        """Return True if any *touched* file matches at least one *pattern*.

        Patterns use gitignore-style globs (``fnmatch``).  A leading ``!``
        negates (exclude) the pattern - same semantics as ``.gitignore``.
        An empty *patterns* list means "always active" (unconditional).
        """
        from fnmatch import fnmatch

        if not patterns:
            return True
        if not touched:
            return False

        for fp in touched:
            included = False
            for pat in patterns:
                negate = pat.startswith("!")
                glob = pat.lstrip("!")
                if fnmatch(fp, glob) or fnmatch(fp, f"**/{glob}"):
                    included = not negate
            if included:
                return True
        return False

    @staticmethod
    def _parse_skill_frontmatter(skill_md: Path) -> dict[str, str | list[str]]:
        """Extract description, when_to_use, triggers, and paths from SKILL.md YAML frontmatter.

        The ``paths`` field (list of gitignore-style glob patterns) enables
        conditional activation: skills declaring ``paths`` are only active when
        recently-touched files match at least one pattern.  Skills without
        ``paths`` are unconditionally active.
        """
        import re as _re

        result: dict[str, str | list[str]] = {
            "description": "",
            "when_to_use": "",
            "triggers": "",
            "paths": [],
        }
        try:
            raw = skill_md.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return result

        fm = _re.match(r"^---\s*\n(.*?)\n---", raw, _re.DOTALL)
        if not fm:
            for line in raw.split("\n")[1:20]:
                stripped = line.strip()
                if stripped and not stripped.startswith(("#", "---", "```")):
                    result["description"] = stripped[:300]
                    break
            return result

        fm_text = fm.group(1)
        try:
            import yaml

            parsed = yaml.safe_load(fm_text) or {}
        except Exception:
            parsed = {}

        if isinstance(parsed, dict):
            description = parsed.get("description")
            when_to_use = parsed.get("when_to_use", parsed.get("whenToUse", ""))
            paths = parsed.get("paths")
            triggers = parsed.get("triggers")

            if isinstance(description, str):
                result["description"] = description
            if isinstance(when_to_use, str):
                result["when_to_use"] = when_to_use
            if isinstance(paths, list):
                result["paths"] = [str(path) for path in paths if str(path).strip()]

            trigger_fragments: list[str] = []

            def _collect_strings(value: object) -> None:
                if isinstance(value, str):
                    trigger_fragments.append(value)
                elif isinstance(value, list):
                    for item in value:
                        _collect_strings(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        _collect_strings(item)

            _collect_strings(triggers)
            result["triggers"] = "|".join(trigger_fragments)

            if result["description"]:
                result["description"] = str(result["description"]).strip()[:300]
                result["when_to_use"] = str(result["when_to_use"]).strip()[:200]
                return result

        in_multiline = ""
        in_paths_list = False
        trigger_fragments: list[str] = []
        paths_list: list[str] = []

        for line in fm_text.split("\n"):
            is_top_level = bool(line.strip()) and not line.startswith((" ", "\t"))
            stripped = line.strip()

            if in_paths_list:
                if stripped.startswith("- "):
                    paths_list.append(stripped[2:].strip().strip("'\""))
                    continue
                elif line.startswith("  ") or line.startswith("\t"):
                    if stripped.startswith("- "):
                        paths_list.append(stripped[2:].strip().strip("'\""))
                    continue
                else:
                    in_paths_list = False

            if in_multiline:
                if line.startswith("  ") or line.startswith("\t"):
                    if in_multiline == "description":
                        result["description"] += " " + stripped
                    elif in_multiline == "when_to_use":
                        result["when_to_use"] += " " + stripped
                    continue
                else:
                    in_multiline = ""

            if is_top_level and stripped.startswith("description:"):
                val = stripped.split(":", 1)[1].strip().strip("'\"")
                if val and val not in (">", "|"):
                    result["description"] = val
                elif val in (">", "|"):
                    in_multiline = "description"

            elif is_top_level and stripped.startswith(("when_to_use:", "whenToUse:")):
                val = stripped.split(":", 1)[1].strip().strip("'\"")
                if val and val not in (">", "|"):
                    result["when_to_use"] = val
                elif val in (">", "|"):
                    in_multiline = "when_to_use"

            elif is_top_level and stripped.startswith("paths:"):
                inline = stripped.split(":", 1)[1].strip()
                if inline.startswith("[") and inline.endswith("]"):
                    for p in inline[1:-1].split(","):
                        p = p.strip().strip("'\"")
                        if p:
                            paths_list.append(p)
                else:
                    in_paths_list = True

            elif is_top_level and ("triggers" in stripped.lower() or "trigger" in stripped.lower()):
                trigger_fragments.extend(_re.findall(r'"([^"]+)"', stripped))

        if not result["description"]:
            for line in raw.split("\n")[1:20]:
                stripped = line.strip()
                if stripped and not stripped.startswith(("#", "---", "```")):
                    result["description"] = stripped[:300]
                    break

        result["description"] = str(result["description"]).strip()[:300]
        result["when_to_use"] = str(result["when_to_use"]).strip()[:200]
        result["triggers"] = "|".join(trigger_fragments)
        result["paths"] = paths_list
        return result

    @staticmethod
    def _has_valid_skill_frontmatter(skill_md: Path) -> bool:
        """Return True when a SKILL.md starts with a YAML frontmatter block."""
        try:
            lines = skill_md.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return False
        if not lines or lines[0].strip() != "---":
            return False
        return any(line.strip() == "---" for line in lines[1:80])

    def _skill_manager_discovery_paths(self) -> list[Path]:
        """Return valid skill directories for the spoon-core SkillManager.

        spoon-core logs an error for every invalid SKILL.md it sees.  The local
        prompt builder can tolerate stale user folders, but the runtime manager
        should only receive directories that contain parseable skill metadata.
        """
        paths: list[Path] = []
        seen: set[str] = set()

        def add(path: Path) -> None:
            try:
                resolved = path.resolve()
            except OSError:
                return
            key = str(resolved)
            if key in seen:
                return
            seen.add(key)
            paths.append(resolved)

        for base in getattr(self, "_skill_paths", []):
            root = Path(base)
            if not root.is_dir():
                continue
            direct_skill_md = root / "SKILL.md"
            if direct_skill_md.exists():
                if self._has_valid_skill_frontmatter(direct_skill_md):
                    add(root)
                continue
            for child in root.iterdir():
                if not child.is_dir():
                    continue
                skill_md = child / "SKILL.md"
                if self._has_valid_skill_frontmatter(skill_md):
                    add(child)

        return paths

    def _iter_skill_candidates(
        self,
        *,
        include_dormant: bool = False,
    ) -> list[tuple[str, Path, Path, bool]]:
        """Return (name, dir, skill_md_path, is_organized) for all discoverable skills.

        Scans in priority order:
        1. ``workspace/skills/`` - primary organized location
        2. All entries in ``_skill_paths`` (includes bundled/dev skills)
        3. Workspace root - unorganized skills the user may have dropped in

        Deduplicates by both name AND resolved (realpath) canonical path so
        symlinks pointing to the same physical directory are counted once.
        Earlier entries take priority (matches Claude Code's first-wins logic).

        **Conditional activation (``paths`` frontmatter)**:
        Skills declaring ``paths`` patterns are dormant until a touched file
        matches.  Set *include_dormant* to ``True`` to include them anyway
        (useful for listing all skills in the prompt with a "dormant" tag).
        """
        skills_dir = self.workspace / "skills"

        # (parent_dir, is_organized)
        scan_dirs: list[tuple[Path, bool]] = []
        if skills_dir.is_dir():
            scan_dirs.append((skills_dir, True))
        for sp in getattr(self, "_skill_paths", []):
            resolved = Path(sp).resolve()
            if resolved.is_dir() and resolved != skills_dir.resolve():
                scan_dirs.append((resolved, True))
        if self.workspace.is_dir():
            scan_dirs.append((self.workspace, False))

        candidates: list[tuple[str, Path, Path, bool]] = []
        seen_names: set[str] = set()
        seen_realpaths: set[str] = set()

        for parent_dir, is_organized in scan_dirs:
            for child in sorted(parent_dir.iterdir()):
                if not (child.is_dir() or child.is_symlink()):
                    continue
                if not is_organized and child.name in self._WORKSPACE_INFRA_DIRS:
                    continue
                skill_md = child / "SKILL.md"
                if not skill_md.exists():
                    continue
                name = child.name
                if name in seen_names:
                    continue

                try:
                    canonical = str(skill_md.resolve())
                except OSError:
                    canonical = str(skill_md)
                if canonical in seen_realpaths:
                    logger.debug(
                        f"Skipping duplicate skill '{name}' "
                        f"(same file already loaded via another path)"
                    )
                    continue

                if not include_dormant:
                    fm = self._parse_skill_frontmatter(skill_md)
                    skill_paths = fm.get("paths", [])
                    if isinstance(skill_paths, list) and skill_paths:
                        if not self._skill_paths_match(
                            skill_paths,
                            self._touched_paths,
                            self.workspace,
                        ):
                            logger.debug(f"Skill '{name}' dormant (paths not matched)")
                            continue

                seen_names.add(name)
                seen_realpaths.add(canonical)
                candidates.append((name, child, skill_md, is_organized))

        return candidates

    def _build_skill_context(
        self,
        skill_name: str,
        skill_dir: Path,
        *,
        is_organized: bool,
    ) -> dict[str, Any]:
        """Build sanitized skill metadata suitable for session persistence."""
        import re as _re
        import sys as _sys

        base_dir = str(skill_dir).replace("\\", "/")
        if _sys.platform == "win32":
            base_dir = _re.sub(r"^([A-Za-z]):", lambda m: f"/{m.group(1).lower()}", base_dir)

        skill_rel = f"skills/{skill_name}" if is_organized else skill_name
        skill_md = skill_dir / "SKILL.md"
        return {
            "name": skill_name,
            "base_dir": base_dir,
            "workspace_relative_path": f"{skill_rel}/",
            "location": self._skill_prompt_location(
                skill_name,
                skill_dir,
                skill_md,
                is_organized=is_organized,
            ),
            "organized": bool(is_organized),
        }

    def _skill_prompt_location(
        self,
        skill_name: str,
        skill_dir: Path,
        skill_md: Path,
        *,
        is_organized: bool,
    ) -> str:
        """Return a model-readable SKILL.md location that actually exists.

        Skills under the active runtime ``workspace/skills`` keep the familiar
        relative form. Skills discovered from configured or bundled/dev paths
        must use their real path; otherwise the model is shown a relative path
        that points at a nonexistent runtime skill.
        """
        try:
            workspace_skills = (self.workspace / "skills").resolve()
            resolved_dir = skill_dir.resolve()
            resolved_md = skill_md.resolve()
            if is_organized and (
                resolved_dir == workspace_skills or workspace_skills in resolved_dir.parents
            ):
                return f"skills/{skill_name}/SKILL.md"
            try:
                return str(resolved_md.relative_to(self.workspace)).replace("\\", "/")
            except (OSError, ValueError):
                return str(resolved_md).replace("\\", "/")
        except Exception:
            return f"skills/{skill_name}/SKILL.md" if is_organized else f"{skill_name}/SKILL.md"

    def _resolve_skill_context_by_name(self, skill_name: str) -> dict[str, Any] | None:
        """Resolve a persisted skill name against the current skill catalog."""
        try:
            candidates = self._iter_skill_candidates(include_dormant=True)
        except Exception:
            return None

        for name, skill_dir, _skill_md, is_organized in candidates:
            if name == skill_name:
                return self._build_skill_context(
                    name,
                    skill_dir,
                    is_organized=is_organized,
                )
        return None

    def _build_request_context_sections(self, message: str) -> str:
        """Build reusable request-derived context without changing user intent."""
        hint_source = self._request_hint_source_text(message)
        hints = self._build_request_execution_hints_from_text(hint_source)
        sections = [
            _BOUNDED_CONTINUATION_BOUNDARY
            if request_is_plain_continuation_only(message)
            else "",
            self._format_continuation_anchor_context(message),
            format_explicit_request_urls_context(hint_source),
            format_explicit_request_values_context(hint_source),
            self._format_local_skill_execution_context(hints),
            self._format_recent_execution_ledger_context(),
            format_explicit_tool_request_context(hints),
            format_current_session_fact_check_context(message),
            format_exact_shell_command_context(message),
        ]
        return "".join(section for section in sections if section)

    def _format_continuation_anchor_context(self, message: str) -> str:
        """Format the selected prior user request for short continuations."""
        current = str(message or "")
        if not request_is_bare_continuation(current):
            return ""
        previous = AgentLoop._previous_user_request_for_continuation(self, current)
        plain_continuation = request_is_plain_continuation_only(current)
        lines = [
            "[CONTINUATION ANCHOR]: The newest user message is a short "
            "continuation, so it can use the nearest prior task only as a "
            "state anchor.",
        ]
        if not previous:
            lines.append(
                "No prior real user request is available in this session. Ask a "
                "concise clarification instead of selecting an older task "
                "from memory, ledger, or tool output."
            )
        elif plain_continuation:
            lines.append(
                "The newest message adds no new count, target, or scope. "
                "The nearest prior user request text is intentionally omitted "
                "so older counts, batches, or repeated-action targets are not "
                "renewed as current permission. Use recent assistant/tool "
                "evidence only to locate an immediate unfinished checkpoint. "
                "After at most one bounded action or status check, report the "
                "current state or ask for explicit scope."
            )
        else:
            lines.extend(
                [
                    "Selected prior user request:",
                    f"- {AgentLoop._compress_message_content(previous.strip(), 900)}",
                ]
            )
            lines.append(
                "Use this selected request as the primary task scope for "
                "task/tool choice. Treat older session compact and "
                "execution-ledger facts as evidence only; do not switch to "
                "another earlier task unless the newest user message "
                "explicitly names it. If live state shows the selected task "
                "is complete, blocked, or ambiguous, report that status or "
                "ask a concise clarification rather than starting a different "
                "external side effect."
            )
        lines.append("")
        return "\n".join(lines)

    def _format_recent_execution_ledger_context(self) -> str:
        try:
            return load_recent_execution_ledger_context(
                workspace=getattr(self, "workspace", None),
                owner=self._current_tool_owner_key(),
            )
        except Exception as exc:
            logger.debug(f"Recent execution ledger context skipped: {exc}")
            return ""

    @staticmethod
    def _format_local_skill_execution_context(hints: dict[str, Any]) -> str:
        """Format installed skill command hints without selecting a route."""
        skills = hints.get("local_executable_skills") if isinstance(hints, dict) else None
        if not isinstance(skills, list) or not skills:
            return ""
        lines = ["[LOCAL SKILL EXECUTION CONTEXT]:"]
        lines.append(
            "Installed skills below expose documented commands. Treat prior "
            "request facts as checkpoints, not completion. When continuing a "
            "skill workflow, use SKILL.md and latest tool evidence to choose "
            "the next documented command; do not infer completion from a plan."
        )
        for skill in skills[:4]:
            if not isinstance(skill, dict):
                continue
            name = str(skill.get("name") or "").strip()
            commands = skill.get("commands")
            if name:
                lines.append(f"- {name}:")
            if isinstance(commands, list):
                for command in commands[:8]:
                    command_text = str(command or "").strip()
                    if command_text:
                        lines.append(f"  - {command_text}")
        if len(lines) <= 2:
            return ""
        lines.append("")
        return "\n".join(lines)

    def _build_step_prompt(self, message: str) -> str:
        """Build a minimal per-step prompt from the user's request.

        Keeps only the user's original request and workspace path.
        Injects env vars so they survive short-term memory pruning.
        """
        _truncated = self._truncate_request_for_prompt(message)
        _ws = self._workspace_posix_path()
        prompt = (
            "[TURN PRIORITY]: Execute only the newest user request. "
            "Any unfinished plan, stale tool sequence, or previous task assumption is superseded "
            "unless the newest user message explicitly says to continue it.\n"
            "[HISTORY BOUNDARY]: Prior conversation is reference only. Do not run prior tasks, "
            "do not append prior-task work, and stop as soon as the newest request is satisfied.\n"
            "[HISTORY VERIFICATION]: If the newest request asks what happened earlier, "
            "mentions previous actions/results, or challenges your prior answer, verify exact "
            "current-session user/tool facts first. Use search_history(mode='recent', "
            "scope='current') for last completed action/result questions, or targeted "
            "search_history(scope='current') when you already have a stable id/path/hash. "
            "If exact evidence is absent, say what is absent instead of filling the gap "
            "from memory or inference.\n"
            f"{_EXTERNAL_SIDE_EFFECT_BOUNDARY}"
            f"{_EXACT_COMMAND_BOUNDARY}"
            f"{_USER_FACING_OUTPUT_BOUNDARY}"
            f"{format_current_datetime_context(bracketed=True)}\n"
            f"[USER REQUEST]: {_truncated}\n"
            f"{self._build_request_context_sections(message)}"
            f"[WORKSPACE]: {_ws}/\n\n" + self.DEFAULT_NEXT_STEP_PROMPT
        )
        skill_zip_context = self._current_turn_skill_zip_context()
        if skill_zip_context:
            prompt = f"{skill_zip_context}\n{prompt}"
        recent_turn_notice = getattr(self, "_recent_turn_notice", None)
        if isinstance(recent_turn_notice, str) and recent_turn_notice.strip():
            prompt = f"[PREVIOUS TURN STATUS]: {recent_turn_notice.strip()}\n" + prompt
        session_recall = self._build_session_recall_context(message)
        if isinstance(session_recall, str) and session_recall.strip():
            prompt = f"{session_recall.strip()}\n" + prompt
        env_section = self._extract_env_for_prompt()
        if env_section:
            prompt += env_section
        return prompt

    def _build_request_context_prompt(self, message: str) -> str:
        """Build the compact request context block used for thinking runs."""
        _truncated = self._truncate_request_for_prompt(message)
        _ws = self._workspace_posix_path()
        prompt = (
            "[TURN PRIORITY]: Execute only the newest user request. "
            "Any unfinished plan, stale tool sequence, or previous task assumption is superseded "
            "unless the newest user message explicitly says to continue it.\n"
            "[HISTORY BOUNDARY]: Prior conversation is reference only. Do not run prior tasks, "
            "do not append prior-task work, and stop as soon as the newest request is satisfied.\n"
            "[HISTORY VERIFICATION]: If the newest request asks what happened earlier, "
            "mentions previous actions/results, or challenges your prior answer, verify exact "
            "current-session user/tool facts first. Use search_history(mode='recent', "
            "scope='current') for last completed action/result questions, or targeted "
            "search_history(scope='current') when you already have a stable id/path/hash. "
            "If exact evidence is absent, say what is absent instead of filling the gap "
            "from memory or inference.\n"
            f"{_EXTERNAL_SIDE_EFFECT_BOUNDARY}"
            f"{_EXACT_COMMAND_BOUNDARY}"
            f"{_USER_FACING_OUTPUT_BOUNDARY}"
            f"{format_current_datetime_context(bracketed=True)}\n"
            f"[USER REQUEST]: {_truncated}\n"
            f"{self._build_request_context_sections(message)}"
            f"[WORKSPACE]: {_ws}/\n"
        )
        skill_zip_context = self._current_turn_skill_zip_context()
        if skill_zip_context:
            prompt = f"{skill_zip_context}\n{prompt}"
        recent_turn_notice = getattr(self, "_recent_turn_notice", None)
        if isinstance(recent_turn_notice, str) and recent_turn_notice.strip():
            prompt = f"[PREVIOUS TURN STATUS]: {recent_turn_notice.strip()}\n" + prompt
        session_recall = self._build_session_recall_context(message)
        if isinstance(session_recall, str) and session_recall.strip():
            prompt = f"{session_recall.strip()}\n" + prompt
        env_section = self._extract_env_for_prompt()
        if env_section:
            prompt += env_section
        return prompt

    def _collect_request_skill_candidates(self) -> list[dict[str, Any]]:
        """Collect installed skill metadata for request-scoped hinting."""
        skill_candidates: list[dict[str, Any]] = []
        try:
            candidates = self._iter_skill_candidates(include_dormant=True)
        except Exception:
            candidates = []
        for name, _skill_dir, skill_md, is_organized in candidates:
            fm = self._parse_skill_frontmatter(skill_md)
            skill_candidates.append(
                {
                    "name": name,
                    "skill_md": skill_md,
                    "is_organized": is_organized,
                    "description": fm.get("description") or "",
                    "when_to_use": fm.get("when_to_use") or "",
                }
            )
        return skill_candidates

    def _collect_available_tool_identifiers(self) -> list[str]:
        """Collect registered tool and MCP identifiers for request-scoped hints."""
        names: set[str] = set()

        registry_tools = getattr(getattr(self, "tools", None), "_tools", {})
        if isinstance(registry_tools, dict):
            names.update(str(name) for name in registry_tools if str(name).strip())
            for tool in registry_tools.values():
                tool_name = getattr(tool, "name", None)
                if tool_name:
                    names.add(str(tool_name))

        manager_tools = getattr(
            getattr(getattr(self, "_agent", None), "available_tools", None),
            "tool_map",
            {},
        )
        if isinstance(manager_tools, dict):
            names.update(str(name) for name in manager_tools if str(name).strip())
            for tool in manager_tools.values():
                tool_name = getattr(tool, "name", None)
                if tool_name:
                    names.add(str(tool_name))

        try:
            for entry in self.get_mcp_catalog():
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if name:
                    names.add(str(name))
                tools = entry.get("tools")
                if isinstance(tools, list):
                    names.update(str(tool) for tool in tools if str(tool).strip())
        except Exception:
            pass

        return sorted(names)

    def _request_hint_source_text(self, message: str) -> str:
        """Return text used for routing-neutral request facts."""
        current = str(message or "")
        if not request_is_bare_continuation(current):
            return current
        if request_is_plain_continuation_only(current):
            return current
        previous = AgentLoop._previous_user_request_for_continuation(self, current)
        if not previous:
            return current
        return f"{previous.rstrip()}\n{current.strip()}"

    @staticmethod
    def _previous_user_request_for_continuation(self: Any, current_message: str) -> str:
        """Return the nearest prior real user request for a short continuation."""
        session = getattr(self, "_session", None)
        messages = []
        if session is not None and hasattr(session, "get_messages"):
            try:
                messages = session.get_messages()
            except Exception:
                messages = []
        if not isinstance(messages, list):
            messages = []
        current_norm = AgentLoop._normalize_comparable_text(current_message)
        for item in reversed(messages):
            if not isinstance(item, dict):
                continue
            if str(item.get("role") or "").strip().lower() != "user":
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if AgentLoop._normalize_comparable_text(content) == current_norm:
                continue
            return content
        return ""

    def _build_request_execution_hints_from_text(self, message: str) -> dict[str, Any]:
        """Build request hints from a chosen text source and current skill catalog."""
        return build_request_execution_hints(
            message,
            self._collect_request_skill_candidates(),
            self._collect_available_tool_identifiers(),
        )

    def _build_request_execution_hints(self, message: str) -> dict[str, Any]:
        """Build request hints from the newest message and current skill catalog."""
        hints = self._build_request_execution_hints_from_text(
            self._request_hint_source_text(message),
        )
        hints["bare_continuation"] = request_is_bare_continuation(message)
        hints["plain_continuation"] = request_is_plain_continuation_only(message)
        AgentLoop._configure_request_scoped_history_tool(self, hints)
        AgentLoop._bind_request_execution_hints_to_tools(self, hints)
        return hints

    def _configure_request_scoped_history_tool(self, hints: dict[str, Any]) -> None:
        """Expose history search only for turns that need earlier transcript facts."""
        if not hasattr(self, "tools"):
            return
        try:
            # Keep the read-only history tool available so the model can obey
            # the generic history-verification rule without language-specific
            # intent detection. The tool itself is scoped and budgeted.
            self.add_tool("search_history")
        except Exception as exc:
            logger.debug(f"Failed to adjust request-scoped history tool: {exc}")
        try:
            requests = hints.get("explicit_tool_requests") if isinstance(hints, dict) else None
            requested_names = {
                "".join(ch for ch in str(request.get("name") or "").casefold() if ch.isalnum())
                for request in (requests or [])
                if isinstance(request, dict) and bool(request.get("available"))
            }
            if requested_names.intersection({"spawn", "subagent", "subagents"}):
                self.add_tool("spawn")
            else:
                self.remove_tool("spawn")
        except Exception as exc:
            logger.debug(f"Failed to adjust request-scoped spawn tool: {exc}")

    def _bind_request_execution_hints_to_tools(self, hints: dict[str, Any]) -> None:
        """Make request hints available to tool calls that lose contextvars."""
        tool_sources: list[Any] = []
        registry_tools = getattr(getattr(self, "tools", None), "_tools", {})
        if isinstance(registry_tools, dict):
            tool_sources.extend(registry_tools.values())
        manager_tools = getattr(
            getattr(getattr(self, "_agent", None), "available_tools", None), "tool_map", {}
        )
        if isinstance(manager_tools, dict):
            tool_sources.extend(manager_tools.values())
        normalized = hints if isinstance(hints, dict) else {}
        seen: set[int] = set()
        for tool in tool_sources:
            identity = id(tool)
            if identity in seen:
                continue
            seen.add(identity)
            try:
                setattr(tool, "_request_execution_hints", normalized)
            except Exception:
                continue

    def _tool_activation_status(self, name: str) -> str:
        """Return whether a registered tool is active, inactive, or missing."""
        tool_name = str(name or "").strip()
        if not tool_name or tool_name not in self.tools:
            return "missing"
        if tool_name in self.tools.get_active_tools():
            return "active"
        return "inactive"

    @staticmethod
    def _truncate_request_for_prompt(
        message: str,
        *,
        head_chars: int = 220,
        tail_chars: int = 180,
    ) -> str:
        """Keep both the head and tail of the latest request for prompt scaffolding."""
        normalized = (message or "").strip()
        if len(normalized) <= head_chars + tail_chars + 24:
            return normalized
        head = normalized[:head_chars].rstrip()
        tail = normalized[-tail_chars:].lstrip()
        return (
            f"{head}\n"
            "[... middle omitted to save tokens; preserve latest tail instructions ...]\n"
            f"{tail}"
        )

    def _apply_request_context_to_system_prompt(
        self,
        message: str,
        *,
        thinking: bool,
    ) -> tuple[str | None, object]:
        """Temporarily append active request context to the agent system prompt."""
        if not getattr(self, "_agent", None):
            return None, _MISSING

        current_prompt = getattr(self._agent, "system_prompt", None)
        if not isinstance(current_prompt, str) or not current_prompt:
            return None, _MISSING

        request_context = self._build_request_context_prompt(message)
        augmented_prompt = f"{current_prompt}\n\n## Active Request Context\n{request_context}"
        self._active_request_base_system_prompt = current_prompt
        self._active_request_augmented_system_prompt = augmented_prompt
        self._agent.system_prompt = augmented_prompt

        original_base_prompt = _MISSING
        if hasattr(self._agent, "_original_system_prompt"):
            original_base_prompt = getattr(self._agent, "_original_system_prompt")
            if isinstance(original_base_prompt, str) and original_base_prompt:
                self._agent._original_system_prompt = (
                    f"{original_base_prompt}\n\n## Active Request Context\n{request_context}"
                )

        return current_prompt, original_base_prompt

    def _restore_request_context_system_prompt(
        self,
        original_prompt: str | None,
        original_base_prompt: object,
    ) -> None:
        """Restore the agent system prompt after a thinking run completes."""
        if not getattr(self, "_agent", None):
            return
        if original_prompt is not None:
            self._agent.system_prompt = original_prompt
        if original_base_prompt is not _MISSING and hasattr(self._agent, "_original_system_prompt"):
            self._agent._original_system_prompt = original_base_prompt
        self._active_request_base_system_prompt = None
        self._active_request_augmented_system_prompt = None

    def _select_next_step_prompt(self, message: str, *, thinking: bool) -> str:
        """Choose the per-step prompt shape for the current request.

        Keep provider-visible continuation turns short and domain-neutral.
        The active request context is injected into the temporary system prompt
        instead, so tool-loop user tails cannot overtake the real user request
        or replay session compacts as executable instructions.
        """
        return self.DEFAULT_NEXT_STEP_PROMPT

    def _extract_env_for_prompt(self) -> str:
        """Extract env vars from .env.local for the step prompt.

        Non-sensitive values are shown directly while private keys are masked.
        Persists across short-term memory pruning.
        """
        env_file = self.workspace / ".env.local"
        if not env_file.exists():
            return ""
        try:
            raw = env_file.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return ""

        env_vars: dict[str, str] = {}
        for line in raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip()
            if not key:
                continue
            _sensitive = any(
                s in key.upper()
                for s in (
                    "PRIVATE",
                    "SECRET",
                    "KEY",
                    "PASSWORD",
                    "TOKEN",
                    "MNEMONIC",
                    "CREDENTIAL",
                    "AUTH",
                    "PASSPHRASE",
                )
            )
            env_vars[key] = "<set>" if _sensitive and val else val

        if not env_vars:
            return ""

        parts = [
            "\n[ENV - from .env.local - do NOT re-read; skill CLI shell calls load it automatically]:"
        ]
        for k, v in env_vars.items():
            parts.append(f"  {k}={v}")
        return "\n".join(parts) + "\n"

    # ------------------------------------------------------------------
    # Dynamic prompt helpers
    # ------------------------------------------------------------------

    def _build_skills_for_prompt(self) -> str:
        """Build Openclaw-style XML metadata for installed skills.

        Uses ``_iter_skill_candidates`` and ``_parse_skill_frontmatter`` to
        build an ``<available_skills>`` XML block.  Unorganized skills (in the
        workspace root) are flagged so the agent knows to move them first.

        Path-conditional skills are included with ``include_dormant=True``
        so the agent is *aware* of them, but they carry a ``<status>dormant``
        tag indicating they will activate when matching files are touched.
        """
        candidates = self._iter_skill_candidates(include_dormant=True)
        if not candidates:
            return ""

        entries: list[str] = []
        for name, _dir, skill_md, is_organized in candidates:
            fm = self._parse_skill_frontmatter(skill_md)
            description = fm["description"] or name
            when_to_use = fm["when_to_use"]
            skill_paths = fm.get("paths", [])

            location = self._skill_prompt_location(
                name,
                _dir,
                skill_md,
                is_organized=is_organized,
            )

            parts = [
                f'<skill name="{name}">',
                f"  <description>{description}</description>",
            ]
            if when_to_use:
                parts.append(f"  <when_to_use>{when_to_use}</when_to_use>")
            parts.append(f"  <location>{location}</location>")
            if not is_organized:
                parts.append(f"  <status>unorganized - move to skills/{name}/ before use</status>")
            elif isinstance(skill_paths, list) and skill_paths:
                is_active = self._skill_paths_match(
                    skill_paths,
                    self._touched_paths,
                    self.workspace,
                )
                if not is_active:
                    parts.append(
                        f"  <status>dormant - activates when files matching "
                        f"{', '.join(skill_paths[:3])} are touched</status>"
                    )
            parts.append("</skill>")
            entries.append("\n".join(parts))

        return "<available_skills>\n" + "\n".join(entries) + "\n</available_skills>"

    @staticmethod
    def _build_dynamic_tools_prompt(inactive_tools: dict[str, "Tool"]) -> str:
        """Build the 'Dynamically Loadable Tools' system-prompt section.

        Lists ALL inactive tools with their descriptions so the AI Agent
        can autonomously decide which to activate. No hardcoded topic
        mapping - the LLM reads tool descriptions and decides for itself.
        """
        lines: list[str] = [
            "\n\n## Dynamically Loadable Tools\n\n"
            "The callable tool list already contains active tools. Call active tools "
            "directly; do not activate a tool that is already callable.\n\n"
            "If the needed capability is not in the callable tool list, inspect this "
            "inactive catalog and call `activate_tool(action='activate', "
            "tool_name='<name>')` for the matching capability. Tool names and "
            "descriptions are the source of truth. Do not recreate a cataloged "
            "capability with generic shell commands before activating the matching "
            "tool. After activation, follow the tool schema. Use shell only when "
            "no listed tool fits or tool activation fails. Prefer specialized tools "
            "over web_search when a matching tool is listed.\n"
        ]

        for tool in inactive_tools.values():
            lines.append(f"- `{tool.name}`: {tool.description}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dynamic tool management
    # ------------------------------------------------------------------

    def add_tool(self, name: str) -> bool:
        """
        Dynamically activate a tool and inject it into the running agent.

        The tool must already be registered in the ToolRegistry.  This method
        activates it in the filter, then adds it to the agent's ToolManager
        so it becomes available for the next LLM step.

        Args:
            name: Registered tool name to activate.

        Returns:
            True if the tool was activated, False otherwise.
        """
        tool = self.tools.get(name)
        if tool is None:
            logger.warning(f"add_tool: '{name}' is not registered")
            return False

        if not self.tools.activate_tool(name):
            logger.debug(f"add_tool: '{name}' is already active")
            return False

        # If the agent is initialized, inject the tool into its ToolManager
        if self._agent and hasattr(self._agent, "available_tools"):
            tm = self._agent.available_tools
            if name not in tm.tool_map:
                tm.add_tool(tool)
                logger.info(f"Injected tool '{name}' into running agent")

        return True

    def add_tools(self, *names: str) -> list[str]:
        """
        Activate multiple tools at once.

        Args:
            *names: Tool names to activate.

        Returns:
            List of tool names that were successfully activated.
        """
        activated = []
        for name in names:
            if self.add_tool(name):
                activated.append(name)
        return activated

    def remove_tool(self, name: str) -> bool:
        """
        Dynamically deactivate a tool and remove it from the running agent.

        Args:
            name: Tool name to deactivate.

        Returns:
            True if the tool was deactivated, False otherwise.
        """
        if not self.tools.deactivate_tool(name):
            return False

        # Remove from agent's ToolManager if running
        if self._agent and hasattr(self._agent, "available_tools"):
            tm = self._agent.available_tools
            if name in tm.tool_map:
                tm.remove_tool(name)
                logger.info(f"Removed tool '{name}' from running agent")

        return True

    def get_skill_catalog(self) -> list[dict[str, Any]]:
        """Return structured metadata for skills visible to this agent.

        This is an observability/catalog surface only. It does not route user
        prompts or decide when a skill should execute.
        """
        catalog: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            candidates = self._iter_skill_candidates(include_dormant=True)
        except Exception as exc:
            return [{"error": str(exc), "status": "failed"}]

        active: set[str] = set()
        if self._skill_manager is not None:
            try:
                active = set(self._skill_manager.list())
            except Exception:
                active = set()

        workspace_skills = (self.workspace / "skills").resolve()
        user_skill_roots = {Path(p).expanduser().resolve() for p in self._user_skill_paths}
        bundled_root = (
            Path(__file__).resolve().parent.parent.parent / "workspace" / "skills"
        ).resolve()

        for name, skill_dir, skill_md, is_organized in candidates:
            if name in seen:
                continue
            seen.add(name)
            try:
                resolved_dir = skill_dir.resolve()
            except Exception:
                resolved_dir = skill_dir
            source = "workspace"
            if resolved_dir == bundled_root or bundled_root in resolved_dir.parents:
                source = "bundled"
            elif any(
                resolved_dir == root or root in resolved_dir.parents for root in user_skill_roots
            ):
                source = "configured"
            elif not (resolved_dir == workspace_skills or workspace_skills in resolved_dir.parents):
                source = "workspace-root"

            fm = self._parse_skill_frontmatter(skill_md)
            skill_paths = fm.get("paths", [])
            status = "available"
            if not is_organized:
                status = "unorganized"
            elif (
                isinstance(skill_paths, list)
                and skill_paths
                and not self._skill_paths_match(
                    skill_paths,
                    self._touched_paths,
                    self.workspace,
                )
            ):
                status = "dormant"

            catalog.append(
                {
                    "name": name,
                    "description": fm.get("description") or name,
                    "when_to_use": fm.get("when_to_use") or "",
                    "paths": skill_paths if isinstance(skill_paths, list) else [],
                    "source": source,
                    "status": status,
                    "active": name in active,
                    "base_dir": str(skill_dir),
                    "skill_md": str(skill_md),
                    "organized": bool(is_organized),
                }
            )
        return catalog

    def get_mcp_catalog(self) -> list[dict[str, Any]]:
        """Return structured metadata for configured MCP servers and loaded tools."""
        loaded_by_server: dict[str, list[str]] = {}
        for tool in self._mcp_tools:
            config = getattr(tool, "mcp_config", {}) or {}
            server_name = getattr(tool, "server_name", None) or getattr(
                tool, "mcp_server_name", None
            )
            if not server_name:
                server_name = str(getattr(tool, "name", "unknown")).split("__", 1)[0]
            loaded_by_server.setdefault(str(server_name), []).append(
                str(getattr(tool, "name", "unknown"))
            )

        catalog: list[dict[str, Any]] = []
        for name, config in self._mcp_config.items():
            transport = config.get("transport") or ("stdio" if config.get("command") else "unknown")
            loaded_tools = loaded_by_server.get(name, [])
            catalog.append(
                {
                    "name": name,
                    "transport": transport,
                    "command": config.get("command"),
                    "url": config.get("url"),
                    "status": "loaded" if loaded_tools else "configured",
                    "tool_count": len(loaded_tools),
                    "tools": loaded_tools,
                }
            )
        for server_name, tools in loaded_by_server.items():
            if server_name not in self._mcp_config:
                catalog.append(
                    {
                        "name": server_name,
                        "transport": "unknown",
                        "status": "loaded",
                        "tool_count": len(tools),
                        "tools": tools,
                    }
                )
        return catalog

    def get_available_tools(self) -> list[dict[str, Any]]:
        """
        List all registered tools with their active/inactive status.

        Returns:
            List of dicts with name, description, and active flag.
        """
        return self.tools.get_all_tool_summaries()
