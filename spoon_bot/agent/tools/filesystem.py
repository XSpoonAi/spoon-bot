"""File system tools: read, write, edit, list.

All tools enforce workspace boundary security to prevent path traversal attacks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os

from spoon_bot.agent.tools.base import Tool
from spoon_bot.agent.tools.execution_context import (
    _REDUNDANT_FILE_READ_MESSAGE,
    capture_tool_output,
    get_request_execution_hints,
    invalidate_file_read_tracking,
    suppress_redundant_file_read,
)
from spoon_bot.agent.tools.path_validator import (
    PathValidator,
    set_default_validator,
    validate_directory_path,
    validate_read_path,
    validate_write_path,
)


class ReadFileTool(Tool):
    """Tool to read file contents with encoding fallback and path traversal protection."""

    def __init__(
        self,
        workspace: Path | str | None = None,
        additional_read_paths: list[Path | str] | None = None,
        max_output: int = 6000,
    ):
        self._workspace = Path(workspace).resolve() if workspace else None
        self._additional_read_paths = additional_read_paths
        self._max_output = max_output
        self._validator = (
            PathValidator(workspace=workspace, additional_read_paths=additional_read_paths)
            if workspace else None
        )

    def set_workspace(self, workspace: Path | str) -> None:
        """Set the workspace boundary for path validation."""
        self._workspace = Path(workspace).resolve()
        self._validator = PathValidator(
            workspace=self._workspace,
            additional_read_paths=self._additional_read_paths,
        )
        set_default_validator(self._validator)

    @property
    def name(self) -> str:
        return "read_file"

    def tool_invocation_dedup_key(self, kwargs: dict[str, Any]) -> None:
        """Use content-range coverage instead of exact-call hard stops."""
        return None

    @property
    def description(self) -> str:
        workspace_note = ""
        if self._workspace:
            workspace_note = f" Files must be within the workspace: {self._workspace}"
        return (
            f"Read file contents.{workspace_note} "
            "Use offset+limit for large files (line-based). "
            "For searching specific values, prefer `grep` tool instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read (must be within workspace)",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start from (1-indexed). Omit to read from start.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max lines to read. Omit for whole file (subject to truncation).",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        offset: int | None = None,
        limit: int | None = None,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> str:
        """Read file contents with path traversal protection."""
        try:
            result = (
                self._validator.validate_read_path(path)
                if self._validator is not None
                else validate_read_path(path, workspace=self._workspace)
            )
            if not result.valid:
                return f"Security Error: {result.error}"

            file_path = result.resolved_path
            assert file_path is not None

            if self._is_unrequested_session_transcript_path(file_path):
                return (
                    "CURRENT_REQUEST_CONTEXT_GUARD: session transcripts are not "
                    "a source for a new workspace/build task. The latest user "
                    "request is already in context; inspect project files or "
                    "create the requested artifact instead of reading sessions/."
                )

            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"

            try:
                async with aiofiles.open(file_path, "r", encoding=encoding) as f:
                    content = await f.read()
            except UnicodeDecodeError:
                async with aiofiles.open(file_path, "r", encoding="latin-1") as f:
                    content = await f.read()

            content_fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
            all_lines = content.split("\n")
            total_lines = len(all_lines)
            start = max(0, (offset or 1) - 1)
            if offset is not None and start >= total_lines:
                return f"Error: Offset {offset} beyond end of file ({total_lines} lines)"

            # Line-range selection (Pi-style offset+limit)
            if offset is not None or limit is not None:
                end = min(total_lines, start + limit) if limit else total_lines
                content = "\n".join(all_lines[start:end])
                visible_start_line = start + 1
                selected_line_count = max(1, end - start)
                range_note = f" | lines {start + 1}-{end}/{total_lines}"
            else:
                visible_start_line = 1
                selected_line_count = total_lines
                range_note = ""

            full_content = content
            total_size = len(full_content)
            parts = file_path.parts
            is_skill_file = "skills" in parts

            # Use relative path to workspace for display, fallback to name
            try:
                if self._workspace:
                    rel = file_path.relative_to(self._workspace)
                    display_path = str(rel).replace("\\", "/")
                else:
                    display_path = file_path.name
            except ValueError:
                display_path = file_path.name

            coverage_start_line = visible_start_line
            coverage_line_count = selected_line_count

            if is_skill_file and file_path.name.casefold() == "skill.md":
                skill_budget = int(self._max_output or 6000)
                if offset is not None or limit is not None or total_size <= skill_budget:
                    if offset is None and limit is None:
                        content = (
                            "[SKILL.md execution contract]\n"
                            "Full SKILL.md content follows; treat it as the "
                            "source of truth for this skill.\n\n"
                            f"{full_content}"
                        )
                    else:
                        content = full_content
                    visible_line_count = selected_line_count
                else:
                    content = self._extract_skill_cli_content(
                        full_content,
                        budget=skill_budget,
                    )
                    visible_line_count = content.count("\n") + 1 if content else 0
                    # This is a synthetic summary, not the raw file range. Do
                    # not mark the full SKILL.md as covered, otherwise a later
                    # offset/limit read for the actual contract body is treated
                    # as a duplicate and the agent can get stuck without the
                    # evidence it asked for.
                    coverage_line_count = 0
            elif self._max_output and total_size > self._max_output:
                visible_prefix = full_content[:self._max_output]
                visible_line_count = visible_prefix.count("\n")
                content = visible_prefix + f"\n... (truncated, {total_size - self._max_output} more chars)"
                coverage_line_count = max(1, visible_line_count)
            else:
                visible_line_count = selected_line_count
                content = full_content
                coverage_line_count = visible_line_count

            duplicate_read = None
            if coverage_line_count > 0:
                duplicate_read = suppress_redundant_file_read(
                    str(file_path),
                    offset=coverage_start_line,
                    limit=coverage_line_count,
                    total_lines=total_lines,
                    content_fingerprint=content_fingerprint,
                    request_key=(
                        f"{file_path}\x1f{offset if offset is not None else ''}"
                        f"\x1f{limit if limit is not None else ''}"
                        f"\x1f{content_fingerprint}"
                    ),
                )

            def _build_result(body: str) -> str:
                body_size = len(body)
                header = f"[file: {display_path} | {body_size} chars{range_note}"
                if is_skill_file:
                    header += " | skill-ref"
                header += "]\n"
                return header + body

            summary_result = _build_result(content)
            full_result = _build_result(full_content)
            read_metadata = {
                "path": str(display_path),
                "resolved_path": str(file_path),
                "sha256": content_fingerprint,
                "bytes": len(full_content.encode("utf-8", errors="replace")),
                "lines": total_lines,
                "complete": bool(
                    offset is None
                    and limit is None
                    and (
                        not self._max_output
                        or total_size <= self._max_output
                        or (is_skill_file and file_path.name.casefold() == "skill.md")
                    )
                ),
                "skill_ref": bool(is_skill_file and file_path.name.casefold() == "skill.md"),
            }
            if duplicate_read is not None:
                if duplicate_read == _REDUNDANT_FILE_READ_MESSAGE:
                    duplicate_with_content = f"{duplicate_read}\n{summary_result}"
                    duplicate_full = f"{duplicate_read}\n{full_result}"
                    capture_tool_output(
                        duplicate_with_content,
                        duplicate_full,
                        metadata=read_metadata,
                    )
                    return duplicate_with_content
                capture_tool_output(duplicate_read, duplicate_read, metadata=read_metadata)
                return duplicate_read
            capture_tool_output(summary_result, full_result, metadata=read_metadata)
            return summary_result

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @staticmethod
    def _is_unrequested_session_transcript_path(path: Path) -> bool:
        normalized = str(path or "").replace("\\", "/").casefold()
        if "/.spoon-bot/llm_context_snapshots/" in normalized:
            return True
        hints = get_request_execution_hints()
        if isinstance(hints, dict) and hints.get("current_session_fact_check_required"):
            return False
        return "/sessions/" in normalized

    @staticmethod
    def _extract_skill_cli_content(content: str, budget: int = 2500) -> str:
        """Extract a compact execution contract from SKILL.md."""
        import re

        lines = content.split("\n")
        budget = max(800, int(budget or 2500))

        def _dedupe(values: list[str], *, limit: int) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for value in values:
                normalized = " ".join(str(value or "").strip().split())
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                out.append(normalized)
                if len(out) >= limit:
                    break
            return out

        def _is_optional_input_prompt_directive(value: str) -> bool:
            lowered = str(value or "").casefold()
            optional_markers = (
                "optional",
                "bonus",
                "referral",
                "invitation",
                "invite",
                "extra ",
                "non-essential",
            )
            input_markers = (
                "ask ",
                "provide",
                "present in",
                "already present",
                "conversation context",
                "context",
                "code",
                "configuration",
                "preference",
                "naming",
            )
            if lowered.startswith("ask "):
                return True
            return any(marker in lowered for marker in optional_markers) and any(
                marker in lowered for marker in input_markers
            )

        frontmatter: list[str] = []
        if lines and lines[0].strip() == "---":
            for line in lines[1:40]:
                if line.strip() == "---":
                    break
                if line.strip():
                    frontmatter.append(line.strip())

        cli_lines = _dedupe(
            [
                line.strip()
                for line in lines
                if re.match(r"^\s*CLI\s*:?\s*=", line, re.IGNORECASE)
            ],
            limit=3,
        )

        command_section: list[str] = []
        in_commands_section = False
        for raw_line in lines:
            stripped = raw_line.strip()
            if re.match(r"^#{1,6}\s+commands\b", stripped, re.IGNORECASE):
                in_commands_section = True
                continue
            if in_commands_section and re.match(r"^#{1,6}\s+\S", stripped):
                break
            if not in_commands_section:
                continue
            if not stripped or stripped.startswith("```"):
                continue
            if stripped.startswith("$CLI") or re.match(r"^[A-Za-z0-9_.\\/-]+\s+", stripped):
                command_section.append(stripped)

        setup_headings = (
            "setup",
            "prerequisite",
            "install",
            "dependency",
            "dependencies",
        )
        setup_section_lines: list[str] = []
        in_setup_section = False
        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip().casefold()
                in_setup_section = any(
                    heading in title for heading in setup_headings
                )
                continue
            if in_setup_section:
                setup_section_lines.append(raw_line)

        setup_commands: list[str] = []
        if setup_section_lines:
            from spoon_bot.agent.request_hints import extract_shell_command_candidates

            command_starters = (
                "cd ",
                "npm ",
                "pnpm ",
                "npx ",
                "yarn ",
                "bun ",
                "deno ",
                "node ",
                "uv ",
                "pip ",
                "poetry ",
                "python ",
                "bash ",
                "sh ",
                "./",
                "../",
                "/",
            )
            candidate_commands = extract_shell_command_candidates(
                "\n".join(setup_section_lines),
                limit=24,
            )
            setup_commands = _dedupe(
                [
                    command
                    for command in candidate_commands
                    if command.casefold().startswith(command_starters)
                ],
                limit=12,
            )
            setup_commands = _dedupe(
                setup_commands
                + [
                    line.strip()
                    for line in setup_section_lines
                    if line.strip().casefold().startswith(command_starters)
                ],
                limit=12,
            )

        contract_headings = (
            "setup",
            "prerequisite",
            "install",
            "dependency",
            "dependencies",
            "primary flow",
            "workflow",
            "procedure",
            "execution rule",
            "rule",
            "steps",
        )
        directive_prefixes = (
            "procedure ",
            "step ",
            "run ",
            "match ",
            "if ",
            "else",
            "otherwise",
            "ask ",
            "before ",
            "after ",
            "once ",
            "when ",
            "decide ",
            "use ",
            "prefer ",
            "complete ",
            "do not ",
            "don't ",
            "stop ",
            "return ",
            "report ",
            "keep ",
            "rule ",
            "- ",
            "$cli",
        )
        contract_lines: list[str] = []
        in_contract_section = False
        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip().casefold()
                in_contract_section = any(
                    heading in title for heading in contract_headings
                )
                continue
            if not in_contract_section:
                continue
            if stripped.startswith("```"):
                continue
            if not stripped:
                continue
            lowered = stripped.casefold()
            if (
                lowered.startswith(directive_prefixes)
                or "$cli" in lowered
                or stripped.startswith('"')
            ):
                if _is_optional_input_prompt_directive(lowered):
                    continue
                contract_lines.append(raw_line.rstrip()[:220])

        contract_lines = _dedupe(contract_lines, limit=90)

        rule_lines = _dedupe(
            [
                line.strip()
                for line in lines
                if line.strip().casefold().startswith("rule ")
            ],
            limit=12,
        )

        sections: list[str] = ["[SKILL.md execution summary]"]
        if frontmatter:
            sections.append("Metadata:\n" + "\n".join(frontmatter[:12]))
        sections.append(
            "Command contract guardrails:\n"
            "- Prompt topic, product, repo, and site words are context, not CLI "
            "arguments, unless SKILL.md documents them.\n"
            "- Replace only placeholders such as <value> or {value}; on argument "
            "or usage errors, check CLI help and retry the documented form."
        )
        sections.append(
            "Workflow-unit scope precedence:\n"
            "- The newest user request selects the workflow and count for this "
            "turn; SKILL.md after-action, continuous, recovery, lifecycle, "
            "settlement, and post-action rules define how to finish each "
            "selected workflow unit.\n"
            "- Follow those rules when they are necessary to complete the "
            "selected unit through its contract-defined terminal outcome or "
            "concrete blocker.\n"
            "- Do not use those rules to start another repeated unit, larger "
            "count, or unrelated side effect unless the newest user request "
            "explicitly asks for that larger scope. For continuation-only "
            "messages, perform at most one bounded unit."
        )
        sections.append(
            "Non-interactive optional-input precedence:\n"
            "- If a contract asks for optional configuration, preference, bonus, "
            "referral, naming, or another non-essential enhancement during an "
            "already selected workflow, do not ask the user and do not stop.\n"
            "- Use the default/no-extra path and continue with the core requested "
            "workflow unless the next core command cannot be formed without that "
            "missing value.\n"
            "- Omitted `ask ...` directives from this compact summary are governed "
            "by this policy; inspect the full SKILL.md only if a core command "
            "cannot be formed or a tool error proves required input is missing.\n"
            "- Do not emit optional-input questions as visible progress before "
            "continuing."
        )
        if cli_lines:
            sections.append("CLI entrypoint:\n" + "\n".join(cli_lines))
        if setup_commands:
            sections.append(
                "Setup commands to run before primary commands:\n"
                + "\n".join(setup_commands)
            )
        if contract_lines:
            sections.append("Operational contract:\n" + "\n".join(contract_lines))
        if command_section:
            sections.append("Documented commands:\n" + "\n".join(command_section[:24]))
        if rule_lines:
            sections.append("Execution rules:\n" + "\n".join(rule_lines))

        summary = "\n\n".join(sections).strip()
        if len(summary) <= budget:
            return summary
        return summary[:budget].rstrip() + "\n... (SKILL.md execution summary truncated)"


class WriteFileTool(Tool):
    """Tool to write content to a file with path traversal protection."""

    def __init__(self, workspace: Path | str | None = None):
        """
        Initialize the write file tool.

        Args:
            workspace: The allowed workspace directory. Writes outside this
                       directory will be rejected for security.
        """
        self._workspace = Path(workspace).resolve() if workspace else None
        self._validator = PathValidator(workspace=workspace) if workspace else None

    def set_workspace(self, workspace: Path | str) -> None:
        """Set the workspace boundary for path validation."""
        self._workspace = Path(workspace).resolve()
        self._validator = PathValidator(workspace=self._workspace)
        set_default_validator(self._validator)

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        workspace_note = ""
        if self._workspace:
            workspace_note = f" Files must be within the workspace: {self._workspace}"
        return (
            "Write content to a new file, or replace an existing file only when "
            "overwrite=true is intentionally set for a whole-file replacement. "
            "For targeted changes to an existing file, use edit_file instead. "
            f"Creates parent directories if needed.{workspace_note}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write to (must be within workspace)",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": (
                        "Set true only for an intentional whole-file replacement. "
                        "Existing files are protected by default; use edit_file "
                        "for targeted changes."
                    ),
                    "default": False,
                },
            },
            "required": ["path", "content"],
        }

    async def execute(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        """Write content to file with path traversal protection."""
        try:
            # Validate the path (write validation is stricter)
            result = validate_write_path(path, workspace=self._workspace)
            if not result.valid:
                return f"Security Error: {result.error}"

            file_path = result.resolved_path
            assert file_path is not None  # Guaranteed by valid=True

            if file_path.exists() and file_path.is_file():
                try:
                    existing = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    existing = file_path.read_text(encoding="latin-1")
                if existing == content:
                    return f"No changes: {path} already has the requested content"
                if not overwrite:
                    return (
                        f"Error: File already exists: {path}. "
                        "Use edit_file for targeted changes, or call write_file "
                        "with overwrite=true only when the whole-file replacement "
                        "is intentional. This is recoverable: if you just generated "
                        "the complete replacement content for this same file, retry "
                        "the same write_file call with overwrite=true."
                    )

            # Create parent directories (already validated to be within workspace)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(content)

            invalidate_file_read_tracking(path, str(file_path))
            action = "overwrote" if overwrite else "wrote"
            digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = None
            result_text = (
                f"Successfully {action} {len(content)} bytes to {path} "
                f"(sha256={digest})"
            )
            capture_tool_output(
                result_text,
                result_text,
                metadata={
                    "path": path,
                    "resolved_path": str(file_path),
                    "operation": action,
                    "bytes": len(content),
                    "sha256": digest,
                    "mtime": mtime,
                },
            )
            return result_text

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"


class EditFileTool(Tool):
    """Tool to edit a file by replacing text with path traversal protection."""

    def __init__(self, workspace: Path | str | None = None):
        """
        Initialize the edit file tool.

        Args:
            workspace: The allowed workspace directory. Edits outside this
                       directory will be rejected for security.
        """
        self._workspace = Path(workspace).resolve() if workspace else None
        self._validator = PathValidator(workspace=workspace) if workspace else None

    def set_workspace(self, workspace: Path | str) -> None:
        """Set the workspace boundary for path validation."""
        self._workspace = Path(workspace).resolve()
        self._validator = PathValidator(workspace=self._workspace)
        set_default_validator(self._validator)

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        workspace_note = ""
        if self._workspace:
            workspace_note = f" Files must be within the workspace: {self._workspace}"
        return (
            "Edit a file by replacing old_text with new_text. "
            "The old_text must exist exactly once in the file. If the same "
            "replacement has already been applied, this returns a no-op success. "
            f"{workspace_note}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to edit (must be within workspace)",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace with",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        **kwargs: Any,
    ) -> str:
        """Edit file by replacing text with path traversal protection."""
        try:
            # Validate the path (use write validation since we're modifying)
            result = validate_write_path(path, workspace=self._workspace)
            if not result.valid:
                return f"Security Error: {result.error}"

            file_path = result.resolved_path
            assert file_path is not None  # Guaranteed by valid=True

            if not file_path.exists():
                return f"Error: File not found: {path}"

            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()

            if old_text not in content:
                if new_text and content.count(new_text) == 1:
                    return f"No change needed: requested edit already applied to {path}"
                return "Error: old_text not found in file. Make sure it matches exactly."

            # Count occurrences
            count = content.count(old_text)
            if count > 1:
                return (
                    f"Warning: old_text appears {count} times. "
                    "Please provide more context to make it unique."
                )

            new_content = content.replace(old_text, new_text, 1)

            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                await f.write(new_content)

            invalidate_file_read_tracking(path, str(file_path))
            digest = hashlib.sha256(new_content.encode("utf-8", errors="replace")).hexdigest()
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = None
            result_text = f"Successfully edited {path} (sha256={digest})"
            capture_tool_output(
                result_text,
                result_text,
                metadata={
                    "path": path,
                    "resolved_path": str(file_path),
                    "operation": "edit",
                    "bytes": len(new_content),
                    "sha256": digest,
                    "mtime": mtime,
                },
            )
            return result_text

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"


class ListDirTool(Tool):
    """Tool to list directory contents with path traversal protection."""

    def __init__(
        self,
        workspace: Path | str | None = None,
        additional_read_paths: list[Path | str] | None = None,
    ):
        self._workspace = Path(workspace).resolve() if workspace else None
        self._additional_read_paths = additional_read_paths
        self._validator = (
            PathValidator(workspace=workspace, additional_read_paths=additional_read_paths)
            if workspace else None
        )

    def set_workspace(self, workspace: Path | str) -> None:
        """Set the workspace boundary for path validation."""
        self._workspace = Path(workspace).resolve()
        self._validator = PathValidator(
            workspace=self._workspace,
            additional_read_paths=self._additional_read_paths,
        )
        set_default_validator(self._validator)

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        workspace_note = ""
        if self._workspace:
            workspace_note = f" Directory must be within the workspace: {self._workspace}"
        return f"List the contents of a directory with file/folder indicators.{workspace_note}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list (must be within workspace)",
                },
                "show_hidden": {
                    "type": "boolean",
                    "description": "Show hidden files (default: false)",
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        show_hidden: bool = False,
        **kwargs: Any,
    ) -> str:
        """List directory contents with path traversal protection."""
        try:
            # Validate the path
            result = (
                self._validator.validate_directory_path(path)
                if self._validator is not None
                else validate_directory_path(path, workspace=self._workspace)
            )
            if not result.valid:
                return f"Security Error: {result.error}"

            dir_path = result.resolved_path
            assert dir_path is not None  # Guaranteed by valid=True

            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"

            items = []
            for item in sorted(dir_path.iterdir()):
                # Skip hidden files unless requested
                if not show_hidden and item.name.startswith("."):
                    continue

                if item.is_dir():
                    items.append(f"[DIR]  {item.name}/")
                elif item.is_symlink():
                    # Check if symlink target is safe to show
                    try:
                        target = item.resolve()
                        if self._workspace and not self._is_within_workspace(target):
                            items.append(f"[LINK] {item.name} -> (outside workspace)")
                        else:
                            items.append(f"[LINK] {item.name}")
                    except OSError:
                        items.append(f"[LINK] {item.name} -> (broken)")
                else:
                    # Show file size
                    try:
                        size = item.stat().st_size
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size // 1024}KB"
                        else:
                            size_str = f"{size // (1024 * 1024)}MB"
                        items.append(f"[FILE] {item.name} ({size_str})")
                    except OSError:
                        items.append(f"[FILE] {item.name}")

            if not items:
                return f"Directory {path} is empty"

            return "\n".join(items)

        except PermissionError:
            return f"Error: Permission denied: {path}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def _is_within_workspace(self, path: Path) -> bool:
        """Check if a path is within the workspace."""
        if self._workspace is None:
            return True
        try:
            path.relative_to(self._workspace)
            return True
        except ValueError:
            return False
