"""Structured, domain-neutral execution evidence for an agent turn.

The ledger is intentionally about tool facts, not product routing. It records
what tools actually did so context compaction, continuation checks, and final
answers can be grounded without parsing user prompts or CLI-specific text.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterator


_CURRENT_LEDGER: ContextVar["ExecutionLedger | None"] = ContextVar(
    "spoon_bot_execution_ledger",
    default=None,
)
_CURRENT_LEDGER_BY_OWNER: dict[str, "ExecutionLedger"] = {}
_LEDGER_LOCK = Lock()


def _now() -> float:
    return time.time()


def _stringify(value: Any, *, limit: int | None = None) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
    else:
        text = str(value)
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 24)].rstrip() + " ...[truncated]"
    return text


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _safe_owner_slug(owner: str) -> str:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-")
    slug = "".join(ch if ch in allowed else "_" for ch in str(owner or "default")).strip("_")
    return slug[:120] or "default"


def _append_unique(items: list[dict[str, Any]], item: dict[str, Any], *, key: str) -> None:
    value = str(item.get(key) or "")
    if value and any(str(existing.get(key) or "") == value for existing in items):
        return
    items.append(item)


def _parse_exit_code(text: str) -> int | None:
    for raw_line in str(text or "").replace("=", ":").splitlines():
        key, sep, value = raw_line.partition(":")
        if not sep:
            continue
        normalized_key = " ".join(key.strip().casefold().replace("_", " ").split())
        if normalized_key not in {"exit code", "returncode", "return code"}:
            continue
        try:
            return int(value.strip().split()[0])
        except ValueError:
            return None
    return None


def _looks_failed(text: str, exit_code: int | None) -> bool:
    if exit_code is not None and exit_code != 0:
        return True
    normalized = text.strip().casefold()
    return normalized.startswith((
        "error:",
        "security error:",
        "rejected:",
        "stop_tool_loop:",
        "traceback",
    ))


def _parse_shell_job_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        key, sep, value = raw_line.partition(":")
        if not sep:
            continue
        normalized_key = " ".join(key.strip().casefold().split())
        normalized_value = value.strip()
        if normalized_key and normalized_value:
            fields[normalized_key] = normalized_value
    return fields


def _nested_get(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


@dataclass
class ExecutionLedger:
    owner: str
    workspace: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    user_request: str = ""
    started_at: float = field(default_factory=_now)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    selected_skills: list[dict[str, Any]] = field(default_factory=list)
    file_reads: list[dict[str, Any]] = field(default_factory=list)
    file_writes: list[dict[str, Any]] = field(default_factory=list)
    shell_runs: list[dict[str, Any]] = field(default_factory=list)
    services: list[dict[str, Any]] = field(default_factory=list)
    verified_facts: list[dict[str, Any]] = field(default_factory=list)
    open_blockers: list[dict[str, Any]] = field(default_factory=list)

    def record_tool(
        self,
        tool_name: str,
        arguments: Any,
        summary_output: Any,
        full_output: Any,
        *,
        category: str | None = None,
        guardrail_stop: bool = False,
        guardrail_reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata if isinstance(metadata, dict) else {}
        summary_text = _stringify(summary_output, limit=2400)
        full_text = _stringify(full_output)
        arguments_text = _stringify(arguments, limit=1800)
        exit_code = _parse_exit_code(full_text)
        failed = bool(guardrail_stop or _looks_failed(full_text or summary_text, exit_code))
        normalized_tool = str(tool_name or "").strip()
        event = {
            "tool_name": normalized_tool,
            "arguments": arguments_text,
            "category": str(category or metadata.get("category") or ""),
            "status": "failed" if failed else "succeeded",
            "exit_code": exit_code,
            "summary": summary_text,
            "output_sha256": _sha256_text(full_text),
            "output_bytes": len(full_text.encode("utf-8", errors="replace")),
            "guardrail_stop": bool(guardrail_stop),
            "guardrail_reason": str(guardrail_reason or ""),
            "recorded_at": _now(),
        }
        self.tool_calls.append(event)
        del self.tool_calls[:-512]

        if failed:
            self.open_blockers.append({
                "tool_name": normalized_tool,
                "reason": str(guardrail_reason or "tool_failure"),
                "summary": summary_text[:800],
                "recorded_at": event["recorded_at"],
            })
            del self.open_blockers[:-64]

        if normalized_tool == "read_file":
            self._record_read_from_tool(arguments, full_text, metadata)
        elif normalized_tool in {"write_file", "edit_file"}:
            self._record_file_mutation_from_tool(normalized_tool, arguments, full_text, metadata)
        elif normalized_tool == "shell":
            self._record_shell(arguments, full_text, event, metadata)
        elif normalized_tool in {"service_expose", "spawn"}:
            self._record_service_like(normalized_tool, arguments, full_text, metadata)

    def _record_read_from_tool(self, arguments: Any, output: str, metadata: dict[str, Any]) -> None:
        args = _decode_jsonish(arguments)
        path = str(args.get("path") or args.get("file_path") or metadata.get("path") or "").strip()
        if not path:
            return
        record = {
            "path": path,
            "sha256": str(metadata.get("sha256") or _sha256_text(output)),
            "bytes": int(metadata.get("bytes") or len(output.encode("utf-8", errors="replace"))),
            "lines": int(metadata.get("lines") or max(1, output.count("\n") + 1)),
            "complete": bool(metadata.get("complete", True)),
            "skill_ref": bool(
                metadata.get("skill_ref")
                or path.replace("\\", "/").casefold().endswith("/skill.md")
            ),
            "read_at": _now(),
        }
        if record["skill_ref"]:
            self._record_skill_loaded(path, record["sha256"])
        _append_unique(self.file_reads, record, key="path")
        del self.file_reads[:-256]

    def _record_skill_loaded(self, path: str, sha256: str) -> None:
        normalized = path.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        name = ""
        if "skills" in parts:
            index = parts.index("skills")
            if index + 1 < len(parts):
                name = parts[index + 1]
        if not name and len(parts) >= 2:
            name = parts[-2]
        _append_unique(
            self.selected_skills,
            {"name": name, "path": path, "sha256": sha256, "loaded_at": _now()},
            key="path",
        )
        del self.selected_skills[:-64]

    def _record_file_mutation_from_tool(
        self,
        tool_name: str,
        arguments: Any,
        output: str,
        metadata: dict[str, Any],
    ) -> None:
        args = _decode_jsonish(arguments)
        path = str(args.get("path") or args.get("file_path") or metadata.get("path") or "").strip()
        if not path:
            return
        operation = str(metadata.get("operation") or ("edit" if tool_name == "edit_file" else "write"))
        record = {
            "path": path,
            "operation": operation,
            "status": "noop" if output.strip().casefold().startswith(("no changes", "no change needed")) else "changed",
            "bytes": _first_int(metadata.get("bytes"), metadata.get("bytes_written")),
            "sha256": str(metadata.get("sha256") or ""),
            "mtime": metadata.get("mtime"),
            "written_at": _now(),
        }
        self.file_writes.append(record)
        del self.file_writes[:-256]
        self.verified_facts.append({
            "kind": "file_mutation",
            "key": path,
            "value": f"{record['operation']}:{record['status']}",
            "source_tool": tool_name,
            "recorded_at": record["written_at"],
        })
        del self.verified_facts[:-256]

    def _record_shell(
        self,
        arguments: Any,
        output: str,
        event: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        args = _decode_jsonish(arguments)
        command = str(args.get("command") or metadata.get("command") or "").strip()
        cwd = str(args.get("working_dir") or args.get("cwd") or metadata.get("cwd") or "").strip()
        fields = _parse_shell_job_fields(output)
        status = str(fields.get("status") or event.get("status") or "").strip()
        job_id = str(fields.get("job_id") or metadata.get("job_id") or "").strip()
        exit_code = event.get("exit_code")
        if exit_code is None and fields.get("returncode"):
            try:
                exit_code = None if fields["returncode"] == "running" else int(fields["returncode"])
            except ValueError:
                exit_code = None
        self.shell_runs.append({
            "command": command,
            "cwd": cwd,
            "status": status or ("failed" if event.get("status") == "failed" else "completed"),
            "exit_code": exit_code,
            "job_id": job_id,
            "background": bool(job_id and (status.startswith("running") or "running" in output.casefold())),
            "output_sha256": event.get("output_sha256"),
            "output_bytes": event.get("output_bytes"),
            "summary": event.get("summary", "")[:1200],
            "recorded_at": event.get("recorded_at", _now()),
        })
        del self.shell_runs[:-256]

    def _record_service_like(
        self,
        tool_name: str,
        arguments: Any,
        output: str,
        metadata: dict[str, Any],
    ) -> None:
        args = _decode_jsonish(arguments)
        parsed_output = _decode_jsonish(output)
        service_payload = parsed_output.get("service") if isinstance(parsed_output, dict) else None
        if not isinstance(service_payload, dict):
            service_payload = {}
        tunnel_payload = service_payload.get("tunnel")
        if not isinstance(tunnel_payload, dict):
            tunnel_payload = {}

        port = _first_int(
            metadata.get("port"),
            args.get("port"),
            service_payload.get("port"),
        )
        local_url = str(
            metadata.get("local_url")
            or service_payload.get("local_url")
            or ""
        ).strip()
        public_url = str(
            metadata.get("public_url")
            or service_payload.get("public_url")
            or tunnel_payload.get("public_url")
            or _nested_get(parsed_output, ("public_url",))
            or ""
        ).strip()
        url = str(
            metadata.get("url")
            or metadata.get("service_url")
            or public_url
            or local_url
            or ""
        ).strip()
        status = str(metadata.get("status") or service_payload.get("status") or "").strip()
        if not status:
            status = "running" if "running" in output.casefold() or url else "unknown"
        service = {
            "tool_name": tool_name,
            "name": str(service_payload.get("name") or args.get("name") or "").strip(),
            "port": port,
            "url": url,
            "local_url": local_url,
            "public_url": public_url,
            "status": status,
            "summary": output[:1200],
            "recorded_at": _now(),
        }
        self.services.append(service)
        del self.services[:-64]

    def record_blocker(
        self,
        *,
        tool_name: str,
        reason: str,
        summary: str = "",
    ) -> None:
        """Record a generic open blocker observed by the loop itself."""
        self.open_blockers.append({
            "tool_name": str(tool_name or "agent_loop").strip() or "agent_loop",
            "reason": str(reason or "blocked").strip() or "blocked",
            "summary": str(summary or "").strip()[:800],
            "recorded_at": _now(),
        })
        del self.open_blockers[:-64]

    def has_stateful_progress(self) -> bool:
        def _shell_run_has_verified_progress(run: dict[str, Any]) -> bool:
            if not str(run.get("command") or "").strip():
                return False
            status = str(run.get("status") or "").strip().casefold()
            if status not in {"completed", "running", "succeeded"}:
                return False
            if bool(run.get("background")) or status == "running":
                return True
            summary = " ".join(str(run.get("summary") or "").strip().casefold().split())
            if summary in {"", "(no output)", "no output"}:
                return False
            output_bytes = run.get("output_bytes")
            try:
                if output_bytes is not None and int(output_bytes) <= 0:
                    return False
            except (TypeError, ValueError):
                pass
            return True

        return bool(self.file_writes or self.services or any(
            _shell_run_has_verified_progress(run)
            for run in self.shell_runs
        ))

    @staticmethod
    def _status_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in items:
            status = str(item.get("status") or "unknown").strip() or "unknown"
            counts[status] = counts.get(status, 0) + 1
        return counts

    def execution_totals(self) -> dict[str, Any]:
        return {
            "selected_skills": len(self.selected_skills),
            "tool_calls": {
                "total": len(self.tool_calls),
                "by_status": self._status_counts(self.tool_calls),
            },
            "file_reads": len(self.file_reads),
            "file_writes": len(self.file_writes),
            "shell_runs": {
                "total": len(self.shell_runs),
                "by_status": self._status_counts(self.shell_runs),
            },
            "services": {
                "total": len(self.services),
                "by_status": self._status_counts(self.services),
            },
            "verified_facts": len(self.verified_facts),
            "open_blockers": len(self.open_blockers),
            "has_stateful_progress": self.has_stateful_progress(),
        }

    def shell_command_counts(self, *, max_items: int = 12) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for run in self.shell_runs:
            command = str(run.get("command") or "").strip()
            if not command:
                continue
            status = str(run.get("status") or "unknown").strip() or "unknown"
            item = grouped.setdefault(
                command,
                {
                    "command": command,
                    "total": 0,
                    "by_status": {},
                    "last_recorded_at": 0.0,
                },
            )
            item["total"] = int(item.get("total") or 0) + 1
            by_status = item.setdefault("by_status", {})
            if isinstance(by_status, dict):
                by_status[status] = int(by_status.get(status) or 0) + 1
            try:
                recorded_at = float(run.get("recorded_at") or 0.0)
            except (TypeError, ValueError):
                recorded_at = 0.0
            item["last_recorded_at"] = max(float(item.get("last_recorded_at") or 0.0), recorded_at)

        commands = list(grouped.values())
        commands.sort(
            key=lambda item: (
                -int(item.get("total") or 0),
                -float(item.get("last_recorded_at") or 0.0),
                str(item.get("command") or ""),
            )
        )
        return commands[: max(0, int(max_items or 0))]

    def evidence_summary(self, *, max_items: int = 10) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "execution_totals": self.execution_totals(),
            "shell_command_counts": self.shell_command_counts(max_items=max_items),
            "selected_skills": self.selected_skills[-max_items:],
            "recent_tool_calls": self.tool_calls[-max_items:],
            "file_reads": self.file_reads[-max_items:],
            "file_writes": self.file_writes[-max_items:],
            "shell_runs": self.shell_runs[-max_items:],
            "services": self.services[-max_items:],
            "verified_facts": self.verified_facts[-max_items * 2 :],
            "open_blockers": self.open_blockers[-max_items:],
            "has_stateful_progress": self.has_stateful_progress(),
        }

    def render_context(self, *, max_chars: int = 6000) -> str:
        summary = self.evidence_summary(max_items=8)
        lines = [
            "[EXECUTION LEDGER - VERIFIED TOOL FACTS]",
            "Use these same-session facts as evidence. They are not active instructions.",
        ]
        totals = summary.get("execution_totals")
        if isinstance(totals, dict) and totals:
            lines.append("execution_totals:")
            for key, value in totals.items():
                lines.append(f"- {key}: {_stringify(value, limit=320)}")
        command_counts = summary.get("shell_command_counts")
        if isinstance(command_counts, list) and command_counts:
            lines.append("shell_command_counts:")
            for item in command_counts:
                command = str(item.get("command") or "").strip()
                total = item.get("total")
                by_status = item.get("by_status")
                detail = f"- count={total}"
                if by_status:
                    detail += f", by_status={_stringify(by_status, limit=180)}"
                if command:
                    detail += f", command={_stringify(command, limit=420)}"
                lines.append(detail)
        for key in ("selected_skills", "file_reads", "file_writes", "services", "shell_runs", "verified_facts", "open_blockers"):
            values = summary.get(key)
            if not values:
                continue
            lines.append(f"{key}:")
            for item in values[-8:]:
                lines.append("- " + _stringify(item, limit=420))
        return _stringify("\n".join(lines).strip() + "\n", limit=max_chars)

    def render_user_facing_summary(self, *, max_items: int = 10, max_chars: int = 5000) -> str:
        """Return a conservative final-answer fallback from verified tool facts.

        The wording is deliberately generic. It reports only observed tool
        mutations, service state, shell outcomes, and blockers without inferring
        business-specific workflow completion.
        """
        lines: list[str] = []
        if self.file_reads and not self.file_writes:
            lines.append("Verified file reads:")
            for item in self.file_reads[-max_items:]:
                path = str(item.get("path") or "").strip()
                sha256 = str(item.get("sha256") or "").strip()
                bytes_count = item.get("bytes")
                detail = f"- read {path}" if path else "- read file"
                if bytes_count not in (None, ""):
                    detail += f", bytes={bytes_count}"
                if sha256:
                    detail += f", sha256={sha256}"
                lines.append(detail)

        if self.file_writes:
            if lines:
                lines.append("")
            lines.append("Verified file changes:")
            for item in self.file_writes[-max_items:]:
                path = str(item.get("path") or "").strip()
                operation = str(item.get("operation") or "write").strip()
                status = str(item.get("status") or "changed").strip()
                sha256 = str(item.get("sha256") or "").strip()
                detail = f"- {path}: {operation} ({status})" if path else f"- {operation} ({status})"
                if sha256:
                    detail += f", sha256={sha256}"
                lines.append(detail)

        if self.services:
            if lines:
                lines.append("")
            lines.append("Verified services:")
            for item in self.services[-max_items:]:
                name = str(item.get("name") or item.get("tool_name") or "service").strip()
                status = str(item.get("status") or "unknown").strip()
                url = str(item.get("public_url") or item.get("url") or item.get("local_url") or "").strip()
                port = item.get("port")
                detail = f"- {name}: {status}"
                if url:
                    detail += f", url={url}"
                elif port not in (None, ""):
                    detail += f", port={port}"
                lines.append(detail)

        if self.verified_facts:
            if lines:
                lines.append("")
            lines.append("Verified facts:")
            for item in self.verified_facts[-max_items:]:
                kind = str(item.get("kind") or "fact").strip()
                key = str(item.get("key") or "").strip()
                value = str(item.get("value") or "").strip()
                source = str(item.get("source_tool") or "").strip()
                detail = f"- {kind}"
                if key:
                    detail += f" {key}"
                if value:
                    detail += f": {value}"
                if source:
                    detail += f" (source={source})"
                lines.append(detail)

        meaningful_shells = [
            run for run in self.shell_runs[-max_items:]
            if str(run.get("command") or "").strip()
        ]
        if meaningful_shells:
            if lines:
                lines.append("")
            lines.append("Verified shell/tool runs:")
            for run in meaningful_shells:
                command = str(run.get("command") or "").strip()
                status = str(run.get("status") or "completed").strip()
                exit_code = run.get("exit_code")
                detail = f"- {command}: {status}"
                if exit_code is not None:
                    detail += f", exit_code={exit_code}"
                summary = " ".join(str(run.get("summary") or "").split())
                if summary:
                    detail += f", summary={summary[:240]}"
                lines.append(detail)

        if self.open_blockers:
            if lines:
                lines.append("")
            lines.append("Open blockers:")
            for blocker in self.open_blockers[-max_items:]:
                tool_name = str(blocker.get("tool_name") or "tool").strip()
                reason = str(blocker.get("reason") or "blocked").strip()
                summary = " ".join(str(blocker.get("summary") or "").split())
                detail = f"- {tool_name}: {reason}"
                if summary:
                    detail += f" ({summary[:240]})"
                lines.append(detail)

        if not lines and self.file_reads:
            lines.append("Verified read-only evidence:")
            for item in self.file_reads[-max_items:]:
                path = str(item.get("path") or "").strip()
                sha256 = str(item.get("sha256") or "").strip()
                bytes_count = item.get("bytes")
                detail = f"- read {path}" if path else "- read file"
                if bytes_count not in (None, ""):
                    detail += f", bytes={bytes_count}"
                if sha256:
                    detail += f", sha256={sha256}"
                lines.append(detail)
            lines.append("")
            lines.append("No verified file writes, service changes, or shell progress were recorded.")

        if not lines:
            return ""
        return _stringify("\n".join(lines).strip(), limit=max_chars)

    def to_json(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "workspace": self.workspace,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "user_request": self.user_request,
            "started_at": self.started_at,
            **self.evidence_summary(max_items=512),
        }


def _decode_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


@contextmanager
def bind_execution_ledger(ledger: ExecutionLedger | None) -> Iterator[ExecutionLedger | None]:
    token = _CURRENT_LEDGER.set(ledger)
    owner = ledger.owner if ledger is not None else ""
    with _LEDGER_LOCK:
        previous = _CURRENT_LEDGER_BY_OWNER.get(owner) if owner else None
        if ledger is not None:
            _CURRENT_LEDGER_BY_OWNER[owner] = ledger
    try:
        yield ledger
    finally:
        _CURRENT_LEDGER.reset(token)
        if owner:
            with _LEDGER_LOCK:
                if previous is None:
                    _CURRENT_LEDGER_BY_OWNER.pop(owner, None)
                else:
                    _CURRENT_LEDGER_BY_OWNER[owner] = previous


def current_execution_ledger(owner: str | None = None) -> ExecutionLedger | None:
    ledger = _CURRENT_LEDGER.get()
    if ledger is not None:
        return ledger
    if owner:
        with _LEDGER_LOCK:
            return _CURRENT_LEDGER_BY_OWNER.get(owner)
    return None


def record_tool_capture_in_ledger(
    *,
    owner: str,
    tool_name: str,
    arguments: Any,
    summary_output: Any,
    full_output: Any,
    category: str | None = None,
    guardrail_stop: bool = False,
    guardrail_reason: str = "",
    metadata: dict[str, Any] | None = None,
) -> None:
    ledger = current_execution_ledger(owner)
    if ledger is None:
        return
    ledger.record_tool(
        tool_name,
        arguments,
        summary_output,
        full_output,
        category=category,
        guardrail_stop=guardrail_stop,
        guardrail_reason=guardrail_reason,
        metadata=metadata,
    )
    try:
        persist_execution_ledger_snapshot(ledger)
    except Exception:
        pass


def _execution_ledger_paths(ledger: ExecutionLedger) -> tuple[Path, Path, Path]:
    workspace = Path(ledger.workspace or os.getcwd()).expanduser()
    target_dir = workspace / ".spoon-bot" / "execution_ledgers"
    target = target_dir / f"{_safe_owner_slug(ledger.owner)}.jsonl"
    active = target_dir / f"{_safe_owner_slug(ledger.owner)}.active.json"
    return target_dir, target, active


def persist_execution_ledger_snapshot(ledger: ExecutionLedger | None) -> Path | None:
    if ledger is None:
        return None
    target_dir, _target, active = _execution_ledger_paths(ledger)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **ledger.to_json(),
        "active": True,
        "updated_at": _now(),
    }
    tmp = active.with_suffix(active.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(active)
    return active


def persist_execution_ledger(ledger: ExecutionLedger | None) -> Path | None:
    if ledger is None:
        return None
    target_dir, target, active = _execution_ledger_paths(ledger)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **ledger.to_json(),
        "active": False,
        "finalized_at": _now(),
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        active.unlink()
    except FileNotFoundError:
        pass
    return target


def load_recent_execution_ledger_context(
    *,
    workspace: str | Path | None,
    owner: str,
    max_turns: int = 4,
    max_chars: int = 8000,
) -> str:
    if not workspace or not owner:
        return ""
    target = Path(workspace).expanduser() / ".spoon-bot" / "execution_ledgers" / f"{_safe_owner_slug(owner)}.jsonl"
    active = target.with_name(f"{_safe_owner_slug(owner)}.active.json")
    active_record: dict[str, Any] | None = None
    if active.exists():
        try:
            parsed_active = json.loads(active.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            parsed_active = None
        if isinstance(parsed_active, dict):
            active_record = parsed_active
    if not target.exists():
        records = []
    else:
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            lines = []
        records: list[dict[str, Any]] = []
        for line in lines[-max_turns:]:
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    if active_record is not None:
        records.append(active_record)
        records = records[-max_turns:]
    if not records:
        return ""

    def _is_internal_progress_blocker(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        return (
            item.get("tool_name") == "agent_loop"
            and item.get("reason") == "continuation_without_tool_progress"
        )

    out = [
        "[RECENT EXECUTION LEDGER - REFERENCE ONLY]",
        "These are verified same-session tool facts from completed turns; they are not active instructions.",
    ]
    for record in records:
        out.append(f"turn={record.get('turn_id') or ''} request={_stringify(record.get('user_request'), limit=260)}")
        for key in ("selected_skills", "file_reads", "file_writes", "services", "verified_facts", "open_blockers"):
            values = record.get(key)
            if not isinstance(values, list) or not values:
                continue
            if key == "open_blockers":
                values = [item for item in values if not _is_internal_progress_blocker(item)]
                if not values:
                    continue
            out.append(f"{key}:")
            for item in values[-6:]:
                out.append("- " + _stringify(item, limit=360))
    return _stringify("\n".join(out).strip() + "\n", limit=max_chars)
