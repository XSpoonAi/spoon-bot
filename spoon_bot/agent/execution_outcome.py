"""Generic, evidence-driven terminal outcomes for one agent request."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExecutionOutcome:
    """A domain-neutral terminal result; absent evidence stays absent."""

    status: str
    reason: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    job_id: str = ""
    elapsed_seconds: float | None = None
    last_progress: str = ""
    retryable: bool | None = None
    resume_hint: str = ""
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value not in ("", None, [])
        }

    def user_message(self) -> str:
        labels = {
            "completed": "任务已完成。",
            "failed": "任务未完成：工具执行失败。",
            "cancelled": "任务已取消。",
            "timed_out": "任务未完成：工具执行超时。",
            "interrupted": "任务已中断。",
        }
        lines = [labels.get(self.status, "任务未完成。")]
        if self.tool_name:
            lines.append(f"工具：{self.tool_name}")
        if self.last_progress:
            lines.append(f"最后进度：{self.last_progress}")
        if self.reason:
            lines.append(f"原因：{self.reason}")
        return "\n".join(lines)


def outcome_from_tool_event(event: dict[str, Any]) -> ExecutionOutcome | None:
    """Extract only explicitly structured terminal metadata from a tool event."""
    metadata = dict(event.get("metadata") or {})
    status = str(metadata.get("status") or event.get("status") or "").strip().casefold()
    terminal = metadata.get("terminal") is True or metadata.get("terminate") is True
    if not terminal and status not in {"failed", "error", "cancelled", "canceled", "timed_out"}:
        return None
    if status in {"error", "failed"}:
        status = "failed"
    elif status in {"canceled", "cancelled"}:
        status = "cancelled"
    elif status == "timed_out":
        status = "timed_out"
    else:
        status = "completed"
    return ExecutionOutcome(
        status=status,
        reason=str(metadata.get("reason") or metadata.get("error") or "").strip(),
        tool_name=str(metadata.get("name") or metadata.get("tool") or "").strip(),
        tool_call_id=str(metadata.get("tool_call_id") or event.get("tool_call_id") or "").strip(),
        job_id=str(metadata.get("job_id") or "").strip(),
        last_progress=str(
            metadata.get("progress")
            or metadata.get("output_summary")
            or metadata.get("output")
            or metadata.get("result")
            or ""
        ).strip(),
        retryable=metadata.get("retryable") if isinstance(metadata.get("retryable"), bool) else None,
        resume_hint=str(metadata.get("resume_hint") or "").strip(),
        evidence_ids=[str(item) for item in metadata.get("evidence_ids") or [] if str(item).strip()],
    )
