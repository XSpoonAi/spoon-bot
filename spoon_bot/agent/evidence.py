"""Structured evidence and claim-grounding primitives.

The model may choose actions and wording, but dynamic facts must originate in
these records. The module intentionally uses generic tool/output structure and
does not route on product or game names.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 2

_KEY_VALUE_RE = re.compile(
    r"(?<![\w.-])([A-Za-z][A-Za-z0-9_.-]{0,79})\s*[=:]\s*"
    r"([^\s,;|]+)"
)
_HEX_ID_RE = re.compile(r"\b0x[0-9a-fA-F]{8,}\b")
_URL_RE = re.compile(r"https?://[^\s)\]>]+")
_NUMERIC_RE = re.compile(r"(?<![0-9A-Za-z_])[-+]?\d+(?:[.,]\d+)?%?(?![0-9A-Za-z_])")
_OUTCOMES = frozenset({"win", "loss", "lose", "draw", "refund", "pending", "unknown"})
_NON_TERMINAL_STATES = frozenset({"pending", "started", "waiting", "revealing", "settling", "running"})
_TERMINAL_STATES = frozenset({"completed", "complete", "finished", "settled", "confirmed", "finalized", "refunded"})
_COMPLETION_CLAIM_RE = re.compile(
    r"(?i)\b(?:completed|complete|finished|settled|registered|joined|deployed|finalized|confirmed)\b"
    r"|(?:已完成|已结算|已注册|已加入|已部署|已确认|已结束)"
)
_OUTCOME_CLAIM_PATTERNS = {
    "win": re.compile(r"(?i)\b(?:win|won|winner)\b|(?:获胜|赢了|胜出)"),
    "loss": re.compile(r"(?i)\b(?:loss|lose|lost)\b|(?:输了|落败)"),
    "draw": re.compile(r"(?i)\bdraw\b|(?:平局)"),
    "refund": re.compile(r"(?i)\brefund(?:ed)?\b|(?:退款|退回)"),
    "pending": re.compile(r"(?i)\bpending\b|(?:待处理|未结算)"),
}

_GAME_FIELD_ALIASES = {
    "game": "game_id",
    "gameid": "game_id",
    "game_id": "game_id",
    "type": "game_type",
    "gametype": "game_type",
    "game_type": "game_type",
    "phase": "phase",
    "outcome": "outcome",
    "result": "outcome",
    "action": "selected_action",
    "spot": "selected_action",
    "target": "selected_action",
    "score": "score",
    "rank": "rank",
    "players": "player_count",
    "playercount": "player_count",
    "player_count": "player_count",
    "cost": "entry_cost",
    "entry": "entry_cost",
    "entrycost": "entry_cost",
    "entry_cost": "entry_cost",
    "additionalcost": "additional_cost",
    "additional_cost": "additional_cost",
    "swapfee": "additional_cost",
    "reward": "reward",
    "prize": "reward",
    "pnl": "net_pnl",
    "netpnl": "net_pnl",
    "net_pnl": "net_pnl",
    "settlement": "settlement_status",
    "settlementstatus": "settlement_status",
    "settlement_status": "settlement_status",
    "tx": "transaction_hashes",
    "txhash": "transaction_hashes",
    "transaction": "transaction_hashes",
    "transactionhash": "transaction_hashes",
}


def _now() -> float:
    return time.time()


def _stable_id(*parts: Any) -> str:
    payload = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _normalize_number(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    raw = str(value).strip().replace(",", "")
    suffix = "%" if raw.endswith("%") else ""
    if suffix:
        raw = raw[:-1]
    try:
        number = Decimal(raw)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite():
        return None
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return (normalized or "0") + suffix


@dataclass(frozen=True)
class Fact:
    fact_id: str
    field: str
    value: Any
    unit: str = ""
    value_type: str = "string"
    derivation: str = "observed"
    source_evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_evidence_ids"] = list(self.source_evidence_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Fact":
        return cls(
            fact_id=str(payload.get("fact_id") or payload.get("id") or ""),
            field=str(payload.get("field") or payload.get("key") or payload.get("label") or ""),
            value=payload.get("value"),
            unit=str(payload.get("unit") or ""),
            value_type=str(payload.get("value_type") or payload.get("kind") or "string"),
            derivation=str(payload.get("derivation") or ("derived" if payload.get("derived") else "observed")),
            source_evidence_ids=tuple(payload.get("source_evidence_ids") or ()),
        )


@dataclass
class EvidenceRecord:
    evidence_id: str
    source_type: str
    source_name: str
    capability_id: str
    turn_id: str
    observed_at: float
    status: str
    freshness: str
    subject: str
    facts: list[Fact] = field(default_factory=list)
    output_hash: str = ""
    raw_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "facts": [fact.to_dict() for fact in self.facts],
        }

    def to_context_dict(self) -> dict[str, Any]:
        """Render provenance once at record level to avoid prompt duplication."""
        payload = self.to_dict()
        payload["facts"] = [
            {
                key: value
                for key, value in fact.to_dict().items()
                if key != "source_evidence_ids"
            }
            for fact in self.facts
        ]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRecord":
        return cls(
            evidence_id=str(payload.get("evidence_id") or ""),
            source_type=str(payload.get("source_type") or "tool"),
            source_name=str(payload.get("source_name") or payload.get("tool_name") or ""),
            capability_id=str(payload.get("capability_id") or payload.get("tool_name") or ""),
            turn_id=str(payload.get("turn_id") or ""),
            observed_at=float(payload.get("observed_at") or payload.get("recorded_at") or 0),
            status=str(payload.get("status") or "succeeded"),
            freshness=str(payload.get("freshness") or "session"),
            subject=str(payload.get("subject") or ""),
            facts=[Fact.from_dict(item) for item in payload.get("facts") or [] if isinstance(item, dict)],
            output_hash=str(payload.get("output_hash") or payload.get("output_sha256") or ""),
            raw_reference=str(payload.get("raw_reference") or ""),
        )


@dataclass
class GameSettlementFacts:
    game_id: str = "unknown"
    game_type: str = "unknown"
    phase: str = "unknown"
    outcome: str = "unknown"
    selected_action: str = "unknown"
    score: str = "unknown"
    rank: str = "unknown"
    player_count: str = "unknown"
    entry_cost: str = "unknown"
    additional_cost: str = "unknown"
    reward: str = "unknown"
    net_pnl: str = "unknown"
    settlement_status: str = "unknown"
    transaction_hashes: list[str] = field(default_factory=list)
    observed_at: float = 0
    subject: str = ""
    source_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnContextEnvelope:
    session_id: str
    turn_id: str
    current_request: str
    continuation_intent: str
    selected_workflow: str
    active_capabilities: list[str]
    workflow_state: dict[str, Any]
    evidence: list[EvidenceRecord]
    unresolved_blockers: list[dict[str, Any]]
    token_budget: dict[str, int]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "evidence": [record.to_dict() for record in self.evidence],
        }

    def render(self, *, max_chars: int = 12000) -> str:
        context_payload = self.to_dict()
        context_payload["evidence"] = [record.to_context_dict() for record in self.evidence]
        payload = json.dumps(context_payload, ensure_ascii=False, sort_keys=True, default=str)
        if len(payload) > max_chars:
            payload = payload[: max_chars - 32] + '..."truncated":true}'
        return "[TURN CONTEXT ENVELOPE]\n" + payload


def extract_facts_from_payload(payload: Any, *, evidence_id: str) -> list[Fact]:
    """Extract bounded scalar facts from JSON or key=value tool output."""
    values: dict[str, Any] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[str(key)] = value
    else:
        text = _stringify(payload)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return extract_facts_from_payload(parsed, evidence_id=evidence_id)
        for match in _KEY_VALUE_RE.finditer(text):
            values.setdefault(match.group(1), match.group(2).strip('"\''))

    facts: list[Fact] = []
    for field_name, raw_value in list(values.items())[:96]:
        normalized_number = _normalize_number(raw_value)
        value_type = "number" if normalized_number is not None else "string"
        value = normalized_number if normalized_number is not None else raw_value
        fact_id = _stable_id(evidence_id, field_name.casefold(), value)
        facts.append(Fact(
            fact_id=fact_id,
            field=field_name,
            value=value,
            value_type=value_type,
            source_evidence_ids=(evidence_id,),
        ))
    return facts


def infer_freshness(facts: Iterable[Fact], *, status: str) -> str:
    fields = {fact.field.casefold().replace("-", "_") for fact in facts}
    if status != "succeeded":
        return "stale"
    if fields & {"balance", "phase", "status", "state", "job_status"}:
        return "live"
    if fields & {"settlement_id", "settlement_status", "tx", "txhash", "transaction_hash"}:
        return "stable"
    return "session"


def evidence_from_tool_result(
    *,
    tool_name: str,
    output: Any,
    turn_id: str = "",
    status: str = "succeeded",
    category: str = "",
    observed_at: float | None = None,
    raw_reference: str = "",
) -> EvidenceRecord:
    text = _stringify(output)
    output_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    evidence_id = _stable_id(turn_id, tool_name, output_hash)
    facts = extract_facts_from_payload(output, evidence_id=evidence_id)
    fact_map = {fact.field.casefold().replace("-", "_"): str(fact.value) for fact in facts}
    subject = (
        fact_map.get("game_id")
        or fact_map.get("gameid")
        or fact_map.get("game")
        or fact_map.get("wallet")
        or fact_map.get("address")
        or ""
    )
    source_type = "tool"
    lowered_category = str(category or "").casefold()
    if lowered_category in {"chain", "backend", "file"}:
        source_type = lowered_category
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_type=source_type,
        source_name=tool_name,
        capability_id=tool_name,
        turn_id=turn_id,
        observed_at=observed_at or _now(),
        status=status,
        freshness=infer_freshness(facts, status=status),
        subject=subject,
        facts=facts,
        output_hash=output_hash,
        raw_reference=raw_reference,
    )


def deduplicate_evidence(records: Iterable[EvidenceRecord]) -> list[EvidenceRecord]:
    """Keep the latest highest-authority copy of each evidence/output digest."""
    authority = {"chain": 6, "tool": 5, "backend": 4, "file": 3, "user": 2, "derived": 1}
    selected: dict[str, EvidenceRecord] = {}
    for record in records:
        key = record.output_hash or record.evidence_id
        previous = selected.get(key)
        if previous is None or (
            authority.get(record.source_type, 0), record.observed_at
        ) > (
            authority.get(previous.source_type, 0), previous.observed_at
        ):
            selected[key] = record
    return sorted(selected.values(), key=lambda item: item.observed_at, reverse=True)


def game_settlement_from_evidence(records: Iterable[EvidenceRecord]) -> list[GameSettlementFacts]:
    """Build per-subject settlement views without combining unrelated games."""
    grouped: dict[str, list[EvidenceRecord]] = {}
    for record in deduplicate_evidence(records):
        if record.status != "succeeded":
            continue
        values = {fact.field.casefold().replace("-", "_"): fact.value for fact in record.facts}
        game_id = str(values.get("game_id") or values.get("gameid") or values.get("game") or "")
        if not game_id:
            continue
        wallet = str(values.get("wallet") or values.get("address") or "")
        settlement = str(values.get("settlement_id") or "")
        subject = "|".join((game_id, wallet, settlement)).strip("|")
        grouped.setdefault(subject, []).append(record)

    settlements: list[GameSettlementFacts] = []
    for subject, subject_records in grouped.items():
        result = GameSettlementFacts(subject=subject)
        for record in sorted(subject_records, key=lambda item: item.observed_at):
            result.observed_at = max(result.observed_at, record.observed_at)
            result.source_evidence_ids.append(record.evidence_id)
            for fact in record.facts:
                normalized = fact.field.casefold().replace("-", "_").replace(".", "")
                target = _GAME_FIELD_ALIASES.get(normalized)
                if not target:
                    continue
                value = str(fact.value)
                if target == "outcome":
                    normalized_outcome = value.casefold()
                    value = "loss" if normalized_outcome == "lose" else normalized_outcome
                    if value not in _OUTCOMES:
                        value = "unknown"
                if target == "transaction_hashes":
                    if value not in result.transaction_hashes:
                        result.transaction_hashes.append(value)
                else:
                    setattr(result, target, value)
        settlements.append(result)
    return settlements


def apply_freshness_policy(
    records: Iterable[EvidenceRecord],
    *,
    current_turn_id: str,
    now: float | None = None,
    backend_ttl_seconds: float = 300,
) -> list[EvidenceRecord]:
    """Mark dynamic and backend facts stale when their reuse window has expired."""
    checked_at = now or _now()
    refreshed: list[EvidenceRecord] = []
    for original in deduplicate_evidence(records):
        record = EvidenceRecord.from_dict(original.to_dict())
        if record.status != "succeeded":
            record.freshness = "stale"
        elif record.freshness == "live" and (
            not current_turn_id or record.turn_id != current_turn_id
        ):
            record.freshness = "stale"
        elif record.source_type == "backend" and (
            checked_at - record.observed_at > backend_ttl_seconds
        ):
            record.freshness = "stale"
        elif record.source_type == "file" and record.raw_reference:
            path = Path(record.raw_reference).expanduser()
            if not path.exists():
                record.freshness = "stale"
            else:
                fact_map = {fact.field.casefold(): str(fact.value) for fact in record.facts}
                expected_mtime = fact_map.get("mtime")
                if expected_mtime and expected_mtime != str(path.stat().st_mtime):
                    record.freshness = "stale"
        refreshed.append(record)
    return refreshed


def aggregate_settled_games(records: Iterable[EvidenceRecord]) -> dict[str, str | int]:
    """Calculate totals only from deduplicated terminal settlement subjects."""
    settlements = game_settlement_from_evidence(records)
    terminal = [
        item for item in settlements
        if item.outcome in {"win", "loss", "draw", "refund"}
        and item.settlement_status.casefold() not in _NON_TERMINAL_STATES
        and item.phase.casefold() not in _NON_TERMINAL_STATES
    ]
    totals = {"settled_games": len(terminal), "entry_cost": Decimal(0), "additional_cost": Decimal(0), "reward": Decimal(0), "net_pnl": Decimal(0)}
    for item in terminal:
        for field_name in ("entry_cost", "additional_cost", "reward", "net_pnl"):
            normalized = _normalize_number(getattr(item, field_name))
            if normalized is not None and not normalized.endswith("%"):
                totals[field_name] += Decimal(normalized)
    return {
        key: (format(value.normalize(), "f") if isinstance(value, Decimal) else value)
        for key, value in totals.items()
    }


@dataclass
class ClaimValidationResult:
    valid: bool
    rejected_claims: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    fallback_reason: str = ""


def validate_claims(
    draft: str,
    records: Iterable[EvidenceRecord],
    *,
    unresolved_blockers: Iterable[dict[str, Any]] = (),
) -> ClaimValidationResult:
    """Validate high-risk literal claims against a closed evidence set."""
    evidence = [
        record for record in deduplicate_evidence(records)
        if record.status == "succeeded" and record.freshness != "stale"
    ]
    authority = {"chain": 6, "tool": 5, "backend": 4, "file": 3, "user": 2, "derived": 1}
    selected_facts: dict[tuple[str, str], tuple[tuple[int, float], EvidenceRecord, Fact]] = {}
    for record in evidence:
        subject = record.subject or record.evidence_id
        for fact in record.facts:
            field_name = fact.field.casefold().replace("-", "_")
            key = (subject, field_name)
            priority = (authority.get(record.source_type, 0), record.observed_at)
            previous = selected_facts.get(key)
            if previous is None or priority > previous[0]:
                selected_facts[key] = (priority, record, fact)
    authoritative = [(record, fact) for _priority, record, fact in selected_facts.values()]
    evidence_text = "\n".join(
        f"{fact.field}={fact.value}" for record, fact in authoritative
    )
    rejected: list[str] = []
    if any(True for _ in unresolved_blockers):
        rejected.append("unresolved_blocker")

    supported_numbers = {match.group(0).replace(",", "").casefold() for match in _NUMERIC_RE.finditer(evidence_text)}
    draft_without_lists = re.sub(r"(?m)^\s*\d+[.)]\s+", "", draft)
    for match in _NUMERIC_RE.finditer(draft_without_lists):
        token = match.group(0).replace(",", "").casefold()
        if token not in supported_numbers:
            rejected.append(f"unsupported_numeric:{token}")

    for pattern, label in ((_HEX_ID_RE, "identifier"), (_URL_RE, "url")):
        supported = {match.group(0) for match in pattern.finditer(evidence_text)}
        for match in pattern.finditer(draft):
            if match.group(0) not in supported:
                rejected.append(f"unsupported_{label}:{match.group(0)}")

    supported_outcomes = {
        str(fact.value).casefold()
        for record, fact in authoritative
        if fact.field.casefold() in {"outcome", "result"}
    }
    normalized_claimed = {
        outcome for outcome, pattern in _OUTCOME_CLAIM_PATTERNS.items()
        if pattern.search(draft)
    }
    normalized_supported = {"loss" if token == "lose" else token for token in supported_outcomes}
    for outcome in normalized_claimed - normalized_supported:
        rejected.append(f"unsupported_outcome:{outcome}")

    if _COMPLETION_CLAIM_RE.search(draft):
        terminal_values = {
            str(fact.value).strip().casefold()
            for record, fact in authoritative
            if fact.field.casefold().replace("-", "_")
            in {"status", "state", "phase", "settlement", "settlement_status", "outcome"}
        }
        has_terminal_marker = any(
            any(marker in value for marker in _TERMINAL_STATES)
            for value in terminal_values
        )
        has_non_terminal_marker = any(
            any(marker in value for marker in _NON_TERMINAL_STATES)
            for value in terminal_values
        )
        if not has_terminal_marker or has_non_terminal_marker:
            rejected.append("unsupported_completion")

    rejected = list(dict.fromkeys(rejected))
    return ClaimValidationResult(
        valid=not rejected,
        rejected_claims=rejected,
        evidence_ids=[record.evidence_id for record in evidence],
        fallback_reason=";".join(rejected),
    )


def conservative_fact_summary(records: Iterable[EvidenceRecord]) -> str:
    """Render verified facts without adding semantic interpretation."""
    lines = ["Verified facts:"]
    for record in deduplicate_evidence(records):
        if record.status != "succeeded":
            continue
        for fact in record.facts:
            unit = f" {fact.unit}" if fact.unit else ""
            lines.append(f"- {fact.field}: {fact.value}{unit}")
    return "\n".join(lines) if len(lines) > 1 else "No verified facts are available."


def render_game_settlement_summary(records: Iterable[EvidenceRecord]) -> str:
    """Render isolated game subjects without inferring unknown fields or causes."""
    settlements = game_settlement_from_evidence(records)
    if not settlements:
        return conservative_fact_summary(records)
    lines = ["Verified game facts:"]
    fields = (
        "game_id", "game_type", "phase", "outcome", "selected_action", "score",
        "rank", "player_count", "entry_cost", "additional_cost", "reward",
        "net_pnl", "settlement_status",
    )
    for settlement in settlements:
        lines.append(f"- subject: {settlement.subject or 'unknown'}")
        for field_name in fields:
            lines.append(f"  {field_name}: {getattr(settlement, field_name)}")
        if settlement.transaction_hashes:
            lines.append(f"  transaction_hashes: {', '.join(settlement.transaction_hashes)}")
    return "\n".join(lines)


def write_claim_validation_audit(
    workspace: str | Path | None,
    *,
    session_id: str,
    turn_id: str,
    result: ClaimValidationResult,
    shadow_mode: bool,
) -> Path | None:
    """Persist a privacy-safe validator decision without raw prompts or outputs."""
    if workspace is None:
        return None
    target_dir = Path(workspace).expanduser() / ".spoon-bot" / "claim_validation"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "audit.jsonl"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "turn_id": turn_id,
        "observed_at": _now(),
        "valid": result.valid,
        "evidence_ids": result.evidence_ids,
        "rejected_claims": result.rejected_claims,
        "fallback_reason": result.fallback_reason,
        "shadow_mode": shadow_mode,
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return target
