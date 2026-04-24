"""Shared channel delivery binding helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable

from loguru import logger

from spoon_bot.bus.events import InboundMessage, OutboundMessage
from spoon_bot.cron.models import CronConversationScope

if TYPE_CHECKING:
    from spoon_bot.channels.base import BaseChannel
    from spoon_bot.session.manager import Session


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_target(target: dict[str, Any] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in (target or {}).items():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            result[str(key)] = text
    return result


def _split_channel_parts(
    channel: str | None,
    account_id: str | None = None,
) -> tuple[str | None, str | None]:
    channel_text = str(channel or "").strip()
    account_text = str(account_id or "").strip() or None
    if not channel_text:
        return None, account_text
    if ":" in channel_text:
        channel_type, derived_account = channel_text.split(":", 1)
        return channel_type, account_text or (derived_account or None)
    return channel_text, account_text


def normalize_channel_name(
    channel: str | None,
    account_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Return a canonical channel name and effective account id."""
    channel_text = str(channel or "").strip()
    if not channel_text:
        return None, str(account_id or "").strip() or None
    _, effective_account = _split_channel_parts(channel_text, account_id)
    if ":" in channel_text:
        return channel_text, effective_account
    if effective_account:
        return f"{channel_text}:{effective_account}", effective_account
    return channel_text, effective_account


def conversation_scope_from_parts(
    *,
    channel: str | None,
    account_id: str | None = None,
    target: dict[str, Any] | None = None,
    session_key: str | None = None,
) -> CronConversationScope | None:
    """Build a stable conversation scope from delivery-like parts."""
    channel_type, effective_account = _split_channel_parts(channel, account_id)
    compact_target = _compact_target(target)
    if channel_type == "telegram":
        conversation_id = compact_target.get("chat_id")
    elif channel_type == "discord":
        conversation_id = compact_target.get("channel_id")
    else:
        conversation_id = (
            compact_target.get("conversation_id")
            or compact_target.get("chat_id")
            or compact_target.get("channel_id")
            or compact_target.get("target_id")
        )
    if not channel_type or not conversation_id:
        return None

    thread_id = (
        compact_target.get("thread_id")
        or compact_target.get("topic_id")
        or compact_target.get("message_thread_id")
    )
    return CronConversationScope(
        channel=channel_type,
        account_id=effective_account,
        conversation_id=conversation_id,
        thread_id=thread_id,
        session_key=str(session_key).strip() or None if session_key else None,
    )


@dataclass(slots=True)
class DeliveryBinding:
    """Resolved outbound delivery target for a session or cron job."""

    channel: str
    account_id: str | None = None
    session_key: str | None = None
    target: dict[str, str] = field(default_factory=dict)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_metadata(self) -> dict[str, Any]:
        """Serialize for session metadata persistence."""
        payload: dict[str, Any] = {
            "channel": self.channel,
            "target": dict(self.target),
            "updated_at": self.updated_at,
        }
        if self.account_id:
            payload["account_id"] = self.account_id
        if self.session_key:
            payload["session_key"] = self.session_key
        return payload

    def to_outbound_metadata(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build channel-specific outbound metadata."""
        metadata = dict(extra or {})
        metadata.setdefault("delivery_binding", self.to_metadata())

        if self.channel.startswith("telegram:"):
            chat_id = self.target.get("chat_id")
            if chat_id is not None:
                metadata.setdefault("chat_id", chat_id)
        elif self.channel.startswith("discord:"):
            channel_id = self.target.get("channel_id")
            if channel_id is not None:
                metadata.setdefault("channel_id", channel_id)

        return metadata

    @classmethod
    def from_metadata(cls, payload: dict[str, Any] | None) -> "DeliveryBinding | None":
        """Deserialize a persisted binding payload."""
        if not isinstance(payload, dict):
            return None

        channel = str(payload.get("channel") or "").strip()
        if not channel:
            return None

        channel, effective_account = normalize_channel_name(
            channel,
            payload.get("account_id"),
        )
        if channel is None:
            return None
        session_key = payload.get("session_key")
        return cls(
            channel=channel,
            account_id=effective_account,
            session_key=str(session_key).strip() if session_key else None,
            target=_compact_target(payload.get("target")),
            updated_at=str(payload.get("updated_at") or _utc_now_iso()),
        )


def binding_from_session_key(session_key: str | None) -> DeliveryBinding | None:
    """Best-effort fallback binding derived from a structured session key."""
    if not session_key or "_" not in session_key:
        return None

    parts = [segment for segment in session_key.split("_") if segment]
    if len(parts) < 3:
        return None

    channel_type = parts[0]
    account_id = "_".join(parts[1:-1]) or None
    target_id = parts[-1]
    if not account_id or not target_id:
        return None

    if channel_type == "telegram":
        return DeliveryBinding(
            channel=f"telegram:{account_id}",
            account_id=account_id,
            session_key=session_key,
            target={"chat_id": target_id},
        )
    if channel_type == "discord":
        return DeliveryBinding(
            channel=f"discord:{account_id}",
            account_id=account_id,
            session_key=session_key,
            target={"channel_id": target_id},
        )
    return None


def conversation_scope_from_binding(binding: DeliveryBinding | None) -> CronConversationScope | None:
    """Build a stable scope from a resolved delivery binding."""
    if binding is None:
        return None
    return conversation_scope_from_parts(
        channel=binding.channel,
        account_id=binding.account_id,
        target=binding.target,
        session_key=binding.session_key,
    )


def conversation_scope_from_session_key(
    session_key: str | None,
) -> CronConversationScope | None:
    """Best-effort fallback scope derived from a structured session key."""
    binding = binding_from_session_key(session_key)
    return conversation_scope_from_binding(binding)


def binding_from_inbound_message(message: InboundMessage) -> DeliveryBinding | None:
    """Extract a delivery binding from inbound channel metadata."""
    metadata = message.metadata or {}
    if message.channel.startswith("telegram:"):
        chat_id = metadata.get("chat_id")
        if chat_id is None:
            return None
        return DeliveryBinding(
            channel=message.channel,
            account_id=message.channel.split(":", 1)[1] if ":" in message.channel else None,
            session_key=message.session_key,
            target={"chat_id": str(chat_id)},
        )

    if message.channel.startswith("discord:"):
        channel_id = metadata.get("channel_id")
        if channel_id is None:
            return None
        return DeliveryBinding(
            channel=message.channel,
            account_id=message.channel.split(":", 1)[1] if ":" in message.channel else None,
            session_key=message.session_key,
            target={"channel_id": str(channel_id)},
        )

    return None


class ChannelDeliveryService:
    """Resolves persisted bindings and sends proactive outbound messages."""

    def __init__(
        self,
        channel_lookup: Callable[[str], "BaseChannel | None"] | None = None,
        channel_names_lookup: Callable[[], Iterable[str]] | None = None,
    ) -> None:
        self._channel_lookup = channel_lookup
        self._channel_names_lookup = channel_names_lookup

    def attach_channel_lookup(
        self,
        channel_lookup: Callable[[str], "BaseChannel | None"],
        channel_names_lookup: Callable[[], Iterable[str]] | None = None,
    ) -> None:
        """Attach channel resolution after manager initialization."""
        self._channel_lookup = channel_lookup
        self._channel_names_lookup = channel_names_lookup

    def build_binding(self, message: InboundMessage) -> DeliveryBinding | None:
        """Build a binding from an inbound message."""
        return binding_from_inbound_message(message)

    def resolve_binding(
        self,
        *,
        explicit: DeliveryBinding | dict[str, Any] | None = None,
        session: "Session | None" = None,
        session_key: str | None = None,
    ) -> DeliveryBinding | None:
        """Resolve the best outbound binding for a cron run."""
        candidates: list[DeliveryBinding | None] = []

        if isinstance(explicit, DeliveryBinding):
            candidates.append(explicit)
        elif isinstance(explicit, dict):
            candidates.append(DeliveryBinding.from_metadata(explicit))

        if session is not None and isinstance(getattr(session, "metadata", None), dict):
            candidates.append(
                DeliveryBinding.from_metadata(session.metadata.get("delivery_binding"))
            )

        candidates.append(binding_from_session_key(session_key))

        for candidate in candidates:
            if candidate is not None:
                return candidate
        return None

    async def deliver(
        self,
        content: str,
        binding: DeliveryBinding,
        *,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OutboundMessage:
        """Send an outbound message using a resolved binding."""
        if self._channel_lookup is None:
            raise RuntimeError("Channel lookup is not configured for delivery service")

        resolved_name, channel = self._resolve_channel(binding)
        if channel is None:
            raise ValueError(f"Channel not available for delivery: {binding.channel}")

        effective_binding = DeliveryBinding(
            channel=resolved_name,
            account_id=binding.account_id,
            session_key=binding.session_key,
            target=dict(binding.target),
        )
        outbound = OutboundMessage(
            content=content,
            channel=resolved_name,
            reply_to=reply_to,
            metadata=effective_binding.to_outbound_metadata(metadata),
        )
        await channel.send(outbound)
        logger.info(f"Delivered proactive message via {resolved_name}")
        return outbound

    def _resolve_channel(
        self,
        binding: DeliveryBinding,
    ) -> tuple[str, BaseChannel | None]:
        assert self._channel_lookup is not None

        for candidate in self._candidate_channel_names(binding):
            channel = self._channel_lookup(candidate)
            if channel is not None:
                return candidate, channel
        return binding.channel, None

    def _candidate_channel_names(self, binding: DeliveryBinding) -> list[str]:
        candidates: list[str] = []

        def _append(name: str | None) -> None:
            if name and name not in candidates:
                candidates.append(name)

        _append(binding.channel)

        normalized_name, effective_account = normalize_channel_name(
            binding.channel,
            binding.account_id,
        )
        _append(normalized_name)

        if self._channel_names_lookup is not None:
            try:
                known_names = list(self._channel_names_lookup())
            except Exception:
                known_names = []
            if ":" not in binding.channel:
                prefix = f"{binding.channel}:"
                if effective_account:
                    _append(f"{binding.channel}:{effective_account}")
                scoped_matches = [name for name in known_names if name.startswith(prefix)]
                if effective_account:
                    exact = f"{binding.channel}:{effective_account}"
                    if exact in scoped_matches:
                        scoped_matches = [exact]
                if len(scoped_matches) == 1:
                    _append(scoped_matches[0])

        return candidates
