"""Feishu/Lark permission policy helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

DM_POLICY_OPEN = "open"
DM_POLICY_ALLOWLIST = "allowlist"
DM_POLICY_DISABLED = "disabled"
DM_POLICY_PAIRING = "pairing"

GROUP_POLICY_OPEN = "open"
GROUP_POLICY_ALLOWLIST = "allowlist"
GROUP_POLICY_DISABLED = "disabled"

GROUP_SESSION_SCOPE_GROUP = "group"
GROUP_SESSION_SCOPE_GROUP_SENDER = "group_sender"
GROUP_SESSION_SCOPE_GROUP_TOPIC = "group_topic"
GROUP_SESSION_SCOPE_GROUP_TOPIC_SENDER = "group_topic_sender"
GROUP_SESSION_SCOPES = {
    GROUP_SESSION_SCOPE_GROUP,
    GROUP_SESSION_SCOPE_GROUP_SENDER,
    GROUP_SESSION_SCOPE_GROUP_TOPIC,
    GROUP_SESSION_SCOPE_GROUP_TOPIC_SENDER,
}


@dataclass(frozen=True)
class FeishuAccessDecision:
    """Resolved access policy for one inbound Feishu message."""

    allowed: bool
    is_direct_message: bool
    reason: str | None = None
    pairing_required: bool = False
    require_mention: bool = False
    dm_policy: str = DM_POLICY_OPEN
    group_policy: str = GROUP_POLICY_ALLOWLIST
    group_session_scope: str = GROUP_SESSION_SCOPE_GROUP_SENDER
    group_config: dict[str, Any] = field(default_factory=dict)
    dm_config: dict[str, Any] = field(default_factory=dict)


def normalize_feishu_allow_entry(raw: Any) -> str:
    """Normalize a single Feishu/Lark allowlist entry."""
    text = str(raw or "").strip()
    if not text:
        return ""
    if text == "*":
        return "*"
    if ":" in text:
        prefix, remainder = text.split(":", 1)
        if prefix.strip().lower() == "feishu":
            text = remainder
    return text.strip().lower()


def normalize_feishu_allowlist(values: Iterable[Any] | None) -> tuple[str, ...]:
    """Normalize and de-duplicate a Feishu allowlist."""
    if values is None:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        entry = normalize_feishu_allow_entry(value)
        if not entry or entry in seen:
            continue
        normalized.append(entry)
        seen.add(entry)
    return tuple(normalized)


def feishu_allowlist_allows(
    allow_from: Iterable[Any] | None,
    *,
    candidates: Iterable[Any],
) -> bool:
    """Return True when any candidate is matched by the allowlist."""
    normalized_allow_from = set(normalize_feishu_allowlist(allow_from))
    if not normalized_allow_from:
        return False
    if "*" in normalized_allow_from:
        return True
    normalized_candidates = normalize_feishu_allowlist(candidates)
    return any(candidate in normalized_allow_from for candidate in normalized_candidates)


def resolve_feishu_group_config(
    groups: dict[str, Any] | None,
    group_id: str | None,
) -> dict[str, Any]:
    """Resolve a per-group config by exact, case-insensitive, or wildcard key."""
    if not isinstance(groups, dict) or not groups:
        return {}
    group_id = str(group_id or "").strip()
    if not group_id:
        return {}
    direct = groups.get(group_id)
    if isinstance(direct, dict):
        return direct
    lowered = group_id.lower()
    for key, value in groups.items():
        if isinstance(key, str) and key.lower() == lowered and isinstance(value, dict):
            return value
    wildcard = groups.get("*")
    return wildcard if isinstance(wildcard, dict) else {}


def resolve_feishu_dm_config(
    dms: dict[str, Any] | None,
    sender_candidates: Iterable[Any],
) -> dict[str, Any]:
    """Resolve a per-DM config by sender id candidate or wildcard key."""
    if not isinstance(dms, dict) or not dms:
        return {}
    normalized_candidates = normalize_feishu_allowlist(sender_candidates)
    for candidate in normalized_candidates:
        direct = dms.get(candidate)
        if isinstance(direct, dict):
            return direct
        for key, value in dms.items():
            if isinstance(key, str) and key.lower() == candidate and isinstance(value, dict):
                return value
    wildcard = dms.get("*")
    return wildcard if isinstance(wildcard, dict) else {}


def resolve_feishu_reply_policy(
    *,
    is_direct_message: bool,
    group_policy: str,
    global_require_mention: bool | None,
    group_config: dict[str, Any] | None,
) -> bool:
    """Resolve whether the bot must be explicitly mentioned for this message."""
    if is_direct_message:
        return False
    if isinstance(group_config, dict) and isinstance(group_config.get("require_mention"), bool):
        return bool(group_config["require_mention"])
    if isinstance(global_require_mention, bool):
        return global_require_mention
    return group_policy != GROUP_POLICY_OPEN


def resolve_feishu_group_session_scope(
    default_scope: str,
    group_config: dict[str, Any] | None,
) -> str:
    """Resolve the effective group session scope for a chat."""
    if isinstance(group_config, dict):
        override = str(group_config.get("group_session_scope", "") or "").strip().lower()
        if override in GROUP_SESSION_SCOPES:
            return override
    if default_scope in GROUP_SESSION_SCOPES:
        return default_scope
    return GROUP_SESSION_SCOPE_GROUP_SENDER


def resolve_feishu_access(
    *,
    sender_candidates: Iterable[Any],
    chat_id: str,
    is_direct_message: bool,
    mentioned_bot: bool,
    allowed_users: Iterable[Any] | None,
    allowed_chats: Iterable[Any] | None,
    dm_policy: str,
    allow_from: Iterable[Any] | None,
    pairing_allow_from: Iterable[Any] | None,
    group_policy: str,
    group_allow_from: Iterable[Any] | None,
    group_sender_allow_from: Iterable[Any] | None,
    groups: dict[str, Any] | None,
    dms: dict[str, Any] | None,
    require_mention: bool | None,
    group_session_scope: str,
) -> FeishuAccessDecision:
    """Resolve the effective access decision for one inbound Feishu message."""
    sender_candidates = normalize_feishu_allowlist(sender_candidates)
    group_config = resolve_feishu_group_config(groups, chat_id) if not is_direct_message else {}
    dm_config = resolve_feishu_dm_config(dms, sender_candidates) if is_direct_message else {}

    if allowed_users and not feishu_allowlist_allows(allowed_users, candidates=sender_candidates):
        return FeishuAccessDecision(
            allowed=False,
            is_direct_message=is_direct_message,
            reason="sender not in allowed_users",
            group_config=group_config,
            dm_config=dm_config,
        )

    if allowed_chats and not feishu_allowlist_allows(allowed_chats, candidates=[chat_id]):
        return FeishuAccessDecision(
            allowed=False,
            is_direct_message=is_direct_message,
            reason="chat not in allowed_chats",
            group_config=group_config,
            dm_config=dm_config,
        )

    if is_direct_message:
        if dm_config.get("enabled") is False:
            return FeishuAccessDecision(
                allowed=False,
                is_direct_message=True,
                dm_policy=dm_policy,
                reason="dm disabled by per-sender config",
                dm_config=dm_config,
            )

        if dm_policy == DM_POLICY_DISABLED:
            return FeishuAccessDecision(
                allowed=False,
                is_direct_message=True,
                dm_policy=dm_policy,
                reason="dm policy disabled",
                dm_config=dm_config,
            )

        effective_dm_allow_from = tuple(
            dict.fromkeys(
                list(normalize_feishu_allowlist(allow_from))
                + list(normalize_feishu_allowlist(pairing_allow_from))
            )
        )
        if dm_policy in {DM_POLICY_ALLOWLIST, DM_POLICY_PAIRING} and not feishu_allowlist_allows(
            effective_dm_allow_from,
            candidates=sender_candidates,
        ):
            return FeishuAccessDecision(
                allowed=False,
                is_direct_message=True,
                dm_policy=dm_policy,
                reason="sender not in dm allowlist",
                pairing_required=dm_policy == DM_POLICY_PAIRING,
                dm_config=dm_config,
            )

        return FeishuAccessDecision(
            allowed=True,
            is_direct_message=True,
            dm_policy=dm_policy,
            require_mention=False,
            dm_config=dm_config,
        )

    if group_config.get("enabled") is False:
        return FeishuAccessDecision(
            allowed=False,
            is_direct_message=False,
            group_policy=group_policy,
            reason="group disabled by per-chat config",
            group_config=group_config,
        )

    if group_policy == GROUP_POLICY_DISABLED:
        return FeishuAccessDecision(
            allowed=False,
            is_direct_message=False,
            group_policy=group_policy,
            reason="group policy disabled",
            group_config=group_config,
        )

    if group_policy == GROUP_POLICY_ALLOWLIST and not feishu_allowlist_allows(
        group_allow_from,
        candidates=[chat_id],
    ):
        return FeishuAccessDecision(
            allowed=False,
            is_direct_message=False,
            group_policy=group_policy,
            reason="chat not in group allowlist",
            group_config=group_config,
        )

    per_group_sender_allow_from = group_config.get("allow_from")
    effective_group_sender_allow_from = (
        per_group_sender_allow_from
        if isinstance(per_group_sender_allow_from, list) and per_group_sender_allow_from
        else group_sender_allow_from
    )
    if effective_group_sender_allow_from and not feishu_allowlist_allows(
        effective_group_sender_allow_from,
        candidates=sender_candidates,
    ):
        return FeishuAccessDecision(
            allowed=False,
            is_direct_message=False,
            group_policy=group_policy,
            reason="sender not in group sender allowlist",
            group_config=group_config,
        )

    effective_require_mention = resolve_feishu_reply_policy(
        is_direct_message=False,
        group_policy=group_policy,
        global_require_mention=require_mention,
        group_config=group_config,
    )
    if effective_require_mention and not mentioned_bot:
        return FeishuAccessDecision(
            allowed=False,
            is_direct_message=False,
            group_policy=group_policy,
            require_mention=True,
            reason="bot not mentioned",
            group_config=group_config,
            group_session_scope=resolve_feishu_group_session_scope(
                group_session_scope,
                group_config,
            ),
        )

    return FeishuAccessDecision(
        allowed=True,
        is_direct_message=False,
        group_policy=group_policy,
        require_mention=effective_require_mention,
        group_config=group_config,
        group_session_scope=resolve_feishu_group_session_scope(
            group_session_scope,
            group_config,
        ),
    )
