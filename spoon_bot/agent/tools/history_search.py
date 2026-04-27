"""Conversation history search tool.

Fallback mechanism for the model to recover earlier tool calls, tool
results, and message bodies *after* runtime context has been compacted.

The tool delegates to the agent's :class:`~spoon_bot.session.manager.SessionManager`,
which in turn uses whatever the active session store can offer:

* ``PostgresSessionStore`` — native ``ILIKE`` / ``LIKE`` / POSIX regex
  filtering at the database layer.
* ``SQLiteSessionStore`` — ``LIKE`` / ``GLOB`` prefilter + Python ``re``
  refinement in memory.
* ``FileSessionStore`` (or any other backend) — in-memory scan of the
  JSONL transcripts (i.e. a grep fallback).

Because the underlying store is consulted (not runtime memory), results
remain available even when messages have been trimmed from the agent's
working context.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Callable, Iterable, Optional, TYPE_CHECKING

from spoon_bot.agent.tools.base import Tool

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from spoon_bot.session.manager import SessionManager


SessionKeyResolver = Callable[[], Optional[str]]


class SearchHistoryTool(Tool):
    """Search persisted conversation history (messages + tool calls/results).

    The model should reach for this tool whenever:

    * It's asked about an earlier turn that is no longer in context
      (e.g. "what did the previous X tool return?").
    * It needs to reconstruct a tool call chain after compaction.
    * The user references a prior topic the current context does not
      cover.
    """

    def __init__(
        self,
        sessions_manager: "SessionManager",
        *,
        default_session_key: str | None = None,
        session_key_resolver: SessionKeyResolver | None = None,
    ) -> None:
        """Create a new :class:`SearchHistoryTool`.

        Args:
            sessions_manager: The agent's :class:`SessionManager`.
            default_session_key: Optional fixed session key used when no
                resolver is provided or the resolver returns ``None``.
            session_key_resolver: Zero-arg callable that returns the
                *current* session key.  Preferred over
                ``default_session_key`` because the agent's active
                session can be switched out-of-band (e.g. by the WS
                gateway on every chat.send).  The resolver is consulted
                on every :meth:`execute` call, guaranteeing the tool
                always targets the live session rather than a stale
                cached value.
        """
        self._sessions_manager = sessions_manager
        self._default_session_key = default_session_key
        self._session_key_resolver = session_key_resolver

    def set_default_session_key(self, session_key: str | None) -> None:
        """Update which session this tool searches by default.

        Kept for backwards compatibility — prefer passing a
        ``session_key_resolver`` at construction time so switches
        happening outside :class:`AgentLoop` (e.g. in the WS gateway)
        are picked up automatically.
        """
        self._default_session_key = session_key

    def set_session_key_resolver(
        self, resolver: SessionKeyResolver | None
    ) -> None:
        """Swap the zero-arg resolver returning the active session key."""
        self._session_key_resolver = resolver

    def _resolve_active_session_key(self) -> str | None:
        """Return the session key the tool should default to right now."""
        if self._session_key_resolver is not None:
            try:
                resolved = self._session_key_resolver()
            except Exception:  # pragma: no cover - defensive
                resolved = None
            if resolved:
                return str(resolved)
        return self._default_session_key

    @staticmethod
    def _is_tool_call_trace_hit(hit: Any) -> bool:
        """Return True when an assistant hit represents a persisted tool call."""
        if str(getattr(hit, "role", "")).lower() != "assistant":
            return False
        extras = getattr(hit, "extras", None) or {}
        return bool(extras.get("tool_calls"))

    @classmethod
    def _is_plain_assistant_reply_hit(cls, hit: Any) -> bool:
        """Return True for assistant narrative/final replies that may be stale."""
        return (
            str(getattr(hit, "role", "")).lower() == "assistant"
            and not cls._is_tool_call_trace_hit(hit)
        )

    @staticmethod
    def _parse_timestamp(value: Any) -> float:
        """Best-effort timestamp parser used for client-side hit prioritization."""
        if value is None:
            return float("-inf")
        text = str(value).strip()
        if not text:
            return float("-inf")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return float("-inf")

    @classmethod
    def _sort_hits(
        cls,
        hits: list[Any],
        *,
        active_session_key: str | None,
    ) -> list[Any]:
        """Prefer current-session, newest-first hits over older stale context."""
        return sorted(
            hits,
            key=lambda hit: (
                1
                if active_session_key
                and str(getattr(hit, "session_key", "")) == active_session_key
                else 0,
                cls._parse_timestamp(getattr(hit, "timestamp", None)),
                int(getattr(hit, "seq", -1)),
            ),
            reverse=True,
        )

    @staticmethod
    def _is_low_signal_cross_session_query(query: str, *, regex: bool) -> bool:
        """Return True for broad queries likely to revive stale unrelated history."""
        if regex:
            return False
        normalized = (query or "").strip()
        if not normalized:
            return True
        if len(normalized) >= 48:
            return False
        if re.search(
            r"(0x[a-f0-9]{6,}|[/\\=:]|[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,8}\b|\b\d{3,}\b)",
            normalized,
            re.IGNORECASE,
        ):
            return False

        tokens = re.findall(r"[0-9A-Za-z\u4e00-\u9fff]+", normalized.lower())
        if not tokens:
            return True
        if len(tokens) <= 3 and sum(len(token) for token in tokens) <= 18:
            return True
        return False

    @property
    def name(self) -> str:
        return "search_history"

    @property
    def description(self) -> str:
        return (
            "Search the persisted conversation history (messages, tool "
            "calls, and tool results) using literal substring or regex "
            "matching. Use this to recover information after the runtime "
            "context has been compacted, for example to look up an earlier "
            "tool result, attachment id, user question, or assistant tool-"
            "call trace. By default the current session is searched and "
            "plain assistant replies are omitted to avoid reviving stale "
            "plans; pass roles=['assistant'] when you explicitly need the "
            "literal earlier assistant wording. Pass scope='all' only when "
            "you intentionally need older sessions and can search with a "
            "specific anchor such as an id, address, tx hash, filename, or "
            "exact tool output fragment. Returns a JSON object "
            "with a 'hits' array containing role, snippet, timestamp, and "
            "tool_call metadata for each match."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Substring (default) or regex (when regex=true) to "
                        "search for in message content and, unless "
                        "include_extras=false, serialized tool_call_id / "
                        "tool_calls / attachments metadata."
                    ),
                },
                "scope": {
                    "type": "string",
                    "enum": ["current", "all"],
                    "description": (
                        "'current' (default) searches only the active "
                        "session; 'all' searches every session owned by "
                        "the agent, and works best with a specific anchor."
                    ),
                },
                "session_key": {
                    "type": "string",
                    "description": (
                        "Explicit session to search. Overrides scope. "
                        "Usually left blank."
                    ),
                },
                "regex": {
                    "type": "boolean",
                    "description": "Interpret query as a regular expression.",
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Match case-sensitively (default false).",
                },
                "roles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Restrict to these message roles, e.g. "
                        "['tool', 'assistant']. When omitted, the tool "
                        "searches broadly but filters out plain assistant "
                        "replies by default; assistant tool-call traces are "
                        "still included."
                    ),
                },
                "include_extras": {
                    "type": "boolean",
                    "description": (
                        "Also match against serialized tool_call_id / "
                        "tool_calls / attachments metadata (default true)."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max hits to return (default 20, max 200).",
                    "minimum": 1,
                    "maximum": 200,
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of hits to skip (for paging).",
                    "minimum": 0,
                },
                "max_content_length": {
                    "type": "integer",
                    "description": (
                        "Truncate each returned message content to this "
                        "many characters (default 1000). Reduces token "
                        "usage when a single hit is very large."
                    ),
                    "minimum": 1,
                    "maximum": 20000,
                },
            },
            "required": ["query"],
        }

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        query: str,
        scope: str = "current",
        session_key: str | None = None,
        regex: bool = False,
        case_sensitive: bool = False,
        roles: Iterable[str] | None = None,
        include_extras: bool = True,
        limit: int = 20,
        offset: int = 0,
        max_content_length: int | None = 1000,
        **kwargs: Any,
    ) -> str:
        import json

        if not isinstance(query, str) or not query.strip():
            return "Error: 'query' must be a non-empty string."

        try:
            limit_int = max(1, min(200, int(limit)))
        except (TypeError, ValueError):
            limit_int = 20
        try:
            offset_int = max(0, int(offset))
        except (TypeError, ValueError):
            offset_int = 0
        mcl: int | None
        if max_content_length is None:
            mcl = None
        else:
            try:
                mcl = max(1, min(20_000, int(max_content_length)))
            except (TypeError, ValueError):
                mcl = 1000

        active_session_key = self._resolve_active_session_key()
        requested_scope = (
            str(scope).lower().strip() if isinstance(scope, str) else "current"
        )
        explicit_scope_all = (
            session_key is None
            and requested_scope == "all"
        )

        # Resolve which session(s) to search.
        if session_key is not None and str(session_key).strip():
            target_session_key: str | None = str(session_key)
        elif explicit_scope_all:
            target_session_key = None
        else:
            # Always ask the resolver first so we pick up WS-driven
            # session switches that bypass AgentLoop's internal hooks.
            target_session_key = active_session_key

        roles_list: list[str] | None
        if roles is None:
            roles_list = None
        else:
            roles_list = [str(r) for r in roles if str(r).strip()]
            if not roles_list:
                roles_list = None

        default_assistant_filter_active = roles_list is None
        raw_offset = 0
        raw_limit = min(
            200,
            max(
                limit_int + offset_int + 20,
                limit_int * 4 if default_assistant_filter_active else limit_int + offset_int,
                120 if target_session_key is None else 40,
            ),
        )

        scope_note: str | None = None
        if (
            explicit_scope_all
            and active_session_key is not None
            and self._is_low_signal_cross_session_query(query, regex=bool(regex))
        ):
            probe_hits = self._sessions_manager.search_messages(
                query,
                session_key=active_session_key,
                regex=bool(regex),
                case_sensitive=bool(case_sensitive),
                roles=roles_list,
                include_extras=bool(include_extras),
                limit=min(200, max(raw_limit, 40)),
                offset=0,
                max_content_length=mcl,
            )
            if default_assistant_filter_active:
                probe_hits = [
                    hit
                    for hit in probe_hits
                    if not self._is_plain_assistant_reply_hit(hit)
                ]
            if probe_hits:
                target_session_key = active_session_key
                scope_note = (
                    "Broad scope='all' query was narrowed to the active session "
                    "because the current session already had matches. Use a "
                    "specific anchor such as an id, address, tx hash, filename, "
                    "or exact tool output fragment when you truly need older sessions."
                )

        try:
            hits = self._sessions_manager.search_messages(
                query,
                session_key=target_session_key,
                regex=bool(regex),
                case_sensitive=bool(case_sensitive),
                roles=roles_list,
                include_extras=bool(include_extras),
                limit=raw_limit,
                offset=raw_offset,
                max_content_length=mcl,
            )
        except re.error as exc:
            return f"Error: invalid regex: {exc}"
        except ValueError as exc:
            return f"Error: {exc}"

        omitted_assistant_replies = 0
        if default_assistant_filter_active:
            filtered_hits = []
            for hit in hits:
                if self._is_plain_assistant_reply_hit(hit):
                    omitted_assistant_replies += 1
                    continue
                filtered_hits.append(hit)
            hits = filtered_hits

        hits = self._sort_hits(hits, active_session_key=active_session_key)
        if offset_int:
            hits = hits[offset_int:]
        hits = hits[:limit_int]

        payload = {
            "query": query,
            "scope": (
                "session" if target_session_key is not None else "all"
            ),
            "session_key": target_session_key,
            "total": len(hits),
            "limit": limit_int,
            "offset": offset_int,
            "hits": [
                {
                    "session_key": h.session_key,
                    "seq": h.seq,
                    "role": h.role,
                    "snippet": h.snippet,
                    "content": h.content,
                    "timestamp": h.timestamp,
                    "matched_in": h.matched_in,
                    "tool_call_id": (
                        h.extras.get("tool_call_id") if h.extras else None
                    ),
                    "tool_calls": (
                        h.extras.get("tool_calls") if h.extras else None
                    ),
                }
                for h in hits
            ],
        }
        if explicit_scope_all and target_session_key is not None:
            payload["requested_scope"] = "all"

        note_parts: list[str] = []
        if scope_note:
            note_parts.append(scope_note)
        if not hits:
            note_parts.append(
                "No matches found. Try a nearby exact substring, or use "
                "regex=true for structured matching. Expand to scope='all' "
                "only when you intentionally need older sessions."
            )
        if default_assistant_filter_active and (omitted_assistant_replies or not hits):
            payload["omitted_assistant_replies"] = omitted_assistant_replies
            note_parts.append(
                "Plain assistant replies are omitted by default to avoid reviving "
                "stale plans; pass roles=['assistant'] if you need the literal "
                "earlier assistant wording."
            )
        if note_parts:
            payload["note"] = " ".join(note_parts)

        return json.dumps(payload, ensure_ascii=False, default=str)
