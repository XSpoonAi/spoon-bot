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
            "tool result, attachment id, or user question. By default the "
            "current session is searched; pass scope='all' to search every "
            "session for the current agent. Returns a JSON object with a "
            "'hits' array containing role, snippet, timestamp, and "
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
                        "the agent."
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
                        "['tool', 'assistant']. Useful for finding only "
                        "tool results or only the user's prior questions."
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
        import re

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

        # Resolve which session(s) to search.
        if session_key is not None and str(session_key).strip():
            target_session_key: str | None = str(session_key)
        elif isinstance(scope, str) and scope.lower() == "all":
            target_session_key = None
        else:
            # Always ask the resolver first so we pick up WS-driven
            # session switches that bypass AgentLoop's internal hooks.
            target_session_key = self._resolve_active_session_key()

        roles_list: list[str] | None
        if roles is None:
            roles_list = None
        else:
            roles_list = [str(r) for r in roles if str(r).strip()]
            if not roles_list:
                roles_list = None

        try:
            hits = self._sessions_manager.search_messages(
                query,
                session_key=target_session_key,
                regex=bool(regex),
                case_sensitive=bool(case_sensitive),
                roles=roles_list,
                include_extras=bool(include_extras),
                limit=limit_int,
                offset=offset_int,
                max_content_length=mcl,
            )
        except re.error as exc:
            return f"Error: invalid regex: {exc}"
        except ValueError as exc:
            return f"Error: {exc}"

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

        if not hits:
            payload["note"] = (
                "No matches found. Try a broader substring, scope='all', "
                "or regex=true for fuzzier matching."
            )

        return json.dumps(payload, ensure_ascii=False, default=str)
