from __future__ import annotations

import asyncio
import json
from pathlib import Path

from spoon_bot.agent.tools.history_search import SearchHistoryTool
from spoon_bot.session.manager import SessionManager
from spoon_bot.session.store import FileSessionStore


def _build_manager(tmp_path: Path) -> SessionManager:
    return SessionManager(store=FileSessionStore(tmp_path / "sessions"))


def _append_message(
    mgr: SessionManager,
    session_key: str,
    role: str,
    content: str,
    *,
    timestamp: str,
    **kwargs: object,
) -> None:
    session = mgr.get_or_create(session_key)
    session.add_message(role, content, **kwargs)
    session.messages[-1]["timestamp"] = timestamp
    mgr.save(session)


def test_scope_all_low_signal_query_prefers_current_session(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "user",
        "我之前玩了几把游戏",
        timestamp="2026-04-24T09:00:00",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "JOINED game=402 spot=A agentId=427",
        timestamp="2026-04-24T09:01:00",
        name="shell",
    )
    _append_message(
        mgr,
        "archived-session",
        "tool",
        "Usage: game list, game status, game context, game snapshot",
        timestamp="2026-04-01T08:00:00",
        name="shell",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    payload = json.loads(asyncio.run(tool.execute(query="game", scope="all")))

    assert payload["requested_scope"] == "all"
    assert payload["scope"] == "session"
    assert payload["session_key"] == "current-session"
    assert payload["total"] >= 1
    assert all(hit["session_key"] == "current-session" for hit in payload["hits"])
    assert "narrowed to the active session" in payload["note"]
    assert "specific anchor" in payload["note"]


def test_low_signal_current_hits_are_sorted_newest_first(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "tool",
        "Usage: game list, game status, game context, game snapshot",
        timestamp="2026-04-24T08:00:00",
        name="shell",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "JOINED game=402 spot=A agentId=427",
        timestamp="2026-04-24T09:10:00",
        name="shell",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "SETTLEMENT game=402 result=WIN rank=1/4 spot=A",
        timestamp="2026-04-24T09:20:00",
        name="shell",
    )
    _append_message(
        mgr,
        "archived-session",
        "tool",
        "game.js game-status game snapshot game mechanics",
        timestamp="2026-03-20T07:00:00",
        name="shell",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    payload = json.loads(asyncio.run(tool.execute(query="game", scope="all", limit=3)))
    contents = [hit["content"] for hit in payload["hits"]]

    assert payload["session_key"] == "current-session"
    assert contents[0].startswith("SETTLEMENT game=402")
    assert contents[1].startswith("JOINED game=402")
    assert contents[2].startswith("Usage: game list")


def test_specific_cross_session_query_still_searches_all_sessions(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "tool",
        "SETTLEMENT game=402 result=WIN rank=1/4 spot=A",
        timestamp="2026-04-24T09:20:00",
        name="shell",
    )
    _append_message(
        mgr,
        "archived-session",
        "tool",
        "SETTLEMENT game=288 result=WIN rank=1/4 spot=A",
        timestamp="2026-04-01T08:00:00",
        name="shell",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    payload = json.loads(
        asyncio.run(tool.execute(query="SETTLEMENT game=", scope="all", limit=5))
    )

    assert payload["scope"] == "all"
    assert payload["session_key"] is None
    assert {hit["session_key"] for hit in payload["hits"]} == {
        "current-session",
        "archived-session",
    }
    assert "requested_scope" not in payload
