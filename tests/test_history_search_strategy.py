from __future__ import annotations

import asyncio
import json
from pathlib import Path

from spoon_bot.agent.tools.execution_context import bind_request_execution_hints
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


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_history_search_budget_is_shared_by_search_and_recent(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "tool",
        "first result",
        timestamp="2026-08-14T08:00:00",
        name="shell",
    )
    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    hints: dict[str, object] = {
        "current_session_fact_check_required": True,
        "history_search_state": {
            "calls": 0,
            "max_calls": 3,
            "exhausted": False,
            "disabled": False,
            "disabled_reason": "",
        },
    }

    with bind_request_execution_hints(hints):
        asyncio.run(tool.execute(query="first"))
        asyncio.run(tool.execute(mode="recent"))
        asyncio.run(tool.execute(query="result"))
        fourth = json.loads(asyncio.run(tool.execute(mode="recent")))

    state = hints["history_search_state"]
    assert isinstance(state, dict)
    assert state["calls"] == 4
    assert state["disabled"] is True
    assert fourth["budget_exhausted"] is True
    assert fourth["guardrail_stop"] is True


def test_scope_all_low_signal_query_prefers_current_session(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "user",
        "What reports did I generate earlier?",
        timestamp="2026-04-24T09:00:00",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "REPORT_CREATED id=402 format=pdf owner=agent-427",
        timestamp="2026-04-24T09:01:00",
        name="shell",
    )
    _append_message(
        mgr,
        "archived-session",
        "tool",
        "Usage: report list, report status, report export",
        timestamp="2026-04-01T08:00:00",
        name="shell",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    payload = json.loads(asyncio.run(tool.execute(query="report", scope="all")))

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
        "Usage: report list, report status, report export",
        timestamp="2026-04-24T08:00:00",
        name="shell",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "REPORT_CREATED id=402 format=pdf owner=agent-427",
        timestamp="2026-04-24T09:10:00",
        name="shell",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "REPORT_SENT id=402 status=complete recipient=team-a",
        timestamp="2026-04-24T09:20:00",
        name="shell",
    )
    _append_message(
        mgr,
        "archived-session",
        "tool",
        "report.py report-status report export report metadata",
        timestamp="2026-03-20T07:00:00",
        name="shell",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    payload = json.loads(asyncio.run(tool.execute(query="report", scope="all", limit=3)))
    contents = [hit["content"] for hit in payload["hits"]]

    assert payload["session_key"] == "current-session"
    assert contents[0].startswith("REPORT_SENT id=402")
    assert contents[1].startswith("REPORT_CREATED id=402")
    assert contents[2].startswith("Usage: report list")


def test_scope_all_low_signal_query_without_current_evidence_does_not_search_old_sessions(
    tmp_path: Path,
) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "archived-session",
        "tool",
        "Game #329 wallet=0x24715FC49C1EBe85f46FD169567C41Ca1d7Cc972 result=LOSE",
        timestamp="2026-04-01T08:00:00",
        name="shell",
    )

    tool = SearchHistoryTool(mgr, default_session_key="fresh-session")
    payload = json.loads(asyncio.run(tool.execute(query="game", scope="all")))

    assert payload["requested_scope"] == "all"
    assert payload["scope"] == "session"
    assert payload["session_key"] == "fresh-session"
    assert payload["total"] == 0
    assert payload["hits"] == []
    assert "active session had no matching evidence" in payload["note"]


def test_specific_cross_session_query_still_searches_all_sessions(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "tool",
        "REPORT_SENT id=402 status=complete recipient=team-a",
        timestamp="2026-04-24T09:20:00",
        name="shell",
    )
    _append_message(
        mgr,
        "archived-session",
        "tool",
        "REPORT_SENT id=288 status=complete recipient=team-b",
        timestamp="2026-04-01T08:00:00",
        name="shell",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    payload = json.loads(
        asyncio.run(tool.execute(query="REPORT_SENT id=", scope="all", limit=5))
    )

    assert payload["scope"] == "all"
    assert payload["session_key"] is None
    assert {hit["session_key"] for hit in payload["hits"]} == {
        "current-session",
        "archived-session",
    }
    assert "requested_scope" not in payload


def test_recent_mode_omits_assistant_summaries_by_default(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "user",
        "汇总一下所有钱包加入的所有的比赛",
        timestamp="2026-07-02T07:20:58",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "Wallet=0x35781804E9a6Cf9278084A0a0B4b3B19F44dE2F2 GAS=0.07 AgentID=485",
        timestamp="2026-07-02T07:21:00",
        name="shell",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "JOINED game=416 spot=A agentId=485; SETTLEMENT game=416 result=WIN",
        timestamp="2026-07-02T07:21:05",
        name="shell",
    )
    _append_message(
        mgr,
        "current-session",
        "assistant",
        "错误总结：另一个钱包没有 Agent，也没有任何比赛记录。",
        timestamp="2026-07-02T07:21:10",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    payload = json.loads(
        asyncio.run(
            tool.execute(
                mode="recent",
                limit=3,
                include_assistant_summaries=True,
            )
        )
    )

    assert not _contains_key(payload, "assistant_summary")
    assert _contains_key(payload, "assistant_summary_omitted")
    assert "JOINED game=416" in json.dumps(payload, ensure_ascii=False)
    assert "错误总结" not in json.dumps(payload, ensure_ascii=False)
    assert "Assistant summaries are omitted by default" in payload["note"]


def test_fact_check_recent_payload_omits_assistant_summaries(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "user",
        "继续玩5把spot 游戏",
        timestamp="2026-07-02T06:29:45",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "Wallet=0x35781804E9a6Cf9278084A0a0B4b3B19F44dE2F2 GAS=0.072 AgentID=485",
        timestamp="2026-07-02T06:29:50",
        name="shell",
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        "SETTLEMENT game=415 result=DRAW; SETTLEMENT game=416 result=WIN",
        timestamp="2026-07-02T06:31:00",
        name="shell",
    )
    _append_message(
        mgr,
        "current-session",
        "assistant",
        "错误总结：这些比赛都属于新钱包。",
        timestamp="2026-07-02T06:31:10",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    with bind_request_execution_hints({"current_session_fact_check_required": True}):
        payload = json.loads(
            asyncio.run(tool.execute(query="0x35781804", scope="current", limit=5))
        )

    assert "same_session_recent" in payload
    assert not _contains_key(payload["same_session_recent"], "assistant_summary")
    assert _contains_key(payload["same_session_recent"], "assistant_summary_omitted")
    assert "错误总结" not in json.dumps(payload["same_session_recent"], ensure_ascii=False)
    assert payload["hits"][0]["evidence_type"] == "tool_result"


def test_assistant_hits_are_marked_as_claims_when_explicitly_requested(
    tmp_path: Path,
) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "assistant",
        "旧钱包没有任何比赛记录。",
        timestamp="2026-07-02T07:44:06",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    payload = json.loads(
        asyncio.run(tool.execute(query="旧钱包", roles=["assistant"], limit=1))
    )

    assert payload["hits"][0]["evidence_type"] == "assistant_claim"
    assert "not factual evidence" in payload["note"]


def test_search_history_omits_prior_search_history_echoes(tmp_path: Path) -> None:
    mgr = _build_manager(tmp_path)
    _append_message(
        mgr,
        "current-session",
        "tool",
        "JOINED game=415 wallet=0xabc SETTLEMENT game=415 result=WIN",
        timestamp="2026-07-02T06:35:00",
        name="shell",
    )
    _append_message(
        mgr,
        "current-session",
        "assistant",
        "",
        timestamp="2026-07-02T06:36:00",
        tool_calls=[
            {
                "id": "call_search",
                "type": "function",
                "function": {
                    "name": "search_history",
                    "arguments": '{"query":"game="}',
                },
            }
        ],
    )
    _append_message(
        mgr,
        "current-session",
        "tool",
        'Observed output of cmd search_history execution: {"content":"JOINED game=415"}',
        timestamp="2026-07-02T06:36:01",
        name="search_history",
        tool_call_id="call_search",
    )

    tool = SearchHistoryTool(mgr, default_session_key="current-session")
    payload = json.loads(asyncio.run(tool.execute(query="game=", limit=10)))

    assert payload["total"] == 1
    assert payload["hits"][0]["tool_call_id"] is None
    assert payload["hits"][0]["evidence_type"] == "tool_result"
    assert "JOINED game=415 wallet=0xabc" in payload["hits"][0]["content"]
