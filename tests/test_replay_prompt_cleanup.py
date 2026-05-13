from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "workspace"
    / "replay-124100617072279552-2"
    / "rerun_full_fixed_case_report_cypherapi_subproc.py"
)


def _load_script_module():
    module_name = "replay_cypherapi_subproc"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_session(path: Path, prompts: list[str]) -> None:
    rows: list[str] = []
    for prompt in prompts:
        rows.append(
            __import__("json").dumps(
                {"role": "user", "content": prompt},
                ensure_ascii=False,
            )
        )
        rows.append(
            __import__("json").dumps(
                {"role": "assistant", "content": "ok"},
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_strip_preloaded_skill_then_remove_duplicate_sections():
    module = _load_script_module()
    raw = (
        "Install this zip and join the latest spot game\n\n"
        "Install this zip and join the latest spot game\n"
        "---\n"
        "[PRE-LOADED SKILL: agent-spot-cypher]\n"
        "very long skill body"
    )

    stripped, had_preload = module.strip_preloaded_skill_block(raw)
    normalized = module.normalize_replay_prompt(stripped)

    assert had_preload is True
    assert normalized == "Install this zip and join the latest spot game"


def test_normalize_replay_prompt_keeps_distinct_sections_once_each():
    module = _load_script_module()
    raw = (
        "Install the skill from the attachment first\n\n"
        "Then join the latest game\n\n"
        "Install the skill from the attachment first\n\n"
        "Then join the latest game\n\n"
        "Finally summarize the full process"
    )

    normalized = module.normalize_replay_prompt(raw)

    assert normalized == (
        "Install the skill from the attachment first\n\n"
        "Then join the latest game\n\n"
        "Finally summarize the full process"
    )


def test_load_turns_skips_exact_duplicate_prompts(tmp_path):
    module = _load_script_module()
    session_file = tmp_path / "session.jsonl"
    _write_session(
        session_file,
        [
            "Install the attached skill and join the latest spot game",
            "Install the attached skill and join the latest spot game",
            "Summarize the last game round",
        ],
    )
    module.SESSION_FILE = session_file

    turns = module.load_turns()

    assert [turn.prompt for turn in turns] == [
        "Install the attached skill and join the latest spot game",
        "Summarize the last game round",
    ]
    assert len(module._LAST_SKIPPED_PROMPTS) == 1
    assert module._LAST_SKIPPED_PROMPTS[0]["reason"] == "exact"


def test_load_turns_skips_near_duplicate_prompts(tmp_path):
    module = _load_script_module()
    session_file = tmp_path / "session.jsonl"
    _write_session(
        session_file,
        [
            "Please install the attached spot skill, join the latest spot game, and record the wallet address.",
            "Install the attached spot skill and join the latest spot game, then record the wallet address.",
            "Only reply with the address and AgentID",
        ],
    )
    module.SESSION_FILE = session_file

    turns = module.load_turns()

    assert [turn.prompt for turn in turns] == [
        "Please install the attached spot skill, join the latest spot game, and record the wallet address.",
        "Only reply with the address and AgentID",
    ]
    assert len(module._LAST_SKIPPED_PROMPTS) == 1
    assert module._LAST_SKIPPED_PROMPTS[0]["reason"] == "similar"


def test_prompt_similarity_keeps_different_stable_anchors():
    module = _load_script_module()

    assert module.prompts_are_similar("Only reply with the address", "Only reply with the address and AgentID") is False
