from __future__ import annotations

from spoon_bot.memory.store import MemoryStore


def test_prompt_memory_excludes_dynamic_facts_but_full_memory_keeps_them(tmp_path) -> None:
    store = MemoryStore(tmp_path)
    store.memory_file.write_text(
        "# Memory\n"
        "- User prefers concise Chinese answers.\n"
        "- Wallet balance: 215 GLD.\n"
        "- Game result: win, reward 75 GLD, rank 1.\n"
        "- The project uses PostgreSQL.\n",
        encoding="utf-8",
    )

    full = store.get_memory_context()
    prompt = store.get_prompt_memory_context()

    assert "Wallet balance: 215 GLD" in full
    assert "Game result: win" in full
    assert "Wallet balance" not in prompt
    assert "Game result" not in prompt
    assert "User prefers concise Chinese answers" in prompt
    assert "The project uses PostgreSQL" in prompt
