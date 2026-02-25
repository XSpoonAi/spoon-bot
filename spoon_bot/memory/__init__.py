"""Memory system for spoon-bot."""

from spoon_bot.memory.store import MemoryStore

__all__ = ["MemoryStore", "SemanticMemoryStore"]


def __getattr__(name: str):
    if name == "SemanticMemoryStore":
        from spoon_bot.memory.semantic_store import SemanticMemoryStore
        return SemanticMemoryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
