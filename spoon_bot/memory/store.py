"""File-based memory store for persistent facts and daily notes."""

from datetime import datetime, date
from pathlib import Path
from typing import Any

from loguru import logger


class MemoryStore:
    """
    File-based memory store for human-readable persistence.

    Memory files:
    - MEMORY.md: Long-term persistent facts and preferences
    - YYYY-MM-DD.md: Daily notes and session summaries
    - projects/*.md: Project-specific context (future)
    """

    def __init__(self, workspace: Path):
        """
        Initialize memory store.

        Args:
            workspace: Path to workspace directory.
        """
        self.workspace = Path(workspace).expanduser().resolve()
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.memory_file = self.memory_dir / "MEMORY.md"
        self._ensure_memory_file()

    def _ensure_memory_file(self) -> None:
        """Ensure the MEMORY.md file exists."""
        if not self.memory_file.exists():
            self.memory_file.write_text(
                "# Long-term Memory\n\n"
                "This file stores persistent facts and preferences.\n\n"
                "## User Preferences\n\n"
                "## Important Facts\n\n",
                encoding="utf-8",
            )
            logger.debug(f"Created memory file: {self.memory_file}")

    def get_memory_context(self) -> str:
        """
        Get memory context for system prompt injection.

        Returns:
            Combined memory content (long-term + today's notes).
        """
        parts = []

        # Long-term memory
        if self.memory_file.exists():
            content = self.memory_file.read_text(encoding="utf-8")
            if content.strip():
                parts.append(f"## Long-term Memory\n\n{content}")

        # Today's notes
        today_file = self._get_daily_file(date.today())
        if today_file.exists():
            content = today_file.read_text(encoding="utf-8")
            if content.strip():
                parts.append(f"## Today's Notes\n\n{content}")

        return "\n\n".join(parts) if parts else ""

    def add_memory(self, content: str, category: str = "Facts") -> None:
        """
        Add a fact to long-term memory.

        Args:
            content: The fact to remember.
            category: Category for the fact (default: "Facts").
        """
        try:
            existing = self.memory_file.read_text(encoding="utf-8")

            # Find or create category section
            category_header = f"## {category}"
            if category_header not in existing:
                existing += f"\n\n{category_header}\n\n"

            # Add the new fact under the category
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            new_fact = f"- [{timestamp}] {content}\n"

            # Insert after category header
            pos = existing.find(category_header)
            if pos != -1:
                # Find end of header line
                end_of_header = existing.find("\n", pos)
                if end_of_header != -1:
                    # Insert after any existing newlines following header
                    insert_pos = end_of_header + 1
                    while insert_pos < len(existing) and existing[insert_pos] == "\n":
                        insert_pos += 1
                    existing = existing[:insert_pos] + new_fact + existing[insert_pos:]

            self.memory_file.write_text(existing, encoding="utf-8")
            logger.debug(f"Added memory: {content[:50]}...")

        except Exception as e:
            logger.error(f"Error adding memory: {e}")

    def add_daily_note(self, content: str) -> None:
        """
        Add a note to today's daily file.

        Args:
            content: The note to add.
        """
        today_file = self._get_daily_file(date.today())

        try:
            timestamp = datetime.now().strftime("%H:%M")

            if today_file.exists():
                existing = today_file.read_text(encoding="utf-8")
            else:
                existing = f"# Notes for {date.today().isoformat()}\n\n"

            existing += f"- [{timestamp}] {content}\n"
            today_file.write_text(existing, encoding="utf-8")
            logger.debug(f"Added daily note: {content[:50]}...")

        except Exception as e:
            logger.error(f"Error adding daily note: {e}")

    def search(self, query: str) -> list[str]:
        """
        Search memory files for matching content.

        Args:
            query: Search query.

        Returns:
            List of matching lines.
        """
        results = []
        query_lower = query.lower()

        # Search long-term memory
        if self.memory_file.exists():
            content = self.memory_file.read_text(encoding="utf-8")
            for line in content.split("\n"):
                if query_lower in line.lower():
                    results.append(f"[MEMORY] {line.strip()}")

        # Search recent daily notes (last 7 days)
        for i in range(7):
            day = date.today()
            from datetime import timedelta
            day = day - timedelta(days=i)
            daily_file = self._get_daily_file(day)

            if daily_file.exists():
                content = daily_file.read_text(encoding="utf-8")
                for line in content.split("\n"):
                    if query_lower in line.lower():
                        results.append(f"[{day.isoformat()}] {line.strip()}")

        return results[:20]  # Limit results

    def remove_memory(self, content: str) -> bool:
        """
        Remove a specific memory entry.

        Args:
            content: Content to remove (partial match).

        Returns:
            True if something was removed.
        """
        try:
            existing = self.memory_file.read_text(encoding="utf-8")
            lines = existing.split("\n")
            new_lines = [line for line in lines if content not in line]

            if len(new_lines) < len(lines):
                self.memory_file.write_text("\n".join(new_lines), encoding="utf-8")
                logger.debug(f"Removed memory containing: {content[:50]}...")
                return True
            return False

        except Exception as e:
            logger.error(f"Error removing memory: {e}")
            return False

    def get_summary(self) -> str:
        """Get a summary of memory contents."""
        parts = []

        # Count long-term memory entries
        if self.memory_file.exists():
            content = self.memory_file.read_text(encoding="utf-8")
            entry_count = content.count("- [")
            parts.append(f"Long-term memory: {entry_count} entries")

        # Count daily notes
        daily_files = list(self.memory_dir.glob("????-??-??.md"))
        if daily_files:
            parts.append(f"Daily notes: {len(daily_files)} days")

        return "\n".join(parts) if parts else "Memory is empty"

    def _get_daily_file(self, day: date) -> Path:
        """Get the path to a daily notes file."""
        return self.memory_dir / f"{day.isoformat()}.md"

    def clear_all(self) -> None:
        """Clear all memory (use with caution!)."""
        self._ensure_memory_file()
        for file in self.memory_dir.glob("????-??-??.md"):
            file.unlink()
        logger.warning("All memory cleared")
