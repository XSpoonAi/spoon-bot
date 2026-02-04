"""Git integration for workspace version control.

Provides automatic git repository management:
- Initialize git repo when workspace is created
- Auto-commit changes after agent processes requests
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger


class GitManager:
    """Manages git operations for a workspace."""

    def __init__(self, workspace: Path | str):
        """
        Initialize the git manager.

        Args:
            workspace: Path to the workspace directory.
        """
        self.workspace = Path(workspace).resolve()
        self._git_available: Optional[bool] = None

    def is_git_available(self) -> bool:
        """Check if git is available on the system."""
        if self._git_available is not None:
            return self._git_available

        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self._git_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            self._git_available = False

        return self._git_available

    def is_repo_initialized(self) -> bool:
        """Check if the workspace is a git repository."""
        git_dir = self.workspace / ".git"
        return git_dir.exists() and git_dir.is_dir()

    def init(self) -> bool:
        """
        Initialize a git repository in the workspace.

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_git_available():
            logger.warning("Git is not available on this system")
            return False

        if self.is_repo_initialized():
            logger.debug(f"Git repo already initialized at {self.workspace}")
            return True

        try:
            result = subprocess.run(
                ["git", "init"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                logger.error(f"Git init failed: {result.stderr}")
                return False

            # Create .gitignore with sensible defaults
            gitignore = self.workspace / ".gitignore"
            if not gitignore.exists():
                gitignore.write_text(
                    "# Python\n"
                    "__pycache__/\n"
                    "*.py[cod]\n"
                    ".venv/\n"
                    "venv/\n"
                    "\n"
                    "# Environment\n"
                    ".env\n"
                    ".env.local\n"
                    "*.local\n"
                    "\n"
                    "# IDE\n"
                    ".idea/\n"
                    ".vscode/\n"
                    "*.swp\n"
                    "*.swo\n"
                    "\n"
                    "# OS\n"
                    ".DS_Store\n"
                    "Thumbs.db\n"
                )

            # Configure default user if not set globally
            self._ensure_git_user_config()

            # Initial commit
            self._run_git("add", "-A")
            self._run_git(
                "commit",
                "-m", "Initial workspace setup",
                "--allow-empty",
            )

            logger.info(f"Git repository initialized at {self.workspace}")
            return True

        except subprocess.TimeoutExpired:
            logger.error("Git init timed out")
            return False
        except Exception as e:
            logger.error(f"Git init error: {e}")
            return False

    def has_changes(self) -> bool:
        """
        Check if there are uncommitted changes in the workspace.

        Returns:
            True if there are changes to commit, False otherwise.
        """
        if not self.is_repo_initialized():
            return False

        try:
            # Check for staged and unstaged changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return False

            # Any output means there are changes
            return bool(result.stdout.strip())

        except (subprocess.TimeoutExpired, Exception):
            return False

    def commit(self, message: str) -> bool:
        """
        Stage all changes and commit with the given message.

        Args:
            message: The commit message.

        Returns:
            True if successful, False otherwise.
        """
        if not self.is_repo_initialized():
            logger.debug("Cannot commit: git repo not initialized")
            return False

        if not self.has_changes():
            logger.debug("No changes to commit")
            return False

        try:
            # Stage all changes
            stage_result = self._run_git("add", "-A")
            if not stage_result:
                return False

            # Sanitize commit message
            safe_message = self._sanitize_commit_message(message)

            # Commit
            commit_result = self._run_git(
                "commit",
                "-m", safe_message,
            )

            if commit_result:
                logger.info(f"Committed changes: {safe_message[:50]}...")
                return True

            return False

        except Exception as e:
            logger.error(f"Git commit error: {e}")
            return False

    def get_status(self) -> str:
        """
        Get a human-readable git status.

        Returns:
            Status string.
        """
        if not self.is_repo_initialized():
            return "Not a git repository"

        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                if result.stdout.strip():
                    return result.stdout.strip()
                return "Clean (no changes)"

            return "Unknown status"

        except Exception:
            return "Unable to get status"

    def get_log(self, limit: int = 5) -> list[dict]:
        """
        Get recent commit history.

        Args:
            limit: Maximum number of commits to return.

        Returns:
            List of commit info dicts.
        """
        if not self.is_repo_initialized():
            return []

        try:
            result = subprocess.run(
                [
                    "git", "log",
                    f"-{limit}",
                    "--format=%H|%s|%ai",
                ],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split("\n"):
                if line and "|" in line:
                    parts = line.split("|", 2)
                    if len(parts) >= 3:
                        commits.append({
                            "hash": parts[0][:8],
                            "message": parts[1],
                            "date": parts[2],
                        })

            return commits

        except Exception:
            return []

    def _ensure_git_user_config(self) -> None:
        """
        Ensure git user.name and user.email are configured for the repo.

        If not configured globally, set local config for spoon-bot.
        """
        try:
            # Check if user.name is configured
            result = subprocess.run(
                ["git", "config", "user.name"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if not result.stdout.strip():
                # Set default local user.name
                self._run_git("config", "user.name", "spoon-bot")

            # Check if user.email is configured
            result = subprocess.run(
                ["git", "config", "user.email"],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if not result.stdout.strip():
                # Set default local user.email
                self._run_git("config", "user.email", "spoon-bot@local")

        except Exception as e:
            logger.debug(f"Failed to ensure git user config: {e}")

    def _run_git(self, *args: str) -> bool:
        """
        Run a git command in the workspace.

        Args:
            *args: Git command arguments.

        Returns:
            True if successful, False otherwise.
        """
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                logger.debug(f"Git {args[0]} failed: {result.stderr}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.error(f"Git {args[0]} timed out")
            return False
        except Exception as e:
            logger.error(f"Git {args[0]} error: {e}")
            return False

    def _sanitize_commit_message(self, message: str) -> str:
        """
        Sanitize a commit message for git.

        Args:
            message: Raw message.

        Returns:
            Sanitized message suitable for git commit.
        """
        # Truncate long messages
        max_length = 500
        if len(message) > max_length:
            message = message[:max_length] + "..."

        # Remove leading/trailing whitespace
        message = message.strip()

        # Replace problematic characters
        message = message.replace("\x00", "")  # Null bytes

        # If message is empty or only whitespace, use a default
        if not message:
            message = "Update workspace files"

        # Prefix with [spoon-bot] if not already prefixed
        if not message.startswith("["):
            message = f"[spoon-bot] {message}"

        return message


# Async wrappers for convenience
async def git_init(workspace: Path | str) -> bool:
    """
    Initialize git repo in workspace (async wrapper).

    Args:
        workspace: Path to workspace directory.

    Returns:
        True if successful.
    """
    manager = GitManager(workspace)
    return await asyncio.to_thread(manager.init)


async def git_commit(workspace: Path | str, message: str) -> bool:
    """
    Commit changes in workspace (async wrapper).

    Args:
        workspace: Path to workspace directory.
        message: Commit message.

    Returns:
        True if successful.
    """
    manager = GitManager(workspace)
    return await asyncio.to_thread(manager.commit, message)


async def git_has_changes(workspace: Path | str) -> bool:
    """
    Check if workspace has uncommitted changes (async wrapper).

    Args:
        workspace: Path to workspace directory.

    Returns:
        True if there are changes.
    """
    manager = GitManager(workspace)
    return await asyncio.to_thread(manager.has_changes)
