"""Tests for path traversal security in filesystem tools.

These tests verify that the PathValidator correctly blocks:
1. Path traversal attacks (../)
2. Access to sensitive system files
3. Symlinks that escape the workspace
4. Paths outside the workspace boundary
"""

import os
import tempfile
from pathlib import Path

import pytest

from spoon_bot.agent.tools.path_validator import (
    PathValidator,
    PathValidationResult,
    validate_read_path,
    validate_write_path,
    validate_directory_path,
)
from spoon_bot.agent.tools.filesystem import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    ListDirTool,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary workspace directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        # Create some test files
        (workspace / "test.txt").write_text("Hello, World!")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "nested.txt").write_text("Nested content")
        yield workspace


@pytest.fixture
def validator(temp_workspace):
    """Create a PathValidator with the temp workspace."""
    return PathValidator(workspace=temp_workspace)


class TestPathValidator:
    """Test PathValidator class directly."""

    def test_valid_path_within_workspace(self, validator, temp_workspace):
        """Test that valid paths within workspace are accepted."""
        result = validator.validate_read_path(temp_workspace / "test.txt")
        assert result.valid is True
        assert result.resolved_path == temp_workspace / "test.txt"
        assert result.error is None

    def test_path_traversal_blocked(self, validator, temp_workspace):
        """Test that path traversal attempts are blocked."""
        # Try to escape using ../
        result = validator.validate_read_path(temp_workspace / ".." / ".." / "etc" / "passwd")
        assert result.valid is False
        assert "outside workspace" in result.error.lower()

    def test_absolute_path_outside_workspace(self, validator, temp_workspace):
        """Test that absolute paths outside workspace are blocked."""
        if os.name == 'nt':  # Windows
            result = validator.validate_read_path("C:\\Windows\\System32\\config\\sam")
        else:
            result = validator.validate_read_path("/etc/passwd")
        assert result.valid is False
        # Could be blocked for either being outside workspace or sensitive path

    def test_sensitive_path_blocked(self, validator, temp_workspace):
        """Test that sensitive paths are blocked."""
        # Even if it doesn't exist, the path pattern should be blocked
        result = validator.validate_read_path("/etc/passwd")
        assert result.valid is False

    def test_ssh_key_blocked(self, validator, temp_workspace):
        """Test that SSH keys are blocked."""
        result = validator.validate_read_path(Path.home() / ".ssh" / "id_rsa")
        assert result.valid is False

    def test_write_path_validation_stricter(self, validator, temp_workspace):
        """Test that write validation is stricter than read."""
        # Try to write outside workspace - should always fail for write
        result = validator.validate_write_path("/tmp/outside.txt")
        assert result.valid is False
        assert "outside workspace" in result.error.lower() or "blocked" in result.error.lower()

    def test_write_to_valid_path(self, validator, temp_workspace):
        """Test that writing to valid paths is allowed."""
        result = validator.validate_write_path(temp_workspace / "new_file.txt")
        assert result.valid is True

    def test_write_to_nested_path(self, validator, temp_workspace):
        """Test that writing to nested paths within workspace is allowed."""
        result = validator.validate_write_path(temp_workspace / "new_dir" / "new_file.txt")
        assert result.valid is True

    def test_directory_validation(self, validator, temp_workspace):
        """Test directory path validation."""
        result = validator.validate_directory_path(temp_workspace / "subdir")
        assert result.valid is True

    def test_directory_outside_workspace(self, validator, temp_workspace):
        """Test that directory listing outside workspace is blocked."""
        result = validator.validate_directory_path("/etc")
        assert result.valid is False


class TestFilesystemTools:
    """Test filesystem tools with path security."""

    @pytest.mark.asyncio
    async def test_read_file_within_workspace(self, temp_workspace):
        """Test reading a file within workspace."""
        tool = ReadFileTool(workspace=temp_workspace)
        result = await tool.execute(path=str(temp_workspace / "test.txt"))
        assert "Hello, World!" in result

    @pytest.mark.asyncio
    async def test_read_file_path_traversal_blocked(self, temp_workspace):
        """Test that path traversal in read is blocked."""
        tool = ReadFileTool(workspace=temp_workspace)
        result = await tool.execute(path=str(temp_workspace / ".." / ".." / "etc" / "passwd"))
        assert "Security Error" in result
        assert "outside workspace" in result.lower()

    @pytest.mark.asyncio
    async def test_read_sensitive_path_blocked(self, temp_workspace):
        """Test that reading sensitive paths is blocked."""
        tool = ReadFileTool(workspace=temp_workspace)
        result = await tool.execute(path="/etc/passwd")
        assert "Security Error" in result

    @pytest.mark.asyncio
    async def test_write_file_within_workspace(self, temp_workspace):
        """Test writing a file within workspace."""
        tool = WriteFileTool(workspace=temp_workspace)
        result = await tool.execute(
            path=str(temp_workspace / "new_file.txt"),
            content="New content"
        )
        assert "Successfully wrote" in result
        assert (temp_workspace / "new_file.txt").read_text() == "New content"

    @pytest.mark.asyncio
    async def test_write_file_path_traversal_blocked(self, temp_workspace):
        """Test that path traversal in write is blocked."""
        tool = WriteFileTool(workspace=temp_workspace)
        result = await tool.execute(
            path=str(temp_workspace / ".." / "malicious.txt"),
            content="Bad content"
        )
        assert "Security Error" in result

    @pytest.mark.asyncio
    async def test_write_outside_workspace_blocked(self, temp_workspace):
        """Test that writing outside workspace is blocked."""
        tool = WriteFileTool(workspace=temp_workspace)
        result = await tool.execute(
            path="/tmp/outside_workspace.txt",
            content="Should not write"
        )
        assert "Security Error" in result

    @pytest.mark.asyncio
    async def test_edit_file_within_workspace(self, temp_workspace):
        """Test editing a file within workspace."""
        tool = EditFileTool(workspace=temp_workspace)
        result = await tool.execute(
            path=str(temp_workspace / "test.txt"),
            old_text="Hello",
            new_text="Goodbye"
        )
        assert "Successfully edited" in result
        assert (temp_workspace / "test.txt").read_text() == "Goodbye, World!"

    @pytest.mark.asyncio
    async def test_edit_file_path_traversal_blocked(self, temp_workspace):
        """Test that path traversal in edit is blocked."""
        tool = EditFileTool(workspace=temp_workspace)
        result = await tool.execute(
            path=str(temp_workspace / ".." / ".." / "etc" / "hosts"),
            old_text="localhost",
            new_text="malicious"
        )
        assert "Security Error" in result

    @pytest.mark.asyncio
    async def test_list_dir_within_workspace(self, temp_workspace):
        """Test listing directory within workspace."""
        tool = ListDirTool(workspace=temp_workspace)
        result = await tool.execute(path=str(temp_workspace))
        assert "[FILE]" in result or "[DIR]" in result
        assert "test.txt" in result
        assert "subdir" in result

    @pytest.mark.asyncio
    async def test_list_dir_path_traversal_blocked(self, temp_workspace):
        """Test that path traversal in list_dir is blocked."""
        tool = ListDirTool(workspace=temp_workspace)
        result = await tool.execute(path=str(temp_workspace / ".." / ".."))
        assert "Security Error" in result

    @pytest.mark.asyncio
    async def test_list_dir_outside_workspace_blocked(self, temp_workspace):
        """Test that listing outside workspace is blocked."""
        tool = ListDirTool(workspace=temp_workspace)
        result = await tool.execute(path="/etc")
        assert "Security Error" in result


class TestSymlinkSecurity:
    """Test symlink-based attacks are blocked."""

    @pytest.mark.asyncio
    async def test_symlink_escape_blocked(self, temp_workspace):
        """Test that symlinks pointing outside workspace are blocked."""
        # Create a symlink pointing outside workspace
        if os.name == 'nt':
            pytest.skip("Symlink test requires admin on Windows")

        symlink_path = temp_workspace / "escape_link"
        try:
            symlink_path.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlink (permission denied)")

        tool = ListDirTool(workspace=temp_workspace)
        # The symlink itself can be listed, but should show as pointing outside
        result = await tool.execute(path=str(temp_workspace))
        # The listing should show the symlink but indicate it's outside workspace
        if "escape_link" in result:
            assert "outside workspace" in result or "broken" in result


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_workspace(self):
        """Test validator with default (cwd) workspace."""
        validator = PathValidator()
        # Should use current directory as workspace
        assert validator.workspace == Path.cwd().resolve()

    def test_expanduser_handling(self, temp_workspace):
        """Test that ~ is expanded correctly."""
        validator = PathValidator(workspace=temp_workspace)
        result = validator.validate_read_path("~/.ssh/id_rsa")
        assert result.valid is False
        # Should be blocked either as outside workspace or sensitive path

    def test_case_sensitivity_blocklist(self, temp_workspace):
        """Test that blocklist is case-insensitive on Windows."""
        validator = PathValidator(workspace=temp_workspace)
        # Try various case combinations for sensitive paths
        for path in ["/ETC/PASSWD", "/Etc/Passwd", "/etc/PASSWD"]:
            result = validator.validate_read_path(path)
            assert result.valid is False

    def test_dotenv_in_workspace_allowed(self, temp_workspace):
        """Test that .env files within workspace can be accessed."""
        # Create a .env file in workspace
        env_file = temp_workspace / ".env"
        env_file.write_text("API_KEY=test")

        validator = PathValidator(workspace=temp_workspace)
        result = validator.validate_read_path(env_file)
        # .env within workspace should be allowed (strict_mode allows it)
        assert result.valid is True

    def test_credentials_outside_workspace_blocked(self, temp_workspace):
        """Test that credentials files outside workspace are blocked."""
        validator = PathValidator(workspace=temp_workspace)
        result = validator.validate_read_path(Path.home() / ".aws" / "credentials")
        assert result.valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
