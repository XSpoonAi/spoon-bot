"""Security tests: shell command validation + path traversal protection.

Merged from:
  - test_shell_security.py (§1: CommandValidator, ShellTool, SafeShellTool)
  - test_path_security.py  (§2: PathValidator, filesystem tools, symlinks)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# ═══════════════════════════════════════════════════════════════════
# §1  Shell Command Security
# ═══════════════════════════════════════════════════════════════════

from spoon_bot.agent.tools.shell import CommandValidator, ShellTool, SafeShellTool


class TestCommandValidator:
    """Tests for the CommandValidator class."""

    @pytest.fixture
    def validator(self):
        return CommandValidator()

    @pytest.fixture
    def whitelist_validator(self):
        return CommandValidator(whitelist_mode=True)

    @pytest.fixture
    def strict_validator(self):
        return CommandValidator(strict_mode=True)

    # -- Dangerous commands --

    class TestDangerousCommands:

        @pytest.mark.parametrize("command", [
            "rm -rf /", "rm -rf /*", "rm -rf ~", "rm -rf ~/*",
            "rm -rf .", "rm -fr /", "sudo rm -rf /", "  rm -rf /  ",
        ])
        def test_blocks_recursive_deletion(self, command):
            v = CommandValidator()
            is_valid, error = v.validate(command)
            assert not is_valid
            assert error is not None
            assert "dangerous" in error.lower() or "blocked" in error.lower()

        @pytest.mark.parametrize("command", [
            "mkfs.ext4 /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
            "dd if=/dev/random of=/dev/sdb",
        ])
        def test_blocks_disk_operations(self, command):
            v = CommandValidator()
            is_valid, _ = v.validate(command)
            assert not is_valid

        @pytest.mark.parametrize("command", [
            "format c:", "format d:",
            "del /f /s /q c:\\", "rd /s /q c:\\",
        ])
        def test_blocks_windows_destructive(self, command):
            v = CommandValidator()
            is_valid, _ = v.validate(command)
            assert not is_valid

        def test_blocks_fork_bomb(self):
            v = CommandValidator()
            is_valid, _ = v.validate(":(){ :|:& };:")
            assert not is_valid

    # -- Injection patterns --

    class TestInjectionPatterns:

        @pytest.mark.parametrize("command", [
            "ls; echo hello",
            "echo hello; cat /etc/passwd",
            "pwd ;cat /etc/shadow",
        ])
        def test_blocks_semicolon_chaining(self, command):
            v = CommandValidator()
            is_valid, error = v.validate(command)
            assert not is_valid
            assert "injection" in error.lower() or "blocked" in error.lower()

        @pytest.mark.parametrize("command", [
            "echo hello\ncat /etc/passwd",
            "echo hello\rwhoami",
            "echo hello\r\nid",
        ])
        def test_blocks_newline_chaining(self, command):
            v = CommandValidator()
            is_valid, error = v.validate(command)
            assert not is_valid
            assert "injection" in error.lower() or "blocked" in error.lower()

        @pytest.mark.parametrize("command", [
            "ls && echo hello",
            "echo test && cat /etc/passwd",
        ])
        def test_blocks_and_chaining(self, command):
            v = CommandValidator()
            is_valid, error = v.validate(command)
            assert not is_valid
            assert "injection" in error.lower() or "blocked" in error.lower()

        @pytest.mark.parametrize("command", [
            "ls || echo hello",
            "false || cat /etc/passwd",
        ])
        def test_blocks_or_chaining(self, command):
            v = CommandValidator()
            is_valid, error = v.validate(command)
            assert not is_valid
            assert "injection" in error.lower() or "blocked" in error.lower()

        @pytest.mark.parametrize("command", [
            "echo $(cat /etc/passwd)",
            "ls $(whoami)",
            "ping $(hostname)",
        ])
        def test_blocks_command_substitution(self, command):
            v = CommandValidator()
            is_valid, error = v.validate(command)
            assert not is_valid
            assert "injection" in error.lower()

        @pytest.mark.parametrize("command", [
            "echo `cat /etc/passwd`",
            "ls `whoami`",
        ])
        def test_blocks_backtick_substitution(self, command):
            v = CommandValidator()
            is_valid, error = v.validate(command)
            assert not is_valid
            assert "injection" in error.lower()

        @pytest.mark.parametrize("command", [
            "echo test > /etc/passwd",
            "cat file > /etc/shadow",
            "echo key > ~/.ssh/authorized_keys",
            "echo export > ~/.bashrc",
        ])
        def test_blocks_dangerous_redirections(self, command):
            v = CommandValidator()
            is_valid, error = v.validate(command)
            assert not is_valid
            assert "injection" in error.lower()

    # -- Whitelist mode --

    class TestWhitelistMode:

        def test_allows_whitelisted(self):
            v = CommandValidator(whitelist_mode=True)
            for cmd in ["ls -la", "git status", "python --version", "npm install"]:
                ok, err = v.validate(cmd)
                assert ok, f"'{cmd}' blocked: {err}"

        def test_blocks_non_whitelisted(self):
            v = CommandValidator(whitelist_mode=True)
            ok, err = v.validate("evil_command --danger")
            assert not ok
            assert "whitelist" in err.lower()

        def test_custom_whitelist(self):
            v = CommandValidator(whitelist_mode=True, custom_whitelist={"mycustomcmd"})
            ok, _ = v.validate("mycustomcmd --arg1")
            assert ok

        def test_handles_full_paths(self):
            v = CommandValidator(whitelist_mode=True)
            ok, _ = v.validate("/usr/bin/ls -la")
            assert ok

    # -- Safe commands --

    class TestSafeCommands:

        @pytest.mark.parametrize("command", [
            "ls -la", "pwd", "git status", "git log --oneline",
            "python --version", "node --version", "npm install",
            "cat README.md", "grep pattern file.txt", "echo 'hello world'",
            "mkdir new_folder", "cp file1.txt file2.txt", "mv old.txt new.txt",
        ])
        def test_allows_safe_commands(self, command):
            v = CommandValidator()
            ok, err = v.validate(command)
            assert ok, f"'{command}' blocked: {err}"

    # -- Pipe handling --

    class TestPipeHandling:

        def test_allows_pipes_by_default(self):
            v = CommandValidator(allow_pipes=True)
            ok, _ = v.validate("ls -la | grep pattern")
            assert ok

        def test_blocks_pipes_when_disabled(self):
            v = CommandValidator(allow_pipes=False)
            ok, err = v.validate("ls -la | grep pattern")
            assert not ok
            assert "pipe" in err.lower()

    # -- Strict mode --

    class TestStrictMode:

        def test_blocks_sensitive_paths(self):
            v = CommandValidator(strict_mode=True)
            for cmd in ["cat /etc/passwd", "nano /etc/shadow", "vim ~/.ssh/id_rsa"]:
                ok, _ = v.validate(cmd)
                assert not ok, f"'{cmd}' should be blocked"

        def test_allows_sensitive_in_normal_mode(self):
            v = CommandValidator(strict_mode=False)
            ok, _ = v.validate("cat /etc/hosts")
            assert ok

    # -- Custom blocklist --

    class TestCustomBlocklist:

        def test_blocks_custom_patterns(self):
            v = CommandValidator(custom_blocklist={"dangerous_tool", "bad_command"})
            ok, err = v.validate("dangerous_tool --arg")
            assert not ok
            assert "blocklist" in err.lower()

    # -- Edge cases --

    class TestEdgeCases:

        def test_empty_command(self):
            v = CommandValidator()
            ok, err = v.validate("")
            assert not ok
            assert "empty" in err.lower()

        def test_whitespace_only(self):
            v = CommandValidator()
            ok, _ = v.validate("   ")
            assert not ok

        def test_sanitize_truncates(self):
            v = CommandValidator()
            s = v.sanitize_for_display("a" * 200, max_length=50)
            assert len(s) <= 53
            assert s.endswith("...")


class TestShellTool:

    @pytest.fixture
    def shell_tool(self):
        return ShellTool(timeout=5)

    @pytest.mark.asyncio
    async def test_blocks_dangerous(self, shell_tool):
        r = await shell_tool.execute("rm -rf /")
        assert "Security Error" in r
        assert "dangerous" in r.lower() or "blocked" in r.lower()

    @pytest.mark.asyncio
    async def test_blocks_injection(self, shell_tool):
        r = await shell_tool.execute("echo hello; cat /etc/passwd")
        assert "Security Error" in r
        assert "injection" in r.lower()

    @pytest.mark.asyncio
    async def test_allows_safe(self, shell_tool):
        r = await shell_tool.execute("echo 'test'")
        assert "Security Error" not in r

    @pytest.mark.asyncio
    async def test_invalid_working_dir(self, shell_tool):
        r = await shell_tool.execute("ls", working_dir="/nonexistent/directory/path")
        assert "Error" in r
        assert "not found" in r.lower()


class TestSafeShellTool:

    @pytest.fixture
    def safe_shell(self):
        return SafeShellTool(timeout=5)

    @pytest.mark.asyncio
    async def test_blocks_non_whitelisted(self, safe_shell):
        r = await safe_shell.execute("some_unknown_command")
        assert "Security Error" in r
        assert "whitelist" in r.lower()

    @pytest.mark.asyncio
    async def test_allows_whitelisted(self, safe_shell):
        r = await safe_shell.execute("echo 'hello'")
        assert "Security Error" not in r

    def test_name(self, safe_shell):
        assert safe_shell.name == "safe_shell"

    def test_description(self, safe_shell):
        assert "whitelist" in safe_shell.description.lower()


# ═══════════════════════════════════════════════════════════════════
# §2  Path Traversal Security
# ═══════════════════════════════════════════════════════════════════

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
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir) / "workspace"
        workspace.mkdir()
        (workspace / "test.txt").write_text("Hello, World!")
        (workspace / "subdir").mkdir()
        (workspace / "subdir" / "nested.txt").write_text("Nested content")
        yield workspace


@pytest.fixture
def validator(temp_workspace):
    return PathValidator(workspace=temp_workspace)


class TestPathValidator:

    def test_valid_path(self, validator, temp_workspace):
        r = validator.validate_read_path(temp_workspace / "test.txt")
        assert r.valid is True
        assert r.resolved_path == temp_workspace / "test.txt"

    def test_traversal_blocked(self, validator, temp_workspace):
        r = validator.validate_read_path(temp_workspace / ".." / ".." / "etc" / "passwd")
        assert r.valid is False
        assert "outside workspace" in r.error.lower()

    def test_absolute_outside_blocked(self, validator, temp_workspace):
        if os.name == "nt":
            r = validator.validate_read_path("C:\\Windows\\System32\\config\\sam")
        else:
            r = validator.validate_read_path("/etc/passwd")
        assert r.valid is False

    def test_sensitive_blocked(self, validator, temp_workspace):
        r = validator.validate_read_path("/etc/passwd")
        assert r.valid is False

    def test_ssh_key_blocked(self, validator, temp_workspace):
        r = validator.validate_read_path(Path.home() / ".ssh" / "id_rsa")
        assert r.valid is False

    def test_write_outside_blocked(self, validator, temp_workspace):
        r = validator.validate_write_path("/tmp/outside.txt")
        assert r.valid is False

    def test_write_valid(self, validator, temp_workspace):
        r = validator.validate_write_path(temp_workspace / "new_file.txt")
        assert r.valid is True

    def test_write_nested(self, validator, temp_workspace):
        r = validator.validate_write_path(temp_workspace / "new_dir" / "new_file.txt")
        assert r.valid is True

    def test_dir_valid(self, validator, temp_workspace):
        r = validator.validate_directory_path(temp_workspace / "subdir")
        assert r.valid is True

    def test_dir_outside_blocked(self, validator, temp_workspace):
        r = validator.validate_directory_path("/etc")
        assert r.valid is False


class TestFilesystemTools:

    @pytest.mark.asyncio
    async def test_read_ok(self, temp_workspace):
        t = ReadFileTool(workspace=temp_workspace)
        r = await t.execute(path=str(temp_workspace / "test.txt"))
        assert "Hello, World!" in r

    @pytest.mark.asyncio
    async def test_read_traversal_blocked(self, temp_workspace):
        t = ReadFileTool(workspace=temp_workspace)
        r = await t.execute(path=str(temp_workspace / ".." / ".." / "etc" / "passwd"))
        assert "Security Error" in r
        assert "outside workspace" in r.lower()

    @pytest.mark.asyncio
    async def test_read_sensitive_blocked(self, temp_workspace):
        t = ReadFileTool(workspace=temp_workspace)
        r = await t.execute(path="/etc/passwd")
        assert "Security Error" in r

    @pytest.mark.asyncio
    async def test_write_ok(self, temp_workspace):
        t = WriteFileTool(workspace=temp_workspace)
        r = await t.execute(path=str(temp_workspace / "new_file.txt"), content="New content")
        assert "Successfully wrote" in r
        assert (temp_workspace / "new_file.txt").read_text() == "New content"

    @pytest.mark.asyncio
    async def test_write_traversal_blocked(self, temp_workspace):
        t = WriteFileTool(workspace=temp_workspace)
        r = await t.execute(path=str(temp_workspace / ".." / "malicious.txt"), content="Bad")
        assert "Security Error" in r

    @pytest.mark.asyncio
    async def test_write_outside_blocked(self, temp_workspace):
        t = WriteFileTool(workspace=temp_workspace)
        r = await t.execute(path="/tmp/outside_workspace.txt", content="x")
        assert "Security Error" in r

    @pytest.mark.asyncio
    async def test_edit_ok(self, temp_workspace):
        t = EditFileTool(workspace=temp_workspace)
        r = await t.execute(path=str(temp_workspace / "test.txt"), old_text="Hello", new_text="Goodbye")
        assert "Successfully edited" in r
        assert (temp_workspace / "test.txt").read_text() == "Goodbye, World!"

    @pytest.mark.asyncio
    async def test_edit_traversal_blocked(self, temp_workspace):
        t = EditFileTool(workspace=temp_workspace)
        r = await t.execute(
            path=str(temp_workspace / ".." / ".." / "etc" / "hosts"),
            old_text="localhost", new_text="malicious",
        )
        assert "Security Error" in r

    @pytest.mark.asyncio
    async def test_list_dir_ok(self, temp_workspace):
        t = ListDirTool(workspace=temp_workspace)
        r = await t.execute(path=str(temp_workspace))
        assert "[FILE]" in r or "[DIR]" in r
        assert "test.txt" in r
        assert "subdir" in r

    @pytest.mark.asyncio
    async def test_list_dir_traversal_blocked(self, temp_workspace):
        t = ListDirTool(workspace=temp_workspace)
        r = await t.execute(path=str(temp_workspace / ".." / ".."))
        assert "Security Error" in r

    @pytest.mark.asyncio
    async def test_list_dir_outside_blocked(self, temp_workspace):
        t = ListDirTool(workspace=temp_workspace)
        r = await t.execute(path="/etc")
        assert "Security Error" in r


class TestSymlinkSecurity:

    @pytest.mark.asyncio
    async def test_symlink_escape_blocked(self, temp_workspace):
        if os.name == "nt":
            pytest.skip("Symlink test requires admin on Windows")
        link = temp_workspace / "escape_link"
        try:
            link.symlink_to("/etc")
        except OSError:
            pytest.skip("Cannot create symlink")
        t = ListDirTool(workspace=temp_workspace)
        r = await t.execute(path=str(temp_workspace))
        if "escape_link" in r:
            assert "outside workspace" in r or "broken" in r


class TestPathEdgeCases:

    def test_empty_workspace(self):
        v = PathValidator()
        assert v.workspace == Path.cwd().resolve()

    def test_expanduser(self, temp_workspace):
        v = PathValidator(workspace=temp_workspace)
        r = v.validate_read_path("~/.ssh/id_rsa")
        assert r.valid is False

    def test_case_insensitive(self, temp_workspace):
        v = PathValidator(workspace=temp_workspace)
        for p in ["/ETC/PASSWD", "/Etc/Passwd", "/etc/PASSWD"]:
            r = v.validate_read_path(p)
            assert r.valid is False

    def test_dotenv_in_workspace_allowed(self, temp_workspace):
        (temp_workspace / ".env").write_text("API_KEY=test")
        v = PathValidator(workspace=temp_workspace)
        r = v.validate_read_path(temp_workspace / ".env")
        assert r.valid is True

    def test_credentials_outside_blocked(self, temp_workspace):
        v = PathValidator(workspace=temp_workspace)
        r = v.validate_read_path(Path.home() / ".aws" / "credentials")
        assert r.valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
