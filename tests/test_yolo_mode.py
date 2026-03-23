"""Tests for YOLO mode feature.

Verifies that:
1. Config validates yolo_mode correctly
2. WorkspaceFSService resolves paths in yolo mode
3. ContextBuilder includes YOLO banner in system prompt
4. Agent can read files in/outside workspace in yolo mode
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from spoon_bot.config import AgentLoopConfig, validate_agent_loop_params


class TestYoloConfig:
    """Test YOLO mode configuration validation."""

    def test_yolo_mode_defaults_to_false(self):
        cfg = AgentLoopConfig()
        assert cfg.yolo_mode is False

    def test_yolo_mode_enabled_requires_existing_dir(self, tmp_path):
        cfg = AgentLoopConfig(workspace=tmp_path, yolo_mode=True)
        assert cfg.yolo_mode is True
        assert cfg.workspace == tmp_path

    def test_yolo_mode_rejects_nonexistent_dir(self):
        fake_path = Path("/nonexistent/path/that/does/not/exist")
        with pytest.raises(ValueError, match="does not exist"):
            AgentLoopConfig(workspace=fake_path, yolo_mode=True)

    def test_validate_agent_loop_params_yolo_defaults_workspace_to_cwd(self):
        cfg = validate_agent_loop_params(yolo_mode=True)
        assert cfg.yolo_mode is True
        assert cfg.workspace == Path.cwd()

    def test_validate_agent_loop_params_yolo_with_explicit_workspace(self, tmp_path):
        cfg = validate_agent_loop_params(workspace=tmp_path, yolo_mode=True)
        assert cfg.workspace == tmp_path
        assert cfg.yolo_mode is True


class TestYoloWorkspaceFS:
    """Test WorkspaceFSService in YOLO mode."""

    def test_yolo_resolve_relative_path(self, tmp_path):
        from spoon_bot.gateway.websocket.workspace_fs import WorkspaceFSService

        (tmp_path / "hello.txt").write_text("world")
        svc = WorkspaceFSService(workspace_root=tmp_path, yolo_mode=True)
        result_sync = svc._stat_sync("hello.txt")
        assert result_sync["type"] == "file"

    def test_yolo_resolve_absolute_within_workspace(self, tmp_path):
        from spoon_bot.gateway.websocket.workspace_fs import WorkspaceFSService

        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "data.txt").write_text("content")
        svc = WorkspaceFSService(workspace_root=tmp_path, yolo_mode=True)
        abs_path = str(sub / "data.txt")
        result = svc._stat_sync(abs_path)
        assert result["type"] == "file"

    def test_yolo_rejects_path_outside_workspace(self, tmp_path):
        from spoon_bot.gateway.websocket.workspace_fs import WorkspaceFSService

        svc = WorkspaceFSService(workspace_root=tmp_path, yolo_mode=True)
        with pytest.raises(ValueError, match="outside workspace"):
            svc._stat_sync("/etc/passwd")

    def test_yolo_read_file(self, tmp_path):
        from spoon_bot.gateway.websocket.workspace_fs import WorkspaceFSService

        test_file = tmp_path / "test.txt"
        test_file.write_text("YOLO content!")
        svc = WorkspaceFSService(workspace_root=tmp_path, yolo_mode=True)
        result = svc._read_sync("test.txt", encoding="utf-8", offset=0, limit=4096)
        assert result["content"] == "YOLO content!"

    def test_yolo_write_file(self, tmp_path):
        from spoon_bot.gateway.websocket.workspace_fs import WorkspaceFSService

        svc = WorkspaceFSService(workspace_root=tmp_path, yolo_mode=True)
        result = svc._write_sync(
            "output.txt",
            content="hello from yolo",
            encoding="utf-8",
            create=True,
            truncate=True,
        )
        assert (tmp_path / "output.txt").read_text() == "hello from yolo"

    def test_non_yolo_sandbox_prefix_still_works(self, tmp_path):
        from spoon_bot.gateway.websocket.workspace_fs import WorkspaceFSService

        (tmp_path / "a.txt").write_text("data")
        svc = WorkspaceFSService(workspace_root=tmp_path, yolo_mode=False)
        result = svc._stat_sync("/workspace/a.txt")
        assert result["type"] == "file"


class TestYoloContextBuilder:
    """Test ContextBuilder YOLO mode banner."""

    def test_yolo_banner_present(self, tmp_path):
        from spoon_bot.agent.context import ContextBuilder

        ctx = ContextBuilder(tmp_path, yolo_mode=True)
        prompt = ctx.build_system_prompt()
        assert "YOLO MODE ACTIVE" in prompt

    def test_no_yolo_banner_by_default(self, tmp_path):
        from spoon_bot.agent.context import ContextBuilder

        ctx = ContextBuilder(tmp_path, yolo_mode=False)
        prompt = ctx.build_system_prompt()
        assert "YOLO MODE ACTIVE" not in prompt
