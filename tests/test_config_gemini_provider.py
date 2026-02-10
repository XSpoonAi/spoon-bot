"""Tests for Gemini and generic provider support in SpoonBotConfig.from_env()."""

import os
from unittest.mock import patch

from spoon_bot.core import SpoonBotConfig


def test_from_env_gemini_provider():
    env = {
        "SPOON_PROVIDER": "gemini",
        "SPOON_MODEL": "gemini-3-flash-preview",
        "GEMINI_API_KEY": "test-gemini-key-123",
    }
    with patch.dict(os.environ, env, clear=False):
        config = SpoonBotConfig.from_env()
    assert config.provider == "gemini"
    assert config.model == "gemini-3-flash-preview"
    assert config.api_key == "test-gemini-key-123"
    # Gemini uses SDK default; no base_url unless explicitly set
    assert config.base_url is None


def test_from_env_gemini_with_base_url():
    env = {
        "SPOON_PROVIDER": "gemini",
        "SPOON_MODEL": "gemini-3-flash-preview",
        "GEMINI_API_KEY": "test-key",
        "GEMINI_BASE_URL": "https://custom-gemini-proxy.example.com",
    }
    with patch.dict(os.environ, env, clear=False):
        config = SpoonBotConfig.from_env()
    assert config.provider == "gemini"
    assert config.base_url == "https://custom-gemini-proxy.example.com"


def test_from_env_generic_provider_key():
    """Unknown providers use {PROVIDER}_API_KEY convention."""
    env = {
        "SPOON_PROVIDER": "deepseek",
        "SPOON_MODEL": "deepseek-chat",
        "DEEPSEEK_API_KEY": "dk-test-key",
    }
    with patch.dict(os.environ, env, clear=False):
        config = SpoonBotConfig.from_env()
    assert config.provider == "deepseek"
    assert config.api_key == "dk-test-key"


def test_from_env_openai_still_works():
    """Regression: existing OpenAI config must not break."""
    env = {
        "SPOON_PROVIDER": "openai",
        "SPOON_MODEL": "gpt-5.3-codex",
        "OPENAI_API_KEY": "sk-test-key",
        "OPENAI_BASE_URL": "https://proxy.example.com/v1",
    }
    with patch.dict(os.environ, env, clear=False):
        config = SpoonBotConfig.from_env()
    assert config.provider == "openai"
    assert config.api_key == "sk-test-key"
    assert config.base_url == "https://proxy.example.com/v1"


def test_from_env_fallback_when_provider_key_missing():
    """When provider-specific key is absent, fall back to ANTHROPIC/OPENAI keys."""
    env = {
        "SPOON_PROVIDER": "gemini",
        "SPOON_MODEL": "gemini-3-flash-preview",
        "OPENAI_API_KEY": "fallback-openai-key",
    }
    # Must explicitly remove provider-specific and anthropic keys from real env
    with patch.dict(os.environ, env, clear=False):
        os.environ.pop("GEMINI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        config = SpoonBotConfig.from_env()
    assert config.provider == "gemini"
    assert config.api_key == "fallback-openai-key"
