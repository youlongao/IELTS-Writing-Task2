"""Tests for LLM provider abstraction layer."""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

# Import directly from submodules to avoid triggering heavy deps in __init__
from src.llm.base import LLMProvider
from src.llm.factory import LLMFactory


class TestLLMProvider:
    """Test the abstract LLM provider interface."""

    def test_abstract_class_cannot_instantiate(self):
        """LLMProvider is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider(model_name="test")


class TestLLMFactory:
    """Test the LLM factory."""

    def test_create_unknown_provider_raises_error(self):
        """Creating an unknown provider should raise ValueError."""
        mock_config = Mock()
        with pytest.raises(ValueError, match="Unknown provider"):
            LLMFactory.create("unknown_provider", mock_config)

    def test_create_openai_without_api_key_raises_error(self):
        """Creating OpenAI provider without API key should raise ValueError."""
        mock_config = Mock()
        mock_config.openai_api_key = None
        mock_config.get_model_config.return_value = {
            "model_name": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 100,
        }
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            LLMFactory.create("openai", mock_config)

    def test_create_deepseek_without_api_key_raises_error(self):
        """Creating DeepSeek provider without API key should raise ValueError."""
        mock_config = Mock()
        mock_config.deepseek_api_key = None
        mock_config.get_model_config.return_value = {
            "model_name": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 100,
            "base_url": "https://api.deepseek.com",
        }
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            LLMFactory.create("deepseek", mock_config)

    def test_create_openai_with_api_key(self):
        """Creating OpenAI provider with valid API key succeeds."""
        # Create mock openai module in sys.modules (lazy import will find it)
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        sys.modules["openai"] = mock_openai

        try:
            mock_config = Mock()
            mock_config.openai_api_key = "sk-test-key"
            mock_config.get_model_config.return_value = {
                "model_name": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 100,
            }

            provider = LLMFactory.create("openai", mock_config)
            assert provider is not None
            assert provider.model_name == "gpt-4o"
        finally:
            del sys.modules["openai"]

    def test_create_deepseek_with_api_key(self):
        """Creating DeepSeek provider with valid API key succeeds."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        sys.modules["openai"] = mock_openai

        try:
            mock_config = Mock()
            mock_config.deepseek_api_key = "deepseek-test-key"
            mock_config.get_model_config.return_value = {
                "model_name": "deepseek-chat",
                "temperature": 0.7,
                "max_tokens": 100,
                "base_url": "https://api.deepseek.com",
            }

            provider = LLMFactory.create("deepseek", mock_config)
            assert provider is not None
            assert provider.model_name == "deepseek-chat"
            mock_openai.OpenAI.assert_called_once_with(
                api_key="deepseek-test-key",
                base_url="https://api.deepseek.com",
            )
        finally:
            del sys.modules["openai"]

    def test_model_overrides(self):
        """Model overrides should take precedence over config."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        sys.modules["openai"] = mock_openai

        try:
            mock_config = Mock()
            mock_config.openai_api_key = "sk-test-key"
            mock_config.get_model_config.return_value = {
                "model_name": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 100,
            }

            provider = LLMFactory.create(
                "openai",
                mock_config,
                model_overrides={"model_name": "gpt-4o-mini", "temperature": 0.3},
            )
            assert provider.model_name == "gpt-4o-mini"
            assert provider.temperature == 0.3
        finally:
            del sys.modules["openai"]
