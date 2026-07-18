"""Shared test fixtures for all tests."""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider for testing."""
    llm = Mock()
    llm.model_name = "test-model"
    llm.generate.return_value = "Mock LLM response with analysis and arguments."
    llm.generate_structured.return_value = {"result": "mock"}
    return llm


@pytest.fixture
def mock_config():
    """Create a mock Config for testing."""
    config = Mock()
    config.default_provider = "openai"
    config.openai_api_key = "sk-test-key"
    config.deepseek_api_key = "deepseek-test-key"
    config.chroma_persist_dir = "/tmp/test_chroma"
    config.chunk_size = 500
    config.chunk_overlap = 50

    def mock_get(key_path, default=None):
        defaults = {
            "rag.embedding_model": "test-model",
            "rag.chunk_size": 500,
            "rag.chunk_overlap": 50,
            "rag.chroma_persist_dir": "/tmp/test_chroma",
            "app.default_provider": "openai",
        }
        return defaults.get(key_path, default)

    config.get.side_effect = mock_get
    config.get_model_config.return_value = {
        "model_name": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": 100,
    }
    return config


@pytest.fixture
def sample_question():
    """A standard IELTS Task 2 question for testing."""
    return (
        "Some people think governments should invest more in public transport "
        "rather than roads. To what extent do you agree or disagree?"
    )


@pytest.fixture
def sample_state(sample_question):
    """Create a basic WorkflowState for testing."""
    from src.agent.state import WorkflowState, Scenario
    return WorkflowState(
        user_input=sample_question,
        scenario=Scenario.GENERATE,
    )
