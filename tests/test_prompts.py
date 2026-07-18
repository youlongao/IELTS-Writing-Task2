"""Tests for prompt template management."""

import pytest
from unittest.mock import Mock, patch, mock_open

from src.prompts.templates import PromptManager, PromptTemplate


class TestPromptTemplate:
    """Test prompt template rendering."""

    def test_render_substitutes_variables(self):
        """render() should substitute {variables} in both system and user."""
        template = PromptTemplate(
            name="test",
            version="v1",
            system="You are a {role}.",
            user="Analyze: {question}",
        )

        result = template.render(role="teacher", question="Test question?")
        assert result["system"] == "You are a teacher."
        assert result["user"] == "Analyze: Test question?"

    def test_render_with_missing_variables(self):
        """Missing variables should remain as placeholders."""
        template = PromptTemplate(
            name="test",
            version="v1",
            system="Hello",
            user="Question: {question}. Topic: {topic}.",
        )

        result = template.render(question="Test?")
        assert "Test?" in result["user"]
        assert "{topic}" in result["user"]


class TestPromptManager:
    """Test prompt manager."""

    def test_get_caches_templates(self):
        """Second call to get() should return cached template."""
        import yaml

        yaml_content = {
            "system": "You are an IELTS tutor.",
            "user": "Analyze: {question}",
        }

        mock_file = mock_open(read_data=yaml.dump(yaml_content))

        with patch("builtins.open", mock_file), \
             patch("pathlib.Path.exists", return_value=True):
            manager = PromptManager(prompts_dir="/fake/prompts")

            template1 = manager.get("question_analyzer", "v1")
            template2 = manager.get("question_analyzer", "v1")

            assert template1 is template2

    def test_get_nonexistent_template_raises_error(self):
        """Getting a template that doesn't exist should raise FileNotFoundError."""
        with patch("pathlib.Path.exists", return_value=False):
            manager = PromptManager(prompts_dir="/fake/prompts")
            with pytest.raises(FileNotFoundError):
                manager.get("nonexistent", "v1")

    def test_list_templates(self):
        """Should discover templates from filesystem."""
        with patch("pathlib.Path.glob") as mock_glob, \
             patch("pathlib.Path.exists", return_value=True):
            mock_path = Mock()
            mock_path.stem = "v1_question_analyzer"
            mock_glob.return_value = [mock_path]

            manager = PromptManager(prompts_dir="/fake/prompts")
            templates = manager.list_templates()
            assert "question_analyzer" in templates

    def test_clear_cache(self):
        """clear_cache should force reload on next get."""
        import yaml

        yaml_content = {"system": "", "user": "Test: {q}"}
        mock_file = mock_open(read_data=yaml.dump(yaml_content))

        with patch("builtins.open", mock_file), \
             patch("pathlib.Path.exists", return_value=True):
            manager = PromptManager(prompts_dir="/fake/prompts")
            manager.get("question_analyzer", "v1")
            assert len(manager._cache) == 1

            manager.clear_cache()
            assert len(manager._cache) == 0

    def test_idea_generator_requires_bilingual_learning_output(self):
        """Idea prompt should ask for Chinese logic and English expression."""
        manager = PromptManager()
        template = manager.get("idea_generator", "v1")

        assert "Chinese-speaking learners" in template.system
        assert "Simplified Chinese" in template.user
        assert "English argument" in template.user

    def test_outline_builder_hides_internal_band_checklist(self):
        """Outline prompt should not expose the generated outline quality checklist."""
        manager = PromptManager()
        template = manager.get("outline_builder", "v1")

        assert "Do not expose internal scoring checklists" in template.system
        assert "Student's Selected Writing Path" in template.user
        assert 'Do not include a "Band 7 readiness checklist"' in template.user

    def test_essay_evaluator_checks_idea_quality_and_actual_band(self):
        """Essay evaluator prompt should cover student essay scoring."""
        manager = PromptManager()
        template = manager.get("essay_evaluator", "v1")

        assert "Idea Quality Check" in template.user
        assert "Actual Band Estimate" in template.user
        assert "Do not use the learner's target band as a score floor" in template.system
        assert "Band 0.0 to Band 9.0" in template.system
        for band in range(10):
            assert f"Band {band}:" in template.system
