"""Tests for harness evaluation framework."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.harness.metrics import MetricsCalculator, MetricScores
from src.harness.evaluator import Evaluator
from src.harness.test_cases import TestCaseManager


class TestMetricsCalculator:
    """Test metrics calculation."""

    def test_calculate_overall_average(self):
        """Overall should be the rounded average of all scores."""
        scores = {
            "task_response": 7.0,
            "coherence_cohesion": 6.5,
            "lexical_resource": 6.0,
            "grammatical_range": 7.0,
        }
        overall = MetricsCalculator.calculate_overall(scores)
        # Average = 6.625, rounded to nearest 0.5 = 6.5
        assert overall == 6.5

    def test_calculate_overall_perfect_scores(self):
        """All 9s should result in overall 9.0."""
        scores = {k: 9.0 for k in ["task_response", "coherence_cohesion", "lexical_resource", "grammatical_range"]}
        overall = MetricsCalculator.calculate_overall(scores)
        assert overall == 9.0

    def test_calculate_overall_empty(self):
        """Empty scores should return 0."""
        overall = MetricsCalculator.calculate_overall({})
        assert overall == 0.0

    def test_band_7_descriptors_exist(self):
        """All four IELTS criteria should have descriptors."""
        descriptors = MetricsCalculator.BAND_7_DESCRIPTORS
        assert "task_response" in descriptors
        assert "coherence_cohesion" in descriptors
        assert "lexical_resource" in descriptors
        assert "grammatical_range" in descriptors

    def test_get_judge_prompt(self):
        """Judge prompt should include question, output, and descriptor."""
        prompt = MetricsCalculator.get_judge_prompt(
            question="Test question?",
            output="Test output.",
            metric="task_response",
        )
        assert "Test question?" in prompt
        assert "Test output." in prompt
        assert "task_response" in prompt.lower()
        assert "Band 7" in prompt


class TestEvaluator:
    """Test evaluator."""

    def test_extract_score_from_response(self):
        """Should extract band score from judge response."""
        response = "Score: 7.5\nJustification: The output meets Band 7 standards."
        score = Evaluator._extract_score(response)
        assert score == 7.5

    def test_extract_score_with_band_prefix(self):
        """Should handle 'Band X' format."""
        response = "Score: Band 6.5\nJustification: Good but needs improvement."
        score = Evaluator._extract_score(response)
        assert score == 6.5

    def test_extract_score_with_slash_format(self):
        """Should handle 'X/9' format."""
        response = "The essay scores 7.0/9 overall."
        score = Evaluator._extract_score(response)
        assert score == 7.0

    def test_extract_score_fallback(self):
        """Should return 5.0 if no score found."""
        response = "This is a well-written essay with good structure."
        score = Evaluator._extract_score(response)
        assert score == 5.0

    def test_calculate_specificity(self):
        """Specific text should score higher than vague text."""
        specific = "For instance, according to a 2023 study by the WHO, 60% of urban residents..."
        vague = "Some people believe that many things are important in general..."

        specific_score = Evaluator._calculate_specificity(specific)
        vague_score = Evaluator._calculate_specificity(vague)

        assert specific_score > vague_score

    def test_evaluate_returns_metric_scores(self):
        """evaluate() should return MetricScores with all dimensions."""
        mock_judge = Mock()
        mock_judge.generate.return_value = "Score: 7.0\nJustification: Meets standards."

        evaluator = Evaluator(judge_llm=mock_judge)
        result = evaluator.evaluate("Test question?", "Test output.")

        assert isinstance(result, MetricScores)
        assert result.overall > 0


class TestTestCaseManager:
    """Test case manager."""

    @patch("src.harness.test_cases.Path.exists")
    def test_loads_test_cases(self, mock_exists):
        """Should load test cases from JSON file."""
        import json
        mock_exists.return_value = True

        mock_data = {
            "questions": [
                {"id": "test_1", "question": "Test?", "question_type": "opinion", "topics": ["education"]},
                {"id": "test_2", "question": "Test 2?", "question_type": "discussion", "topics": ["technology"]},
            ]
        }

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)

            manager = TestCaseManager(data_dir="/fake/path")
            assert manager.count == 2

    def test_get_by_question_type(self):
        """get_by_question_type should filter correctly."""
        import json

        mock_data = {
            "questions": [
                {"id": "1", "question": "Q1?", "question_type": "opinion", "topics": ["education"]},
                {"id": "2", "question": "Q2?", "question_type": "discussion", "topics": ["technology"]},
            ]
        }

        with patch("src.harness.test_cases.Path.exists", return_value=True), \
             patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)

            manager = TestCaseManager(data_dir="/fake/path")
            opinion_cases = manager.get_by_question_type("opinion")
            assert len(opinion_cases) == 1
            assert opinion_cases[0]["id"] == "1"
