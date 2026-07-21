"""Tests for harness evaluation framework."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.harness.metrics import MetricsCalculator, MetricScores
from src.harness.evaluator import Evaluator, _extract_essay_body
from src.harness.reporter import Reporter
from src.harness.compare import compare_results, format_comparison_report
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

    def test_calculate_overall_ignores_diagnostic_metrics(self):
        """Overall band should only use the four official IELTS criteria."""
        scores = {
            "task_response": 6.0,
            "coherence_cohesion": 6.0,
            "lexical_resource": 5.0,
            "grammatical_range": 5.0,
            "specificity_score": 9.0,
            "band_alignment": 9.0,
        }
        overall = MetricsCalculator.calculate_overall(scores)

        assert overall == 5.5

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

    def test_band_alignment_uses_official_scores_without_bonus(self):
        """Band alignment should not pass when official scores are low."""
        scores = {
            "lexical_resource": 4.7,
            "grammatical_range": 4.7,
        }

        assert Evaluator._calculate_band_alignment(scores) == 4.5

    def test_evaluate_returns_metric_scores(self):
        """evaluate() should return MetricScores with all dimensions."""
        mock_judge = Mock()
        mock_judge.generate.return_value = "Score: 7.0\nJustification: Meets standards."

        evaluator = Evaluator(judge_llm=mock_judge)
        result = evaluator.evaluate("Test question?", "Test output.")

        assert isinstance(result, MetricScores)
        assert result.overall > 0

    def test_batch_rule_evaluate_returns_rule_results(self):
        """batch_rule_evaluate should generate rule-only harness results."""
        cases = [{
            "id": "case_1",
            "question": "Some people think online education is effective. Discuss both views.",
        }]
        output = (
            "Question Type: Discussion\n"
            "Position: balanced view\n"
            "Main Idea 1: Online education increases access.\n"
            "Main Idea 2: It can reduce classroom interaction.\n"
            "For example, rural students can join remote courses.\n"
            "Suggested Outline: Introduction, Body 1, Body 2, Conclusion."
        )

        results = Evaluator.batch_rule_evaluate(cases, lambda _: output)

        assert len(results) == 1
        assert results[0]["error"] is None
        assert results[0]["scores"]["rule_score"] > 0
        assert results[0]["scores"]["rule_checks"]["has_outline"] is True

    def test_reporter_terminal_summary_is_ascii_readable(self):
        """Reporter output should avoid mojibake in terminal summaries."""
        results = [{
            "test_case_id": "case_1",
            "scores": {
                "overall": 7.0,
                "rule_score": 7.7,
                "rule_checks": {"has_outline": True},
            },
            "latency_seconds": 1.0,
            "error": None,
        }]

        summary = Reporter.terminal_summary(results)

        assert "IELTS Writing Task 2 - Harness Evaluation Results" in summary
        assert "Aggregate Scores" in summary
        assert "鈥" not in summary

    def test_extract_essay_body_removes_study_notes(self):
        """Harness should judge the essay body, not bilingual study notes."""
        sample_output = (
            "This is the first paragraph.\n\n"
            "This is the conclusion.\n\n"
            "---\n"
            "## Vocabulary Notes\n"
            "- phrase: Chinese explanation"
        )

        body = _extract_essay_body(sample_output)

        assert "This is the conclusion." in body
        assert "Vocabulary Notes" not in body


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


class TestHarnessCompare:
    """Test fixed-judge provider comparison helpers."""

    def test_compare_results_picks_higher_overall_winner(self):
        results_a = [{
            "test_case_id": "case_1",
            "user_input": "Test question?",
            "scores": {"overall": 7.0, "task_response": 7.0},
            "error": None,
            "error_labels": [],
        }]
        results_b = [{
            "test_case_id": "case_1",
            "user_input": "Test question?",
            "scores": {"overall": 6.0, "task_response": 6.0},
            "error": None,
            "error_labels": [],
        }]

        comparison = compare_results("deepseek", results_a, "openai", results_b, "openai")

        assert comparison["overall_winner"] == "deepseek"
        assert comparison["winner_counts"]["deepseek"] == 1
        assert comparison["metric_comparisons"][0]["winner"] == "deepseek"

    def test_format_comparison_report_shows_fixed_judge(self):
        comparison = {
            "provider_a": "deepseek",
            "provider_b": "openai",
            "judge_provider": "openai",
            "test_case_count": 1,
            "overall_winner": "tie",
            "winner_counts": {"deepseek": 0, "openai": 0, "tie": 1},
            "metric_comparisons": [],
            "case_comparisons": [{
                "test_case_id": "case_1",
                "question": "Test question?",
                "provider_a_overall": 6.5,
                "provider_b_overall": 6.5,
                "difference": 0.0,
                "winner": "tie",
            }],
        }

        report = format_comparison_report(comparison)

        assert "Fixed judge: openai" in report
        assert "Provider A: deepseek" in report
        assert "Provider B: openai" in report
