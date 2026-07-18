"""A/B comparison engine for comparing models, prompts, and RAG configs."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .evaluator import Evaluator
from .test_cases import TestCaseManager

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of an A/B comparison."""

    config_a_name: str
    config_b_name: str
    test_case_count: int
    metric_comparisons: List[Dict]
    winner_summary: Dict[str, str]  # metric -> "A" | "B" | "tie"
    overall_winner: str  # "A" | "B" | "tie"


class Comparator:
    """Compares two agent configurations (models, prompts, RAG setups).

    Runs both configurations against the same test cases and compares
    their evaluation scores side by side.

    Usage:
        comparator = Comparator(evaluator, test_cases)
        result = comparator.compare(
            config_a={"name": "GPT-4o + v1 prompts", "pipeline": pipeline_a},
            config_b={"name": "DeepSeek + v1 prompts", "pipeline": pipeline_b},
        )
    """

    def __init__(self, evaluator: Evaluator, test_case_manager: TestCaseManager):
        """Initialize the comparator.

        Args:
            evaluator: Evaluator instance for scoring outputs.
            test_case_manager: TestCaseManager with question bank.
        """
        self.evaluator = evaluator
        self.test_cases = test_case_manager

    def compare(
        self,
        config_a: Dict[str, Any],
        config_b: Dict[str, Any],
        test_count: int = 10,
        topic: Optional[str] = None,
    ) -> ComparisonResult:
        """Run A/B comparison between two configurations.

        Args:
            config_a: Dict with 'name' (str) and 'pipeline' (callable).
                      pipeline(question: str) -> output: str
            config_b: Same format as config_a.
            test_count: Number of test cases to use.
            topic: Optional topic filter.

        Returns:
            ComparisonResult with detailed per-metric comparison.
        """
        cases = self.test_cases.sample(test_count, topic)

        logger.info(
            f"Comparing '{config_a['name']}' vs '{config_b['name']}' "
            f"on {len(cases)} test cases"
        )

        # Run evaluation for config A
        logger.info(f"Running config A: {config_a['name']}")
        results_a = self.evaluator.batch_evaluate(cases, config_a["pipeline"])

        # Run evaluation for config B
        logger.info(f"Running config B: {config_b['name']}")
        results_b = self.evaluator.batch_evaluate(cases, config_b["pipeline"])

        # Compare per metric
        metrics = [
            "task_response",
            "coherence_cohesion",
            "lexical_resource",
            "grammatical_range",
            "specificity_score",
            "overall",
        ]

        metric_comparisons = []
        winner_counts = {"A": 0, "B": 0, "tie": 0}

        for metric in metrics:
            scores_a = [
                r["scores"].get(metric, 0) for r in results_a if not r.get("error")
            ]
            scores_b = [
                r["scores"].get(metric, 0) for r in results_b if not r.get("error")
            ]

            if not scores_a or not scores_b:
                continue

            avg_a = sum(scores_a) / len(scores_a)
            avg_b = sum(scores_b) / len(scores_b)
            diff = avg_a - avg_b

            winner = "A" if diff > 0.1 else "B" if diff < -0.1 else "tie"
            winner_counts[winner] = winner_counts.get(winner, 0) + 1

            metric_comparisons.append({
                "metric": metric,
                "avg_a": round(avg_a, 2),
                "avg_b": round(avg_b, 2),
                "difference": round(diff, 2),
                "winner": winner,
            })

        # Determine overall winner
        if winner_counts["A"] > winner_counts["B"]:
            overall = "A"
        elif winner_counts["B"] > winner_counts["A"]:
            overall = "B"
        else:
            overall = "tie"

        return ComparisonResult(
            config_a_name=config_a["name"],
            config_b_name=config_b["name"],
            test_case_count=len(cases),
            metric_comparisons=metric_comparisons,
            winner_summary={
                m["metric"]: m["winner"] for m in metric_comparisons
            },
            overall_winner=overall,
        )
