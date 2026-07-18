"""Results reporter — formats harness evaluation results for display."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from .metrics import MetricScores

logger = logging.getLogger(__name__)


class Reporter:
    """Formats and outputs harness evaluation results.

    Supports: terminal tables, JSON export, CSV export.
    """

    @staticmethod
    def terminal_summary(results: List[Dict]) -> str:
        """Generate a terminal-friendly summary of evaluation results.

        Args:
            results: List of evaluation result dicts.

        Returns:
            Formatted string for terminal output.
        """
        if not results:
            return "No results to display."

        # Calculate aggregate statistics
        valid_results = [r for r in results if not r.get("error")]
        error_count = len(results) - len(valid_results)

        lines = [
            "=" * 60,
            "  IELTS Writing Task 2 — Harness Evaluation Results",
            "=" * 60,
            f"",
            f"  Total test cases: {len(results)}",
            f"  Successful: {len(valid_results)}",
            f"  Errors: {error_count}",
            f"",
            "  ── Aggregate Scores ──",
            "",
        ]

        metrics = [
            "task_response",
            "coherence_cohesion",
            "lexical_resource",
            "grammatical_range",
            "specificity_score",
            "band_alignment",
            "overall",
        ]

        for metric in metrics:
            scores = [
                r["scores"].get(metric, 0) for r in valid_results
            ]
            if not scores:
                continue

            avg = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            # Determine if this meets Band 7 target
            status = "✅" if avg >= 6.5 else "⚠️" if avg >= 6.0 else "❌"

            lines.append(
                f"  {status} {metric:25s}  "
                f"Avg: {avg:.1f}  "
                f"Min: {min_score:.1f}  "
                f"Max: {max_score:.1f}"
            )

        lines.extend([
            "",
            "  ── Individual Results ──",
            "",
        ])

        for r in valid_results:
            case_id = r.get("test_case_id", "unknown")
            overall = r["scores"].get("overall", 0.0)
            latency = r.get("latency_seconds", 0)
            lines.append(
                f"  {case_id:20s}  Overall: {overall:.1f}  "
                f"Latency: {latency:.1f}s"
            )

        if error_count > 0:
            lines.extend(["", "  ── Errors ──", ""])
            for r in results:
                if r.get("error"):
                    lines.append(
                        f"  ❌ {r.get('test_case_id', 'unknown')}: {r['error']}"
                    )

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    @staticmethod
    def to_json(results: List[Dict], output_path: str):
        """Export results as JSON file.

        Args:
            results: List of evaluation result dicts.
            output_path: Path to output JSON file.
        """
        output = {
            "summary": {
                "total": len(results),
                "successful": len([r for r in results if not r.get("error")]),
                "errors": len([r for r in results if r.get("error")]),
            },
            "results": results,
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"Results exported to: {output_path}")

    @staticmethod
    def to_csv(results: List[Dict], output_path: str):
        """Export results as CSV file.

        Args:
            results: List of evaluation result dicts.
            output_path: Path to output CSV file.
        """
        import csv

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "test_case_id",
            "task_response",
            "coherence_cohesion",
            "lexical_resource",
            "grammatical_range",
            "specificity_score",
            "band_alignment",
            "overall",
            "latency_seconds",
            "error",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                scores = r.get("scores", {})
                writer.writerow({
                    "test_case_id": r.get("test_case_id", ""),
                    "task_response": scores.get("task_response", ""),
                    "coherence_cohesion": scores.get("coherence_cohesion", ""),
                    "lexical_resource": scores.get("lexical_resource", ""),
                    "grammatical_range": scores.get("grammatical_range", ""),
                    "specificity_score": scores.get("specificity_score", ""),
                    "band_alignment": scores.get("band_alignment", ""),
                    "overall": scores.get("overall", ""),
                    "latency_seconds": r.get("latency_seconds", ""),
                    "error": r.get("error", ""),
                })

        logger.info(f"CSV exported to: {output_path}")

    @staticmethod
    def comparison_table(comparison_result) -> str:
        """Generate a comparison table for A/B test results.

        Args:
            comparison_result: ComparisonResult from Comparator.

        Returns:
            Formatted string for terminal output.
        """
        result = comparison_result
        lines = [
            "",
            "=" * 70,
            f"  A/B Comparison: {result.config_a_name} vs {result.config_b_name}",
            "=" * 70,
            f"",
            f"  Test cases: {result.test_case_count}",
            f"  Overall winner: {result.overall_winner}",
            f"",
            f"  {'Metric':30s} {'A':>6s} {'B':>6s} {'Diff':>8s}  Winner",
            f"  {'─' * 30} {'─' * 6} {'─' * 6} {'─' * 8}  {'─' * 6}",
        ]

        for mc in result.metric_comparisons:
            winner_label = (
                f"← {result.config_a_name}" if mc["winner"] == "A"
                else f"← {result.config_b_name}" if mc["winner"] == "B"
                else "tie"
            )
            lines.append(
                f"  {mc['metric']:30s} "
                f"{mc['avg_a']:6.2f} {mc['avg_b']:6.2f} "
                f"{mc['difference']:+8.2f}  {winner_label}"
            )

        lines.extend(["", "=" * 70])
        return "\n".join(lines)
