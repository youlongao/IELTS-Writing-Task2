"""Results reporter - formats harness evaluation results for display."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class Reporter:
    """Formats and outputs harness evaluation results.

    Supports: terminal summaries, JSON export, CSV export.
    """

    @staticmethod
    def terminal_summary(results: List[Dict]) -> str:
        """Generate a terminal-friendly summary of evaluation results."""
        if not results:
            return "No results to display."

        valid_results = [r for r in results if not r.get("error")]
        error_count = len(results) - len(valid_results)
        input_sources = sorted({r.get("input_source", "unknown") for r in results})
        harness_targets = sorted({r.get("harness_target", "unknown") for r in results})
        generation_models = sorted({
            r.get("model_versions", {}).get("generation", "")
            for r in valid_results
            if r.get("model_versions", {}).get("generation")
        })
        judge_models = sorted({
            r.get("model_versions", {}).get("judge", "")
            for r in valid_results
            if r.get("model_versions", {}).get("judge")
        })
        token_totals = Reporter._sum_token_usage(valid_results)

        lines = [
            "=" * 60,
            "  IELTS Writing Task 2 - Harness Evaluation Results",
            "=" * 60,
            "",
            f"  Total test cases: {len(results)}",
            f"  Successful: {len(valid_results)}",
            f"  Errors: {error_count}",
            f"  Harness target: {', '.join(harness_targets)}",
            f"  Input source: {', '.join(input_sources)}",
            f"  Generation model: {', '.join(generation_models) or 'unknown'}",
            f"  Judge model: {', '.join(judge_models) or 'n/a'}",
            (
                "  Token usage: "
                f"generation={token_totals['generation_total_tokens']} total, "
                f"judge={token_totals['judge_total_tokens']} total"
            ),
            "",
            "  -- Aggregate Scores --",
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
            "rule_score",
        ]

        for metric in metrics:
            scores = [
                r["scores"].get(metric, 0)
                for r in valid_results
                if metric in r.get("scores", {})
            ]
            if not scores:
                continue

            avg = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            status = "PASS" if avg >= 6.5 else "WARN" if avg >= 6.0 else "FAIL"

            lines.append(
                f"  {status:4s} {metric:25s}  "
                f"Avg: {avg:.1f}  "
                f"Min: {min_score:.1f}  "
                f"Max: {max_score:.1f}"
            )

        rule_results = [
            r for r in valid_results
            if "rule_checks" in r.get("scores", {})
        ]
        if rule_results:
            lines.extend(["", "  -- Rule Check Pass Rates --", ""])
            rule_names = sorted({
                name
                for r in rule_results
                for name in r["scores"]["rule_checks"].keys()
            })
            for rule_name in rule_names:
                passed = sum(
                    1
                    for r in rule_results
                    if r["scores"]["rule_checks"].get(rule_name)
                )
                rate = passed / len(rule_results) * 100
                status = "PASS" if rate >= 80 else "WARN" if rate >= 60 else "FAIL"
                lines.append(f"  {status:4s} {rule_name:25s}  {rate:5.1f}%")

        error_labels = Reporter._count_error_labels(valid_results)
        if error_labels:
            lines.extend(["", "  -- Error Labels --", ""])
            for label, count in sorted(error_labels.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"  {label:30s}  {count}")

        lines.extend(["", "  -- Individual Results --", ""])

        for r in valid_results:
            case_id = r.get("test_case_id", "unknown")
            overall = r["scores"].get("overall", 0.0)
            latency = r.get("latency_seconds", 0)
            question = r.get("user_input", r.get("question", "")).replace("\n", " ")
            question_preview = question[:90] + ("..." if len(question) > 90 else "")
            lines.append(
                f"  {case_id:20s}  Overall: {overall:.1f}  "
                f"Latency: {latency:.1f}s"
            )
            lines.append(f"    Input: {question_preview}")

        if error_count > 0:
            lines.extend(["", "  -- Errors --", ""])
            for r in results:
                if r.get("error"):
                    lines.append(
                        f"  ERROR {r.get('test_case_id', 'unknown')}: {r['error']}"
                    )

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    @staticmethod
    def _sum_token_usage(results: List[Dict]) -> Dict[str, int]:
        totals = {
            "generation_total_tokens": 0,
            "judge_total_tokens": 0,
        }
        for result in results:
            token_usage = result.get("token_usage", {})
            generation = token_usage.get("generation", {})
            judge = token_usage.get("judge", {})
            totals["generation_total_tokens"] += int(generation.get("total_tokens", 0) or 0)
            totals["judge_total_tokens"] += int(judge.get("total_tokens", 0) or 0)
        return totals

    @staticmethod
    def _count_error_labels(results: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for result in results:
            for label in result.get("error_labels", []):
                counts[label] = counts.get(label, 0) + 1
        return counts

    @staticmethod
    def to_json(results: List[Dict], output_path: str):
        """Export results as JSON file."""
        output = {
            "run_metadata": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "run_type": "single",
            },
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
        """Export results as CSV file."""
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
            "rule_score",
            "rule_passed",
            "rule_total",
            "latency_seconds",
            "input_source",
            "generation_model",
            "judge_model",
            "generation_total_tokens",
            "judge_total_tokens",
            "error_labels",
            "error",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                scores = r.get("scores", {})
                token_usage = r.get("token_usage", {})
                model_versions = r.get("model_versions", {})
                writer.writerow({
                    "test_case_id": r.get("test_case_id", ""),
                    "task_response": scores.get("task_response", ""),
                    "coherence_cohesion": scores.get("coherence_cohesion", ""),
                    "lexical_resource": scores.get("lexical_resource", ""),
                    "grammatical_range": scores.get("grammatical_range", ""),
                    "specificity_score": scores.get("specificity_score", ""),
                    "band_alignment": scores.get("band_alignment", ""),
                    "overall": scores.get("overall", ""),
                    "rule_score": scores.get("rule_score", ""),
                    "rule_passed": scores.get("rule_passed", ""),
                    "rule_total": scores.get("rule_total", ""),
                    "latency_seconds": r.get("latency_seconds", ""),
                    "input_source": r.get("input_source", ""),
                    "generation_model": model_versions.get("generation", ""),
                    "judge_model": model_versions.get("judge", ""),
                    "generation_total_tokens": token_usage.get("generation", {}).get("total_tokens", ""),
                    "judge_total_tokens": token_usage.get("judge", {}).get("total_tokens", ""),
                    "error_labels": "|".join(r.get("error_labels", [])),
                    "error": r.get("error", ""),
                })

        logger.info(f"CSV exported to: {output_path}")

    @staticmethod
    def comparison_table(comparison_result) -> str:
        """Generate a comparison table for A/B test results."""
        result = comparison_result
        lines = [
            "",
            "=" * 70,
            f"  A/B Comparison: {result.config_a_name} vs {result.config_b_name}",
            "=" * 70,
            "",
            f"  Test cases: {result.test_case_count}",
            f"  Overall winner: {result.overall_winner}",
            "",
            f"  {'Metric':30s} {'A':>6s} {'B':>6s} {'Diff':>8s}  Winner",
            f"  {'-' * 30} {'-' * 6} {'-' * 6} {'-' * 8}  {'-' * 6}",
        ]

        for mc in result.metric_comparisons:
            winner_label = (
                result.config_a_name if mc["winner"] == "A"
                else result.config_b_name if mc["winner"] == "B"
                else "tie"
            )
            lines.append(
                f"  {mc['metric']:30s} "
                f"{mc['avg_a']:6.2f} {mc['avg_b']:6.2f} "
                f"{mc['difference']:+8.2f}  {winner_label}"
            )

        lines.extend(["", "=" * 70])
        return "\n".join(lines)
