"""Fixed-judge A/B comparison CLI for harness evaluations."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .evaluator import Evaluator, _build_generation_pipeline
from .metrics import MetricsCalculator
from .reporter import Reporter
from .test_cases import TestCaseManager


COMPARISON_METRICS = [
    *MetricsCalculator.IELTS_CRITERIA,
    "specificity_score",
    "band_alignment",
    "overall",
]


def build_cases(
    question: str = "",
    test_count: int = 3,
    topic: str = "",
    seed: int = 42,
) -> List[Dict]:
    """Build a stable test set from manual input or the local question bank."""
    if question:
        return [{
            "id": "manual_001",
            "question": question,
            "source": "manual_cli",
        }]

    random.seed(seed)
    test_cases = TestCaseManager()
    cases = test_cases.sample(test_count, topic or None)
    for case in cases:
        case.setdefault("source", "local_test_case_bank")
    return cases


def compare_results(
    provider_a: str,
    results_a: List[Dict],
    provider_b: str,
    results_b: List[Dict],
    judge_provider: str,
) -> Dict:
    """Compare two result sets produced on the same cases by a fixed judge."""
    by_id_b = {result.get("test_case_id"): result for result in results_b}
    case_comparisons = []

    for result_a in results_a:
        case_id = result_a.get("test_case_id")
        result_b = by_id_b.get(case_id)
        if not result_b:
            continue

        overall_a = result_a.get("scores", {}).get("overall", 0.0)
        overall_b = result_b.get("scores", {}).get("overall", 0.0)
        diff = round(overall_a - overall_b, 2)
        winner = provider_a if diff > 0.1 else provider_b if diff < -0.1 else "tie"

        case_comparisons.append({
            "test_case_id": case_id,
            "question": result_a.get("user_input", result_a.get("question", "")),
            "provider_a_overall": overall_a,
            "provider_b_overall": overall_b,
            "difference": diff,
            "winner": winner,
            "provider_a_error_labels": result_a.get("error_labels", []),
            "provider_b_error_labels": result_b.get("error_labels", []),
        })

    metric_comparisons = []
    for metric in COMPARISON_METRICS:
        scores_a = [
            result.get("scores", {}).get(metric)
            for result in results_a
            if not result.get("error") and metric in result.get("scores", {})
        ]
        scores_b = [
            result.get("scores", {}).get(metric)
            for result in results_b
            if not result.get("error") and metric in result.get("scores", {})
        ]
        if not scores_a or not scores_b:
            continue

        avg_a = sum(scores_a) / len(scores_a)
        avg_b = sum(scores_b) / len(scores_b)
        diff = round(avg_a - avg_b, 2)
        winner = provider_a if diff > 0.1 else provider_b if diff < -0.1 else "tie"
        metric_comparisons.append({
            "metric": metric,
            "provider_a_avg": round(avg_a, 2),
            "provider_b_avg": round(avg_b, 2),
            "difference": diff,
            "winner": winner,
        })

    winner_counts = {
        provider_a: sum(1 for item in case_comparisons if item["winner"] == provider_a),
        provider_b: sum(1 for item in case_comparisons if item["winner"] == provider_b),
        "tie": sum(1 for item in case_comparisons if item["winner"] == "tie"),
    }
    if winner_counts[provider_a] > winner_counts[provider_b]:
        overall_winner = provider_a
    elif winner_counts[provider_b] > winner_counts[provider_a]:
        overall_winner = provider_b
    else:
        overall_winner = "tie"

    return {
        "provider_a": provider_a,
        "provider_b": provider_b,
        "judge_provider": judge_provider,
        "test_case_count": len(case_comparisons),
        "overall_winner": overall_winner,
        "winner_counts": winner_counts,
        "metric_comparisons": metric_comparisons,
        "case_comparisons": case_comparisons,
    }


def format_comparison_report(comparison: Dict) -> str:
    """Format a fixed-judge A/B comparison for terminal display."""
    provider_a = comparison["provider_a"]
    provider_b = comparison["provider_b"]
    lines = [
        "=" * 74,
        "  IELTS Writing Task 2 - Fixed-Judge Provider Comparison",
        "=" * 74,
        "",
        f"  Provider A: {provider_a}",
        f"  Provider B: {provider_b}",
        f"  Fixed judge: {comparison['judge_provider']}",
        f"  Test cases: {comparison['test_case_count']}",
        f"  Overall winner: {comparison['overall_winner']}",
        (
            "  Case wins: "
            f"{provider_a}={comparison['winner_counts'].get(provider_a, 0)}, "
            f"{provider_b}={comparison['winner_counts'].get(provider_b, 0)}, "
            f"tie={comparison['winner_counts'].get('tie', 0)}"
        ),
        "",
        "  -- Metric Comparison --",
        "",
        f"  {'Metric':28s} {provider_a:>10s} {provider_b:>10s} {'Diff':>8s}  Winner",
        f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 8}  {'-' * 10}",
    ]

    for item in comparison["metric_comparisons"]:
        lines.append(
            f"  {item['metric']:28s} "
            f"{item['provider_a_avg']:10.2f} "
            f"{item['provider_b_avg']:10.2f} "
            f"{item['difference']:+8.2f}  "
            f"{item['winner']}"
        )

    lines.extend(["", "  -- Case Comparison --", ""])
    for item in comparison["case_comparisons"]:
        question = item["question"].replace("\n", " ")
        preview = question[:86] + ("..." if len(question) > 86 else "")
        lines.append(
            f"  {item['test_case_id']:18s} "
            f"{provider_a}: {item['provider_a_overall']:.1f}  "
            f"{provider_b}: {item['provider_b_overall']:.1f}  "
            f"winner: {item['winner']}"
        )
        lines.append(f"    Input: {preview}")

    lines.extend(["", "=" * 74])
    return "\n".join(lines)


def export_comparison(
    output_path: str,
    comparison: Dict,
    results_a: List[Dict],
    results_b: List[Dict],
):
    """Export comparison plus both raw harness traces."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "run_type": "comparison",
        },
        "comparison": comparison,
        "provider_a_results": results_a,
        "provider_b_results": results_b,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two generation providers with one fixed judge."
    )
    parser.add_argument("--question", default="")
    parser.add_argument("--test-count", type=int, default=3)
    parser.add_argument("--topic", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provider-a", default="deepseek", choices=["openai", "deepseek"])
    parser.add_argument("--provider-b", default="openai", choices=["openai", "deepseek"])
    parser.add_argument("--judge-provider", default="openai", choices=["openai", "deepseek"])
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    from ..llm.factory import LLMFactory
    from ..utils.config import Config

    cases = build_cases(
        question=args.question,
        test_count=args.test_count,
        topic=args.topic,
        seed=args.seed,
    )

    config = Config()
    judge = LLMFactory.create(args.judge_provider, config)
    evaluator = Evaluator(judge)

    pipeline_a = _build_generation_pipeline(args.provider_a)
    pipeline_b = _build_generation_pipeline(args.provider_b)

    results_a = evaluator.batch_evaluate(cases, pipeline_a)
    results_b = evaluator.batch_evaluate(cases, pipeline_b)

    comparison = compare_results(
        args.provider_a,
        results_a,
        args.provider_b,
        results_b,
        args.judge_provider,
    )

    print(Reporter.terminal_summary(results_a))
    print()
    print(Reporter.terminal_summary(results_b))
    print()
    print(format_comparison_report(comparison))

    if args.json_output:
        export_comparison(args.json_output, comparison, results_a, results_b)


if __name__ == "__main__":
    main()
