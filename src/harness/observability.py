"""Utilities for loading Harness run artifacts into dashboard rows."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


SCORE_METRICS = [
    "task_response",
    "coherence_cohesion",
    "lexical_resource",
    "grammatical_range",
    "specificity_score",
    "band_alignment",
    "overall",
]


def discover_run_files(runs_dir: str | Path = "runs") -> List[Path]:
    """Return JSON Harness artifacts in newest-first order."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def load_run_file(path: str | Path) -> List[Dict]:
    """Load one Harness JSON artifact and flatten it into case-level rows."""
    artifact_path = Path(path)
    with open(artifact_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    run_time = _run_time(payload, artifact_path)
    run_id = artifact_path.stem

    if "comparison" in payload:
        comparison = payload.get("comparison", {})
        rows = []
        rows.extend(
            _flatten_results(
                payload.get("provider_a_results", []),
                run_id=run_id,
                run_time=run_time,
                file_path=artifact_path,
                run_type="comparison",
                provider_role="provider_a",
                provider_name=comparison.get("provider_a", ""),
                fixed_judge=comparison.get("judge_provider", ""),
                comparison=comparison,
            )
        )
        rows.extend(
            _flatten_results(
                payload.get("provider_b_results", []),
                run_id=run_id,
                run_time=run_time,
                file_path=artifact_path,
                run_type="comparison",
                provider_role="provider_b",
                provider_name=comparison.get("provider_b", ""),
                fixed_judge=comparison.get("judge_provider", ""),
                comparison=comparison,
            )
        )
        return rows

    return _flatten_results(
        payload.get("results", []),
        run_id=run_id,
        run_time=run_time,
        file_path=artifact_path,
        run_type=payload.get("run_metadata", {}).get("run_type", "single"),
        provider_role="provider",
        provider_name="",
        fixed_judge="",
        comparison={},
    )


def load_runs(runs_dir: str | Path = "runs") -> List[Dict]:
    """Load every Harness artifact under runs_dir."""
    rows: List[Dict] = []
    for path in discover_run_files(runs_dir):
        try:
            rows.extend(load_run_file(path))
        except Exception as exc:
            rows.append({
                "run_id": path.stem,
                "run_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "file_name": path.name,
                "run_type": "load_error",
                "error": str(exc),
            })
    return rows


def summarize_rows(rows: Iterable[Dict]) -> Dict:
    """Create high-level counters and averages for dashboard cards."""
    rows = [row for row in rows if not row.get("error")]
    if not rows:
        return {
            "runs": 0,
            "cases": 0,
            "avg_overall": 0.0,
            "avg_latency": 0.0,
            "generation_tokens": 0,
            "judge_tokens": 0,
            "top_error_labels": {},
        }

    run_count = len({row.get("run_id") for row in rows})
    avg_overall = _average(row.get("overall", 0.0) for row in rows)
    avg_latency = _average(row.get("latency_seconds", 0.0) for row in rows)
    generation_tokens = sum(int(row.get("generation_total_tokens", 0) or 0) for row in rows)
    judge_tokens = sum(int(row.get("judge_total_tokens", 0) or 0) for row in rows)

    label_counter: Counter[str] = Counter()
    for row in rows:
        for label in row.get("error_labels", []):
            label_counter[label] += 1

    return {
        "runs": run_count,
        "cases": len(rows),
        "avg_overall": round(avg_overall, 2),
        "avg_latency": round(avg_latency, 2),
        "generation_tokens": generation_tokens,
        "judge_tokens": judge_tokens,
        "top_error_labels": dict(label_counter.most_common(8)),
    }


def _flatten_results(
    results: List[Dict],
    run_id: str,
    run_time: str,
    file_path: Path,
    run_type: str,
    provider_role: str,
    provider_name: str,
    fixed_judge: str,
    comparison: Dict,
) -> List[Dict]:
    rows = []
    case_winners = {
        item.get("test_case_id"): item
        for item in comparison.get("case_comparisons", [])
    }
    for result in results:
        scores = result.get("scores", {})
        token_usage = result.get("token_usage", {})
        model_versions = result.get("model_versions", {})
        case_id = result.get("test_case_id", "")
        winner_info = case_winners.get(case_id, {})
        generation_model = model_versions.get("generation", "")
        judge_model = model_versions.get("judge", "")
        row = {
            "run_id": run_id,
            "run_time": run_time,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "run_type": run_type,
            "provider_role": provider_role,
            "provider": provider_name or _provider_from_model(generation_model),
            "fixed_judge": fixed_judge or _provider_from_model(judge_model),
            "generation_model": generation_model,
            "judge_model": judge_model,
            "harness_target": result.get("harness_target", ""),
            "input_source": result.get("input_source", ""),
            "test_case_id": case_id,
            "question": result.get("user_input", result.get("question", "")),
            "latency_seconds": float(result.get("latency_seconds", 0.0) or 0.0),
            "generation_total_tokens": int(token_usage.get("generation", {}).get("total_tokens", 0) or 0),
            "judge_total_tokens": int(token_usage.get("judge", {}).get("total_tokens", 0) or 0),
            "error_labels": result.get("error_labels", []),
            "case_winner": winner_info.get("winner", ""),
            "error": result.get("error"),
        }
        for metric in SCORE_METRICS:
            row[metric] = float(scores.get(metric, 0.0) or 0.0)
        rows.append(row)
    return rows


def _run_time(payload: Dict, path: Path) -> str:
    generated_at = payload.get("run_metadata", {}).get("generated_at")
    if generated_at:
        return generated_at
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _provider_from_model(model_name: str) -> str:
    name = (model_name or "").lower()
    if "deepseek" in name:
        return "deepseek"
    if name.startswith("gpt") or name.startswith("o"):
        return "openai"
    return model_name or "unknown"


def _average(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
