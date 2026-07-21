"""Tests for Harness observability artifact parsing."""

import json

from src.harness.observability import load_run_file, summarize_rows


def test_load_single_run_artifact(tmp_path):
    artifact = tmp_path / "single.json"
    artifact.write_text(
        json.dumps({
            "run_metadata": {"generated_at": "2026-07-21T12:00:00", "run_type": "single"},
            "results": [{
                "test_case_id": "case_1",
                "harness_target": "sample_essay",
                "input_source": "manual_cli",
                "user_input": "Question?",
                "scores": {"overall": 7.0, "task_response": 7.0},
                "latency_seconds": 12.5,
                "token_usage": {
                    "generation": {"total_tokens": 100},
                    "judge": {"total_tokens": 50},
                },
                "model_versions": {
                    "generation": "deepseek-chat",
                    "judge": "deepseek-chat",
                },
                "error_labels": ["generic_or_underdeveloped"],
                "error": None,
            }],
        }),
        encoding="utf-8",
    )

    rows = load_run_file(artifact)

    assert len(rows) == 1
    assert rows[0]["run_time"] == "2026-07-21T12:00:00"
    assert rows[0]["provider"] == "deepseek"
    assert rows[0]["fixed_judge"] == "deepseek"
    assert rows[0]["overall"] == 7.0


def test_load_comparison_artifact(tmp_path):
    artifact = tmp_path / "compare.json"
    artifact.write_text(
        json.dumps({
            "run_metadata": {"generated_at": "2026-07-21T12:00:00", "run_type": "comparison"},
            "comparison": {
                "provider_a": "deepseek",
                "provider_b": "openai",
                "judge_provider": "deepseek",
                "case_comparisons": [{
                    "test_case_id": "case_1",
                    "winner": "openai",
                }],
            },
            "provider_a_results": [{
                "test_case_id": "case_1",
                "user_input": "Question?",
                "scores": {"overall": 7.0},
                "token_usage": {"generation": {"total_tokens": 100}, "judge": {"total_tokens": 40}},
                "model_versions": {"generation": "deepseek-chat", "judge": "deepseek-chat"},
                "error_labels": [],
                "error": None,
            }],
            "provider_b_results": [{
                "test_case_id": "case_1",
                "user_input": "Question?",
                "scores": {"overall": 7.5},
                "token_usage": {"generation": {"total_tokens": 200}, "judge": {"total_tokens": 60}},
                "model_versions": {"generation": "gpt-5.5", "judge": "deepseek-chat"},
                "error_labels": ["no_major_error"],
                "error": None,
            }],
        }),
        encoding="utf-8",
    )

    rows = load_run_file(artifact)

    assert len(rows) == 2
    assert {row["provider"] for row in rows} == {"deepseek", "openai"}
    assert rows[0]["case_winner"] == "openai"
    assert rows[1]["fixed_judge"] == "deepseek"


def test_summarize_rows_counts_labels_and_tokens():
    rows = [
        {
            "run_id": "run_1",
            "overall": 7.0,
            "latency_seconds": 10.0,
            "generation_total_tokens": 100,
            "judge_total_tokens": 40,
            "error_labels": ["generic_or_underdeveloped"],
        },
        {
            "run_id": "run_1",
            "overall": 7.5,
            "latency_seconds": 20.0,
            "generation_total_tokens": 200,
            "judge_total_tokens": 60,
            "error_labels": ["generic_or_underdeveloped", "weak_task_response"],
        },
    ]

    summary = summarize_rows(rows)

    assert summary["runs"] == 1
    assert summary["cases"] == 2
    assert summary["avg_overall"] == 7.25
    assert summary["generation_tokens"] == 300
    assert summary["judge_tokens"] == 100
    assert summary["top_error_labels"]["generic_or_underdeveloped"] == 2
