"""Core evaluation engine for single and batch quality checks."""

import logging
import time
from typing import Callable, Dict, List, Optional

from ..llm.base import LLMProvider
from .metrics import MetricsCalculator, MetricScores

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates agent pipeline outputs against IELTS standards.

    Uses LLM-as-judge for subjective metrics and structural checks
    for automated metrics.
    """

    def __init__(self, judge_llm: LLMProvider):
        """Initialize the evaluator.

        Args:
            judge_llm: LLM provider used as the judge (can be different
                       from the model being evaluated).
        """
        self.judge = judge_llm
        self.metrics_calc = MetricsCalculator()

    def evaluate(
        self,
        question: str,
        output: str,
        metrics: Optional[List[str]] = None,
    ) -> MetricScores:
        """Evaluate a single output against IELTS standards.

        Args:
            question: The original IELTS question.
            output: The agent's output text.
            metrics: Specific metrics to evaluate. Defaults to all.

        Returns:
            MetricScores with all evaluated dimensions.
        """
        if metrics is None:
            metrics = [
                "task_response",
                "coherence_cohesion",
                "lexical_resource",
                "grammatical_range",
            ]

        scores = {}
        comments = {}

        for metric in metrics:
            try:
                prompt = MetricsCalculator.get_judge_prompt(
                    question, output, metric
                )
                response = self.judge.generate(
                    prompt,
                    temperature=0.2,  # Low temperature for consistent judging
                    max_tokens=300,
                )

                score = self._extract_score(response)
                scores[metric] = score
                comments[metric] = response[:200]

                logger.info(f"Metric '{metric}': Band {score}")

            except Exception as e:
                logger.error(f"Failed to evaluate metric '{metric}': {e}")
                scores[metric] = 0.0
                comments[metric] = f"Evaluation error: {e}"

        official_scores = {
            metric: scores[metric]
            for metric in MetricsCalculator.IELTS_CRITERIA
            if metric in scores
        }

        # Calculate specificity score (automated diagnostic)
        scores["specificity_score"] = self._calculate_specificity(output)

        # Calculate band alignment (diagnostic; not part of overall)
        scores["band_alignment"] = self._calculate_band_alignment(official_scores)

        # Overall score follows the four official IELTS criteria only.
        overall = MetricsCalculator.calculate_overall(official_scores)

        return MetricScores(
            task_response=scores.get("task_response", 0.0),
            coherence_cohesion=scores.get("coherence_cohesion", 0.0),
            lexical_resource=scores.get("lexical_resource", 0.0),
            grammatical_range=scores.get("grammatical_range", 0.0),
            specificity_score=scores.get("specificity_score", 0.0),
            band_alignment=scores.get("band_alignment", 0.0),
            overall=overall,
            comments=comments,
        )

    @staticmethod
    def rule_evaluate(question: str, output: str) -> Dict[str, object]:
        """Run cheap deterministic checks before any LLM-as-judge call.

        This is the first quality layer for the product: it checks whether an
        answer has the pieces learners need before judging style or depth.
        """
        text = output.lower()
        question_lower = question.lower()
        topic_terms = [
            word.strip(".,;:!?()[]")
            for word in question_lower.split()
            if len(word.strip(".,;:!?()[]")) >= 5
        ]

        checks = {
            "has_question_type": any(term in text for term in ["question type", "opinion", "discussion"]),
            "has_position": any(term in text for term in ["position", "stance", "agree", "disagree"]),
            "has_multiple_ideas": (
                text.count("main idea") >= 2
                or text.count("idea") >= 2
            ),
            "has_example": any(term in text for term in ["example", "for instance", "for example"]),
            "has_outline": any(term in text for term in ["outline", "introduction", "body", "conclusion"]),
            "uses_question_terms": any(term in text for term in topic_terms[:8]),
            "not_too_complex": len(output.split()) <= 1200,
        }
        passed = sum(1 for value in checks.values() if value)
        return {
            "passed": passed,
            "total": len(checks),
            "score": round(passed / len(checks) * 9, 1),
            "checks": checks,
        }

    def batch_evaluate(
        self,
        test_cases: List[Dict],
        output_getter,
    ) -> List[Dict]:
        """Evaluate multiple test cases.

        Args:
            test_cases: List of test case dicts with 'id' and 'question'.
            output_getter: Callable that takes a question and returns output.

        Returns:
            List of evaluation result dicts.
        """
        results = []

        for i, case in enumerate(test_cases):
            logger.info(
                f"Evaluating case {i + 1}/{len(test_cases)}: {case.get('id', 'unknown')}"
            )

            start_time = time.time()

            try:
                output = output_getter(case["question"])
                generation_trace = getattr(output_getter, "last_trace", {})
                if hasattr(self.judge, "reset_usage"):
                    self.judge.reset_usage()
                scores = self.evaluate(case["question"], output)
                latency = time.time() - start_time
                score_dict = scores.to_dict()
                judge_usage = getattr(self.judge, "usage_total", {})

                results.append({
                    "test_case_id": case.get("id", f"case_{i}"),
                    "harness_target": generation_trace.get("harness_target", "sample_essay"),
                    "input_source": case.get("source", "local_test_case_bank"),
                    "user_input": case["question"],
                    "question": case["question"],
                    "retrieval_query": generation_trace.get("retrieval_query", ""),
                    "recalled_chunks": generation_trace.get("recalled_chunks", []),
                    "final_prompt": generation_trace.get("final_prompt", ""),
                    "prompt_calls": generation_trace.get("prompt_calls", []),
                    "output": output[:500],
                    "model_output": output,
                    "scores": score_dict,
                    "scoring_result": score_dict,
                    "error_labels": self._label_errors(score_dict, generation_trace),
                    "latency_seconds": round(latency, 2),
                    "token_usage": {
                        "generation": generation_trace.get("token_usage", {}),
                        "judge": judge_usage,
                    },
                    "model_versions": {
                        "generation": generation_trace.get("model_name", ""),
                        "judge": getattr(self.judge, "model_name", ""),
                    },
                    "error": None,
                })

            except Exception as e:
                logger.error(f"Failed on test case {case.get('id', 'unknown')}: {e}")
                results.append({
                    "test_case_id": case.get("id", f"case_{i}"),
                    "harness_target": "sample_essay",
                    "input_source": case.get("source", "local_test_case_bank"),
                    "user_input": case.get("question", ""),
                    "question": case.get("question", ""),
                    "retrieval_query": "",
                    "recalled_chunks": [],
                    "final_prompt": "",
                    "prompt_calls": [],
                    "output": "",
                    "model_output": "",
                    "scores": MetricScores().to_dict(),
                    "scoring_result": MetricScores().to_dict(),
                    "error_labels": ["runtime_error"],
                    "latency_seconds": 0,
                    "token_usage": {},
                    "model_versions": {},
                    "error": str(e),
                })

        return results

    @staticmethod
    def batch_rule_evaluate(
        test_cases: List[Dict],
        output_getter: Callable[[str], str],
    ) -> List[Dict]:
        """Generate outputs and run deterministic checks only.

        This mode is useful for quick smoke/regression tests because it does
        not call an LLM judge. It still calls the selected generation provider.
        """
        results = []

        for i, case in enumerate(test_cases):
            logger.info(
                f"Rule-evaluating case {i + 1}/{len(test_cases)}: {case.get('id', 'unknown')}"
            )
            start_time = time.time()

            try:
                output = output_getter(case["question"])
                generation_trace = getattr(output_getter, "last_trace", {})
                rule_result = Evaluator.rule_evaluate(case["question"], output)
                latency = time.time() - start_time
                score_dict = {
                    "specificity_score": Evaluator._calculate_specificity(output),
                    "overall": rule_result["score"],
                    "rule_score": rule_result["score"],
                    "rule_passed": rule_result["passed"],
                    "rule_total": rule_result["total"],
                    "rule_checks": rule_result["checks"],
                    "comments": {},
                }

                results.append({
                    "test_case_id": case.get("id", f"case_{i}"),
                    "harness_target": generation_trace.get("harness_target", "sample_essay"),
                    "input_source": case.get("source", "local_test_case_bank"),
                    "user_input": case["question"],
                    "question": case["question"],
                    "retrieval_query": generation_trace.get("retrieval_query", ""),
                    "recalled_chunks": generation_trace.get("recalled_chunks", []),
                    "final_prompt": generation_trace.get("final_prompt", ""),
                    "prompt_calls": generation_trace.get("prompt_calls", []),
                    "output": output[:500],
                    "model_output": output,
                    "scores": score_dict,
                    "scoring_result": score_dict,
                    "error_labels": Evaluator._label_rule_errors(rule_result),
                    "latency_seconds": round(latency, 2),
                    "token_usage": {
                        "generation": generation_trace.get("token_usage", {}),
                        "judge": {},
                    },
                    "model_versions": {
                        "generation": generation_trace.get("model_name", ""),
                        "judge": "",
                    },
                    "error": None,
                })

            except Exception as e:
                logger.error(f"Failed on test case {case.get('id', 'unknown')}: {e}")
                results.append({
                    "test_case_id": case.get("id", f"case_{i}"),
                    "harness_target": "sample_essay",
                    "input_source": case.get("source", "local_test_case_bank"),
                    "user_input": case.get("question", ""),
                    "question": case.get("question", ""),
                    "retrieval_query": "",
                    "recalled_chunks": [],
                    "final_prompt": "",
                    "prompt_calls": [],
                    "output": "",
                    "model_output": "",
                    "scores": MetricScores().to_dict(),
                    "scoring_result": MetricScores().to_dict(),
                    "error_labels": ["runtime_error"],
                    "latency_seconds": 0,
                    "token_usage": {},
                    "model_versions": {},
                    "error": str(e),
                })

        return results

    @staticmethod
    def _extract_score(response: str) -> float:
        """Extract band score from LLM judge response.

        Args:
            response: The judge's response text.

        Returns:
            Extracted score as float (1.0-9.0).
        """
        for line in response.split("\n"):
            line = line.strip()
            if line.lower().startswith("score:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    # Handle formats like "7.0", "Band 7", "7/9"
                    score_text = score_text.replace("Band", "").replace("/9", "").strip()
                    score = float(score_text)
                    return max(1.0, min(9.0, score))
                except (ValueError, IndexError):
                    pass

        # Fallback: search for any number that looks like a band score
        import re
        numbers = re.findall(r"\b([1-9](?:\.[05])?)\b", response)
        for num in numbers:
            score = float(num)
            if 1.0 <= score <= 9.0:
                return score

        return 5.0  # Default to mid-range if extraction fails

    @staticmethod
    def _calculate_specificity(output: str) -> float:
        """Calculate how specific (vs generic) the output is.

        Rewards: concrete examples, specific numbers, named entities.
        Penalizes: vague language, templates without specifics.

        Args:
            output: The agent's output text.

        Returns:
            Specificity score (1.0-9.0).
        """
        output_lower = output.lower()

        # Indicators of specificity
        specific_indicators = [
            "for instance", "for example", "such as", "according to",
            "research shows", "studies have", "in 20", "percent",
            "million", "billion", "specifically", "notably",
        ]
        # Indicators of vagueness
        vague_indicators = [
            "some people", "many things", "in general", "generally speaking",
            "it is said that", "some say", "many believe",
        ]

        specific_count = sum(1 for ind in specific_indicators if ind in output_lower)
        vague_count = sum(1 for ind in vague_indicators if ind in output_lower)

        # Base score
        specificity = 5.0
        specificity += specific_count * 0.5
        specificity -= vague_count * 0.5

        return max(1.0, min(9.0, specificity))

    @staticmethod
    def _calculate_band_alignment(scores: Dict[str, float]) -> float:
        """Calculate how well the output aligns with Band 7 standards.

        Band alignment is a diagnostic proxy for readiness against the target
        band. It is based only on official IELTS criterion scores.

        Args:
            scores: Dict of metric_name -> score.

        Returns:
            Band alignment score (1.0-9.0).
        """
        if not scores:
            return 0.0

        avg_score = sum(scores.values()) / len(scores)
        return max(1.0, min(9.0, round(avg_score * 2) / 2))

    @staticmethod
    def _label_errors(scores: Dict[str, object], trace: Dict[str, object]) -> List[str]:
        """Attach coarse failure labels for product-quality diagnosis."""
        labels = []
        if scores.get("task_response", 0) < 6.0:
            labels.append("weak_task_response")
        if scores.get("coherence_cohesion", 0) < 6.0:
            labels.append("weak_coherence")
        if scores.get("lexical_resource", 0) < 6.0:
            labels.append("weak_lexical_resource")
        if scores.get("grammatical_range", 0) < 6.0:
            labels.append("weak_grammar_range")
        if scores.get("specificity_score", 0) < 6.0:
            labels.append("generic_or_underdeveloped")
        if not trace.get("recalled_chunks"):
            labels.append("no_rag_chunks")
        return labels or ["no_major_error"]

    @staticmethod
    def _label_rule_errors(rule_result: Dict[str, object]) -> List[str]:
        """Translate failed deterministic checks into readable labels."""
        checks = rule_result.get("checks", {})
        labels = [
            f"missing_{name[4:]}" if name.startswith("has_") else f"failed_{name}"
            for name, passed in checks.items()
            if not passed
        ]
        return labels or ["no_major_error"]


def _build_generation_pipeline(provider: str = None):
    from ..agent.state import Scenario, WorkflowState
    from ..agent.workflow import AgentWorkflow
    from ..llm.factory import LLMFactory
    from ..prompts.templates import PromptManager
    from ..rag.embedder import Embedder
    from ..rag.retriever import Retriever
    from ..rag.store import VectorStore
    from ..utils.config import Config

    config = Config()
    provider_name = provider or config.default_provider
    llm = LLMFactory.create(provider_name, config)

    retriever = None
    try:
        embedder = Embedder(config.get("rag.embedding_model", "all-MiniLM-L6-v2"))
        store = VectorStore(config.chroma_persist_dir)
        retriever = Retriever(embedder, store)
    except Exception as exc:
        logger.warning("Harness RAG initialization skipped: %s", exc)

    workflow = AgentWorkflow(llm=llm, retriever=retriever, prompt_manager=PromptManager())

    def pipeline(question: str) -> str:
        if hasattr(llm, "reset_usage"):
            llm.reset_usage()
        state = WorkflowState(user_input=question, scenario=Scenario.GENERATE)
        state = workflow.run(state)
        raw_sample_essay = ""
        output = ""
        if not state.has_errors():
            raw_sample_essay = workflow.generate_sample_essay(state)
            output = _extract_essay_body(raw_sample_essay)
        pipeline.last_trace = {
            "user_input": state.user_input,
            "harness_target": "sample_essay",
            "retrieval_query": state.trace.get("retrieval_query", ""),
            "recalled_chunks": _summarize_recalled_chunks(state.rag_raw_results),
            "final_prompt": state.trace.get("final_prompt", ""),
            "prompt_calls": state.trace.get("prompt_calls", []),
            "raw_sample_essay": raw_sample_essay,
            "model_name": llm.model_name,
            "token_usage": getattr(llm, "usage_total", {}),
            "workflow_errors": state.errors,
        }
        return output

    pipeline.last_trace = {}

    return pipeline


def _extract_essay_body(sample_essay_output: str) -> str:
    """Keep only the English essay body before study notes."""
    if "---" in sample_essay_output:
        return sample_essay_output.split("---", 1)[0].strip()
    return sample_essay_output.strip()


def _summarize_recalled_chunks(raw_results: Dict[str, List[Dict]], max_chars: int = 260) -> List[Dict]:
    chunks = []
    for collection_name, items in raw_results.items():
        for index, item in enumerate(items):
            content = item.get("content", "")
            chunks.append({
                "collection": collection_name,
                "rank": index + 1,
                "distance": item.get("distance", 0.0),
                "metadata": item.get("metadata", {}),
                "content_preview": content[:max_chars],
            })
    return chunks


def _build_pipeline(provider: str = None, judge_provider: str = None):
    from ..llm.factory import LLMFactory
    from ..utils.config import Config

    config = Config()
    judge_provider_name = judge_provider or config.get("harness.judge_model", "openai")
    judge_llm = LLMFactory.create(judge_provider_name, config)
    pipeline = _build_generation_pipeline(provider)
    return judge_llm, pipeline


def main():
    """CLI for internal quality checks."""
    import argparse
    import random

    from .reporter import Reporter
    from .test_cases import TestCaseManager

    parser = argparse.ArgumentParser(description="Run IELTS agent quality harness.")
    parser.add_argument("--question", default=None, help="Evaluate one explicit IELTS Task 2 question.")
    parser.add_argument("--test-count", type=int, default=3)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for local test case sampling.")
    parser.add_argument("--provider", default=None, choices=["openai", "deepseek"])
    parser.add_argument("--judge-provider", default=None, choices=["openai", "deepseek"])
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--json-output", default=None, help="Optional path for full trace JSON export.")
    parser.add_argument("--csv-output", default=None, help="Optional path for summary CSV export.")
    args = parser.parse_args()

    if args.question:
        cases = [{
            "id": "manual_001",
            "question": args.question,
            "source": "manual_cli",
        }]
    else:
        random.seed(args.seed)
        test_cases = TestCaseManager()
        cases = test_cases.sample(args.test_count, args.topic)
        for case in cases:
            case.setdefault("source", "local_test_case_bank")

    if args.rules_only:
        pipeline = _build_generation_pipeline(args.provider)
        results = Evaluator.batch_rule_evaluate(cases, pipeline)
        print(Reporter.terminal_summary(results))
        if args.json_output:
            Reporter.to_json(results, args.json_output)
        if args.csv_output:
            Reporter.to_csv(results, args.csv_output)
        return

    judge, pipeline = _build_pipeline(args.provider, args.judge_provider)
    evaluator = Evaluator(judge)
    results = evaluator.batch_evaluate(cases, pipeline)
    print(Reporter.terminal_summary(results))
    if args.json_output:
        Reporter.to_json(results, args.json_output)
    if args.csv_output:
        Reporter.to_csv(results, args.csv_output)


if __name__ == "__main__":
    main()
