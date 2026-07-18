"""Core evaluation engine for single and batch quality checks."""

import logging
import time
from typing import Dict, List, Optional

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

        # Calculate specificity score (automated)
        scores["specificity_score"] = self._calculate_specificity(output)

        # Calculate band alignment (automated)
        scores["band_alignment"] = self._calculate_band_alignment(scores)

        # Overall score
        overall = MetricsCalculator.calculate_overall(scores)

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
                scores = self.evaluate(case["question"], output)
                latency = time.time() - start_time

                results.append({
                    "test_case_id": case.get("id", f"case_{i}"),
                    "question": case["question"],
                    "output": output[:500],
                    "scores": scores.to_dict(),
                    "latency_seconds": round(latency, 2),
                    "error": None,
                })

            except Exception as e:
                logger.error(f"Failed on test case {case.get('id', 'unknown')}: {e}")
                results.append({
                    "test_case_id": case.get("id", f"case_{i}"),
                    "question": case.get("question", ""),
                    "output": "",
                    "scores": MetricScores().to_dict(),
                    "latency_seconds": 0,
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

        Band alignment measures how close all scores are to Band 7,
        with penalties for being significantly below.

        Args:
            scores: Dict of metric_name -> score.

        Returns:
            Band alignment score (1.0-9.0).
        """
        if not scores:
            return 0.0

        target = 7.0
        deviations = [abs(s - target) for s in scores.values()]
        avg_deviation = sum(deviations) / len(deviations)

        # Lower deviation = higher alignment
        alignment = target - avg_deviation
        return max(1.0, min(9.0, alignment + 2))  # +2 to bring it into reasonable range


def _build_pipeline(provider: str = None, judge_provider: str = None):
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
    judge_provider_name = judge_provider or config.get("harness.judge_model", "openai")
    llm = LLMFactory.create(provider_name, config)
    judge_llm = LLMFactory.create(judge_provider_name, config)

    retriever = None
    try:
        embedder = Embedder(config.get("rag.embedding_model", "all-MiniLM-L6-v2"))
        store = VectorStore(config.chroma_persist_dir)
        retriever = Retriever(embedder, store)
    except Exception as exc:
        logger.warning("Harness RAG initialization skipped: %s", exc)

    workflow = AgentWorkflow(llm=llm, retriever=retriever, prompt_manager=PromptManager())

    def pipeline(question: str) -> str:
        state = WorkflowState(user_input=question, scenario=Scenario.GENERATE)
        state = workflow.run(state)
        parts = []
        if state.analysis:
            parts.append(f"Question Type: {state.analysis.question_type_en}")
            parts.append(f"Core Issue: {state.analysis.controversy}")
        if state.arguments:
            parts.append(state.arguments[0].main_idea_en)
        if state.outline and state.outline.tips:
            parts.append("Suggested Outline:\n" + state.outline.tips[0])
        return "\n\n".join(parts)

    return judge_llm, pipeline


def main():
    """CLI for internal quality checks."""
    import argparse

    from .reporter import Reporter
    from .test_cases import TestCaseManager

    parser = argparse.ArgumentParser(description="Run IELTS agent quality harness.")
    parser.add_argument("--test-count", type=int, default=3)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--provider", default=None, choices=["openai", "deepseek"])
    parser.add_argument("--judge-provider", default=None, choices=["openai", "deepseek"])
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()

    test_cases = TestCaseManager()
    cases = test_cases.sample(args.test_count, args.topic)

    if args.rules_only:
        for case in cases:
            print(f"{case.get('id')}: rules-only requires a generated output; use full mode for pipeline output.")
        return

    judge, pipeline = _build_pipeline(args.provider, args.judge_provider)
    evaluator = Evaluator(judge)
    results = evaluator.batch_evaluate(cases, pipeline)
    print(Reporter.terminal_summary(results))


if __name__ == "__main__":
    main()
