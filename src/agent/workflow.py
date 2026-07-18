"""Core workflow for the IELTS Task 2 learning agent."""

from __future__ import annotations

import logging
import re
from typing import Optional

from ..knowledge.structures import StructureManager
from ..knowledge.topics import QuestionTypeDetector, TopicMatcher
from ..knowledge.vocabulary import CollocationManager
from ..llm.base import LLMProvider
from ..prompts.templates import PromptManager
from ..rag.retriever import Retriever
from .state import (
    Argument,
    EssayOutline,
    EvaluationResult,
    QuestionAnalysis,
    Scenario,
    WorkflowState,
)

logger = logging.getLogger(__name__)


class AgentWorkflow:
    """Orchestrates the product learning loop."""

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Optional[Retriever] = None,
        prompt_manager: Optional[PromptManager] = None,
        collocation_manager: Optional[CollocationManager] = None,
        structure_manager: Optional[StructureManager] = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_manager or PromptManager()
        self.collocations = collocation_manager or CollocationManager()
        self.structures = structure_manager or StructureManager()

    def run(self, state: WorkflowState) -> WorkflowState:
        logger.info("Starting workflow: %s", state.scenario.value)

        try:
            state = self._step_1_analyze_question(state)
            state = self._step_2_retrieve_knowledge(state)

            if state.scenario == Scenario.DEEPEN:
                return self._step_3_deepen_ideas(state)
            if state.scenario == Scenario.EVALUATE:
                return self._step_3_evaluate_outline(state)

            state = self._step_3_generate_ideas(state)
            state = self._step_4_build_outline(state)
        except Exception as exc:
            logger.error("Workflow error: %s", exc, exc_info=True)
            state.add_error(f"Workflow failed: {exc}")

        return state

    def _step_1_analyze_question(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 1: analyzing question")
        question_type = QuestionTypeDetector.detect(state.user_input)
        topics = TopicMatcher.detect_topic(state.user_input)

        try:
            prompt = self.prompts.get("question_analyzer", state.prompt_version)
            rendered = prompt.render(question=state.user_input)
            response = self.llm.generate(
                rendered["user"],
                system_message=rendered["system"],
                temperature=0.3,
            )
            state.analysis = self._parse_question_analysis(response, question_type, topics)
        except Exception as exc:
            logger.warning("LLM question analysis failed, using local detection: %s", exc)
            state.analysis = QuestionAnalysis(
                question_type=question_type.get("key", "opinion"),
                question_type_zh=question_type.get("zh", "Opinion"),
                question_type_en=question_type.get("en", "Opinion Essay"),
                question_subtype=question_type.get("subtype", ""),
                question_subtype_zh=question_type.get("subtype_zh", ""),
                controversy="Local fallback: identify the prompt type, core issue, and likely topic before generating ideas.",
                stance_recommendations=[],
                keywords=[],
                topics=[topic.get("key", "") for topic in topics],
            )

        state.model_used = self.llm.model_name
        return state

    def _step_2_retrieve_knowledge(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 2: retrieving knowledge")
        topic = state.analysis.topics[0] if state.analysis and state.analysis.topics else None
        question_type = state.analysis.question_type if state.analysis else "opinion"

        if not self.retriever:
            vocab_context = self.collocations.format_for_prompt(topic, limit=15)
            structure_context = self.structures.format_for_prompt(question_type)
            state.rag_context = self._fallback_context(structure_context, vocab_context)
            state.rag_raw_results = {}
            return state

        try:
            query = state.user_input
            if state.analysis and state.analysis.controversy:
                query = f"{state.analysis.controversy}\n{state.user_input}"
            results = self.retriever.retrieve(
                query=query,
                topic=topic,
                question_type=question_type,
            )
            state.rag_raw_results = results
            state.rag_context = self.retriever.format_context(results)
        except Exception as exc:
            logger.warning("RAG retrieval failed, falling back to local references: %s", exc)
            vocab_context = self.collocations.format_for_prompt(topic, limit=15)
            structure_context = self.structures.format_for_prompt(question_type)
            state.rag_context = self._fallback_context(structure_context, vocab_context)

        return state

    def _step_3_generate_ideas(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 3: generating idea options")
        if not state.analysis:
            state.add_error("Cannot generate ideas without question analysis")
            return state

        try:
            prompt = self.prompts.get("idea_generator", state.prompt_version)
            rendered = prompt.render(
                question=state.user_input,
                question_type_zh=state.analysis.question_type_zh,
                question_type_en=state.analysis.question_type_en,
                question_subtype_zh=state.analysis.question_subtype_zh,
                controversy=state.analysis.controversy,
                topics=", ".join(state.analysis.topics),
                stance_sections=self._build_stance_sections(state),
                rag_context=state.rag_context,
            )
            response = self.llm.generate(
                rendered["user"],
                system_message=rendered["system"],
                temperature=0.7,
                max_tokens=4096,
            )
            state.arguments = self._parse_arguments(response)
        except Exception as exc:
            logger.error("Idea generation failed: %s", exc)
            state.add_error(f"Idea generation failed: {exc}")

        return state

    def _step_3_deepen_ideas(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 3: deepening user idea")
        user_idea = state.user_idea or state.user_input
        prompt = f"""You are an IELTS Task 2 argument coach for Chinese-speaking learners.

Original question:
{state.user_input}

Student idea:
{user_idea}

Relevant writing references:
{state.rag_context}

Expand the idea for a Band 6.5-7.5 learner. Do not write a full essay.
Use Simplified Chinese to explain the reasoning, then show how to express the same idea in IELTS English.
Return:
1. Chinese main idea: explain what the idea means and why it fits the question
2. Chinese development logic: Claim -> Reason -> Result/Impact -> Link back
3. English argument sentence: one clear topic sentence
4. English expression examples: 2-3 IELTS-style sentences, not a full paragraph
5. Usable example: concrete example with Chinese explanation
6. Useful vocabulary: English phrase + Chinese meaning
"""
        try:
            response = self.llm.generate(prompt, temperature=0.5, max_tokens=1500)
            state.arguments = self._parse_arguments(response)
        except Exception as exc:
            logger.error("Idea deepening failed: %s", exc)
            state.add_error(f"Idea deepening failed: {exc}")
        return state

    def _step_3_evaluate_outline(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 3: evaluating outline")
        outline_text = state.user_outline or state.user_input
        prompt = f"""You are an IELTS Writing Task 2 teacher for Chinese-speaking learners.

Question:
{state.user_input}

Student outline:
{outline_text}

Reference standards:
{state.rag_context}

Evaluate the outline as an idea-and-structure plan, not as a finished essay.
Score each criterion from 1.0 to 9.0.
Use Simplified Chinese for strengths, risks, and suggestions. When suggesting improvements, include English topic sentence or phrase examples where useful.

Return exactly this format:
Task Response: X.X
Coherence and Cohesion: X.X
Lexical Resource: X.X
Grammar Range and Accuracy: X.X
Overall: X.X
Strengths:
- ...
Risks:
- ...
Suggestions:
- ...
"""
        try:
            response = self.llm.generate(prompt, temperature=0.3, max_tokens=1200)
            state.evaluation = self._parse_evaluation(response)
        except Exception as exc:
            logger.error("Outline evaluation failed: %s", exc)
            state.add_error(f"Outline evaluation failed: {exc}")
        return state

    def _step_4_build_outline(self, state: WorkflowState) -> WorkflowState:
        logger.info("Step 4: building outline")
        if not state.arguments:
            state.add_error("Cannot build outline without generated ideas")
            return state

        try:
            prompt = self.prompts.get("outline_builder", state.prompt_version)
            rendered = prompt.render(
                question=state.user_input,
                question_type_zh=state.analysis.question_type_zh if state.analysis else "",
                question_type_en=state.analysis.question_type_en if state.analysis else "",
                question_subtype_zh=state.analysis.question_subtype_zh if state.analysis else "",
                stance=state.selected_stance or "A clear, task-appropriate writing path",
                arguments=self._format_arguments_for_outline(state.arguments),
                rag_context=state.rag_context,
            )
            response = self.llm.generate(
                rendered["user"],
                system_message=rendered["system"],
                temperature=0.5,
                max_tokens=3000,
            )
            state.outline = self._parse_outline(response)
        except Exception as exc:
            logger.error("Outline building failed: %s", exc)
            state.add_error(f"Outline building failed: {exc}")
        return state

    def evaluate_essay(
        self,
        question: str,
        essay: str,
        outline: str = "",
        rag_context: str = "",
        prompt_version: str = "v1",
    ) -> str:
        """Evaluate a learner essay for idea quality and an actual IELTS band estimate."""
        prompt = self.prompts.get("essay_evaluator", prompt_version)
        rendered = prompt.render(
            question=question,
            essay=essay,
            outline=outline,
            rag_context=rag_context,
        )
        return self.llm.generate(
            rendered["user"],
            system_message=rendered["system"],
            temperature=0.3,
            max_tokens=2200,
        )

    @staticmethod
    def _fallback_context(structure_context: str, vocab_context: str) -> str:
        parts = [part for part in [structure_context, vocab_context] if part.strip()]
        if parts:
            return "\n\n".join(parts)
        return "No local writing reference was available. Use general IELTS Task 2 standards."

    @staticmethod
    def _build_stance_sections(state: WorkflowState) -> str:
        qtype = state.analysis.question_type if state.analysis else "opinion"

        if qtype == "discussion":
            return """## View A
Generate 1-2 strong arguments for the first view in the question.

## View B
Generate 1-2 strong arguments for the second view in the question.

## Own View
Generate a clear personal position that can synthesize or choose between the two views."""

        if qtype == "advantages_disadvantages":
            return """## Advantages Outweigh Disadvantages
Generate 1-2 arguments showing why the benefits are more important.

## Disadvantages Outweigh Advantages
Generate 1-2 arguments showing why the drawbacks are more serious.

## Balanced View
Generate a nuanced position that weighs conditions, scope, or long-term effects."""

        if qtype == "problem_solution":
            return """## Problems
Identify 2-3 concrete problems caused by the issue.

## Solutions
Suggest practical solutions that directly match those problems."""

        if qtype in {"two_part", "two_part_question"}:
            return """## Question 1
Generate a direct answer to the first question with reasoning and example.

## Question 2
Generate a direct answer to the second question with reasoning and example."""

        return """## Agree
Generate 1-2 arguments supporting the statement.

## Disagree
Generate 1-2 arguments challenging the statement.

## Partly Agree
Generate a nuanced position with a clear leaning, not a vague 'it depends' answer."""

    @staticmethod
    def _parse_question_analysis(response: str, question_type: dict, topics: list) -> QuestionAnalysis:
        return QuestionAnalysis(
            question_type=question_type.get("key", "opinion"),
            question_type_zh=question_type.get("zh", "Opinion"),
            question_type_en=question_type.get("en", "Opinion Essay"),
            question_subtype=question_type.get("subtype", ""),
            question_subtype_zh=question_type.get("subtype_zh", ""),
            controversy=response,
            stance_recommendations=[],
            keywords=[],
            topics=[topic.get("key", "") for topic in topics],
        )

    @staticmethod
    def _parse_arguments(response: str) -> list:
        return [Argument(number=1, main_idea_en=response, main_idea_zh="")]

    @staticmethod
    def _parse_outline(response: str) -> EssayOutline:
        outline = EssayOutline()
        outline.tips = [response]
        return outline

    @staticmethod
    def _parse_evaluation(response: str) -> EvaluationResult:
        def score(label: str) -> float:
            pattern = rf"{re.escape(label)}\s*:\s*([1-9](?:\.[05])?)"
            match = re.search(pattern, response, flags=re.IGNORECASE)
            return float(match.group(1)) if match else 0.0

        result = EvaluationResult(
            task_response=score("Task Response"),
            coherence_cohesion=score("Coherence and Cohesion"),
            lexical_resource=score("Lexical Resource"),
            grammatical_range=score("Grammar Range and Accuracy"),
            overall_band=score("Overall"),
            suggestions=[response],
        )
        if result.overall_band == 0.0:
            scored = [
                result.task_response,
                result.coherence_cohesion,
                result.lexical_resource,
                result.grammatical_range,
            ]
            valid = [value for value in scored if value > 0]
            result.overall_band = round((sum(valid) / len(valid)) * 2) / 2 if valid else 0.0
        return result

    @staticmethod
    def _format_arguments_for_outline(arguments: list) -> str:
        lines = []
        for arg in arguments:
            if arg.number == 0:
                continue
            lines.append(f"Idea {arg.number}: {arg.main_idea_en}")
            if arg.main_idea_zh:
                lines.append(f"  ({arg.main_idea_zh})")
        return "\n".join(lines) if lines else "No generated idea available."
