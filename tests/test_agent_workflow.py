"""Tests for the agent workflow pipeline."""

from unittest.mock import Mock

from src.agent.scenarios import ScenarioRouter
from src.agent.state import EssayOutline, QuestionAnalysis, Scenario, WorkflowState
from src.agent.workflow import AgentWorkflow


class TestScenarioRouter:
    """Test scenario detection."""

    def test_detect_generate_scenario(self):
        question = "Some people think governments should invest more in public transport."
        scenario, topic = ScenarioRouter.detect(question)
        assert scenario == Scenario.GENERATE
        assert topic is None

    def test_detect_deepen_scenario(self):
        query = "help me develop this idea: public transport reduces pollution"
        scenario, topic = ScenarioRouter.detect(query)
        assert scenario == Scenario.DEEPEN
        assert topic is None

    def test_detect_deepen_with_idea_flag(self):
        scenario, topic = ScenarioRouter.detect("government should invest", has_idea=True)
        assert scenario == Scenario.DEEPEN
        assert topic is None

    def test_detect_evaluate_with_outline_flag(self):
        scenario, topic = ScenarioRouter.detect("some outline", has_outline=True)
        assert scenario == Scenario.EVALUATE
        assert topic is None

    def test_detect_evaluate_keyword(self):
        scenario, topic = ScenarioRouter.detect("check my outline and tell me what band it may get")
        assert scenario == Scenario.EVALUATE
        assert topic is None

    def test_detect_practice_scenario(self):
        scenario, topic = ScenarioRouter.detect("education")
        assert scenario == Scenario.PRACTICE
        assert topic == "education"


class TestWorkflowState:
    """Test workflow state management."""

    def test_initial_state(self):
        state = WorkflowState(user_input="test question")
        assert state.user_input == "test question"
        assert state.scenario == Scenario.GENERATE
        assert state.analysis is None
        assert state.arguments == []
        assert state.outline is None

    def test_add_error(self):
        state = WorkflowState()
        state.add_error("Test error")
        assert state.has_errors()
        assert len(state.errors) == 1

    def test_no_errors_initially(self):
        state = WorkflowState()
        assert not state.has_errors()


class TestAgentWorkflow:
    """Test the agent workflow pipeline."""

    def test_workflow_initialization(self):
        mock_llm = Mock()
        workflow = AgentWorkflow(llm=mock_llm)
        assert workflow.llm == mock_llm
        assert workflow.prompts is not None

    def test_step_1_analyze_question(self):
        mock_llm = Mock()
        mock_llm.model_name = "test-model"
        mock_llm.generate.return_value = "## Question Type\nOpinion\n\n## Core Issue\nFunding priorities."

        workflow = AgentWorkflow(llm=mock_llm)
        state = WorkflowState(
            user_input="Some people think governments should invest more in public transport."
        )

        result = workflow._step_1_analyze_question(state)
        assert result.analysis is not None
        assert result.model_used == "test-model"

    def test_step_1_fallback_on_llm_failure(self):
        mock_llm = Mock()
        mock_llm.model_name = "test-model"
        mock_llm.generate.side_effect = Exception("API error")

        workflow = AgentWorkflow(llm=mock_llm)
        state = WorkflowState(
            user_input="Discuss both views and give your opinion on free education."
        )

        result = workflow._step_1_analyze_question(state)
        assert result.analysis is not None
        assert result.analysis.question_type == "discussion"

    def test_step_2_without_retriever(self):
        mock_llm = Mock()
        workflow = AgentWorkflow(llm=mock_llm, retriever=None)

        state = WorkflowState(user_input="test")
        state.analysis = QuestionAnalysis(question_type="opinion", topics=["education"])

        result = workflow._step_2_retrieve_knowledge(state)
        assert result.rag_context

    def test_step_4_without_arguments_should_error(self):
        mock_llm = Mock()
        workflow = AgentWorkflow(llm=mock_llm)

        state = WorkflowState(user_input="test")
        state.analysis = QuestionAnalysis()

        result = workflow._step_4_build_outline(state)
        assert result.has_errors()

    def test_full_pipeline_smoke(self):
        mock_llm = Mock()
        mock_llm.model_name = "test-model"
        mock_llm.generate.return_value = "Analysis result with arguments and outline suggestions."

        workflow = AgentWorkflow(llm=mock_llm, retriever=None)
        state = WorkflowState(
            user_input="Some people think governments should invest more in public transport. To what extent do you agree?",
            scenario=Scenario.GENERATE,
        )

        result = workflow.run(state)
        assert result.analysis is not None
        assert not result.has_errors()

    def test_evaluate_scenario_stops_after_step_3(self):
        mock_llm = Mock()
        mock_llm.model_name = "test-model"
        mock_llm.generate.return_value = (
            "Task Response: 6.5\n"
            "Coherence and Cohesion: 6.5\n"
            "Lexical Resource: 6.0\n"
            "Grammar Range and Accuracy: 6.0\n"
            "Overall: 6.5\n"
            "Suggestions:\n- Good structure."
        )

        workflow = AgentWorkflow(llm=mock_llm, retriever=None)
        state = WorkflowState(
            user_input="Should university be free?",
            scenario=Scenario.EVALUATE,
            user_outline="Introduction... Body 1... Body 2... Conclusion...",
        )

        result = workflow.run(state)
        assert result.evaluation is not None
        assert result.outline is None

    def test_evaluate_essay_uses_essay_evaluator_prompt(self):
        mock_llm = Mock()
        mock_llm.model_name = "test-model"
        mock_llm.generate.return_value = "## Idea Quality Check\nGood position."

        workflow = AgentWorkflow(llm=mock_llm, retriever=None)
        result = workflow.evaluate_essay(
            question="Should university be free?",
            essay="Universities should be free because education benefits society.",
            outline="Body 1: social mobility.",
            rag_context="Task Response standards.",
        )

        assert "Idea Quality Check" in result
        prompt = mock_llm.generate.call_args.args[0]
        assert "Student Essay" in prompt
        assert "Actual Band Estimate" in prompt

    def test_generate_sample_essay_uses_essay_generator_prompt(self):
        mock_llm = Mock()
        mock_llm.model_name = "test-model"
        mock_llm.generate.return_value = "Sample essay."

        workflow = AgentWorkflow(llm=mock_llm, retriever=None)
        state = WorkflowState(
            user_input="Should university be free?",
            scenario=Scenario.GENERATE,
        )
        state.analysis = QuestionAnalysis(
            question_type_zh="观点类",
            question_type_en="Opinion Essay",
        )
        state.outline = EssayOutline()
        state.outline.tips = ["Introduction, Body 1, Body 2, Conclusion."]

        result = workflow.generate_sample_essay(state)

        assert result == "Sample essay."
        prompt = mock_llm.generate.call_args.args[0]
        assert "Write a 250-280 word sample IELTS Task 2 essay" in prompt
        assert state.trace["final_prompt_stage"] == "essay_generator"
