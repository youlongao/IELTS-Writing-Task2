"""Workflow state management for the agent pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Scenario(str, Enum):
    """User scenario type."""

    GENERATE = "generate"  # Full pipeline: question -> ideas -> outline
    DEEPEN = "deepen"  # Expand existing shallow ideas
    EVALUATE = "evaluate"  # Evaluate a user-submitted outline
    PRACTICE = "practice"  # Topic-based practice with random question


@dataclass
class QuestionAnalysis:
    """Output of Step 1: Question Analysis."""

    question_type: str = ""  # opinion, discussion, problem_solution, etc.
    question_type_zh: str = ""
    question_type_en: str = ""
    question_subtype: str = ""
    question_subtype_zh: str = ""
    controversy: str = ""  # Core issue analysis
    stance_recommendations: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)  # Detected topic areas


@dataclass
class Argument:
    """A single argument with full logic chain."""

    number: int = 0
    main_idea_en: str = ""
    main_idea_zh: str = ""
    claim_en: str = ""
    claim_zh: str = ""
    reason_en: str = ""
    reason_zh: str = ""
    result_en: str = ""
    result_zh: str = ""
    link_en: str = ""
    link_zh: str = ""
    example_en: str = ""
    example_zh: str = ""
    collocations: List[Dict[str, str]] = field(default_factory=list)
    pitfalls: str = ""  # Logic pitfalls to avoid


@dataclass
class OutlineSection:
    """A single section of the essay outline."""

    purpose: str = ""
    topic_sentence_en: str = ""
    explanation_en: str = ""
    example_en: str = ""
    link_en: str = ""
    vocabulary: List[str] = field(default_factory=list)
    cohesive_devices: List[str] = field(default_factory=list)
    word_count_estimate: str = ""


@dataclass
class EssayOutline:
    """Complete 4-paragraph essay outline."""

    introduction: OutlineSection = field(default_factory=OutlineSection)
    body_paragraph_1: OutlineSection = field(default_factory=OutlineSection)
    body_paragraph_2: OutlineSection = field(default_factory=OutlineSection)
    conclusion: OutlineSection = field(default_factory=OutlineSection)
    tips: List[str] = field(default_factory=list)
    checklist: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Output of outline evaluation."""

    overall_band: float = 0.0
    task_response: float = 0.0
    coherence_cohesion: float = 0.0
    lexical_resource: float = 0.0
    grammatical_range: float = 0.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class WorkflowState:
    """Central state object flowing through the agent pipeline.

    Each step reads from and writes to this state object,
    accumulating outputs as the pipeline progresses.
    """

    # === Input ===
    user_input: str = ""  # The original user question or input
    scenario: Scenario = Scenario.GENERATE
    topic_filter: Optional[str] = None  # For PRACTICE scenario
    user_idea: Optional[str] = None  # For DEEPEN scenario
    user_outline: Optional[str] = None  # For EVALUATE scenario
    selected_stance: Optional[str] = None
    selected_argument_indices: Optional[List[int]] = None

    # === Step 1: Question Analysis ===
    analysis: Optional[QuestionAnalysis] = None

    # === Step 2: RAG Retrieval ===
    rag_context: str = ""  # Formatted context from retrieval
    rag_raw_results: Dict[str, List[Dict]] = field(default_factory=dict)

    # === Step 3: Idea Generation ===
    arguments: List[Argument] = field(default_factory=list)

    # === Step 4: Outline ===
    outline: Optional[EssayOutline] = None

    # === Evaluation (for EVALUATE scenario) ===
    evaluation: Optional[EvaluationResult] = None

    # === Metadata ===
    model_used: str = ""
    prompt_version: str = "v1"
    errors: List[str] = field(default_factory=list)

    def add_error(self, error: str):
        """Record an error that occurred during processing."""
        self.errors.append(error)

    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
