"""Evaluation metrics for IELTS writing quality assessment.

Each metric scores on a 1-9 band scale aligned with official IELTS descriptors.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MetricScores:
    """Container for all evaluation metric scores."""

    task_response: float = 0.0
    coherence_cohesion: float = 0.0
    lexical_resource: float = 0.0
    grammatical_range: float = 0.0
    specificity_score: float = 0.0  # How specific vs generic the arguments are
    band_alignment: float = 0.0  # Overall alignment with Band 7 descriptors
    overall: float = 0.0

    comments: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "task_response": self.task_response,
            "coherence_cohesion": self.coherence_cohesion,
            "lexical_resource": self.lexical_resource,
            "grammatical_range": self.grammatical_range,
            "specificity_score": self.specificity_score,
            "band_alignment": self.band_alignment,
            "overall": self.overall,
            "comments": self.comments,
        }


class MetricsCalculator:
    """Calculates IELTS-aligned quality metrics for agent outputs.

    Uses a combination of structural checks and LLM-as-judge evaluation.
    """

    # IELTS Band 7 descriptors for each criterion
    BAND_7_DESCRIPTORS = {
        "task_response": """
            - Addresses all parts of the question
            - Presents a clear position throughout
            - Main ideas are extended and supported
            - No irrelevant content
        """,
        "coherence_cohesion": """
            - Logically organizes information and ideas
            - Clear progression throughout
            - Uses a range of cohesive devices appropriately
            - Each paragraph has a clear central topic
        """,
        "lexical_resource": """
            - Uses a sufficient range of vocabulary
            - Uses some less common and idiomatic vocabulary
            - Shows awareness of style and collocation
            - Occasional errors in word choice
        """,
        "grammatical_range": """
            - Uses a variety of complex structures
            - Produces frequent error-free sentences
            - Good control of grammar and punctuation
            - Some errors persist but don't impede communication
        """,
    }

    @classmethod
    def get_judge_prompt(
        cls, question: str, output: str, metric: str
    ) -> str:
        """Generate an LLM-as-judge evaluation prompt for a specific metric.

        Args:
            question: The original IELTS question.
            output: The agent's output to evaluate.
            metric: The metric to evaluate.

        Returns:
            Evaluation prompt string.
        """
        descriptor = cls.BAND_7_DESCRIPTORS.get(metric, "")

        return f"""## IELTS Writing Task 2 — Quality Evaluation

You are an experienced IELTS examiner. Evaluate the following AI-generated content
for the specific criterion below. Score on a 1-9 band scale.

### Original Question:
{question}

### AI-Generated Output (to evaluate):
{output[:2000]}

### Evaluation Criterion: {metric}

**Band 7 Descriptor**:
{descriptor}

### Instructions:
1. Read the AI output carefully against the Band 7 descriptor above.
2. Determine if the output MEETS, EXCEEDS, or FALLS BELOW Band 7 for this criterion.
3. Assign a band score (1.0-9.0, increments of 0.5).
4. Provide a brief justification (2-3 sentences).

### Response Format:
Score: X.X
Justification: [Your brief justification]
"""

    @classmethod
    def calculate_overall(cls, scores: Dict[str, float]) -> float:
        """Calculate overall band score from individual metrics.

        Uses IELTS-style averaging (rounded to nearest 0.5).

        Args:
            scores: Dict of metric_name -> score.

        Returns:
            Overall band score.
        """
        if not scores:
            return 0.0

        avg = sum(scores.values()) / len(scores)
        # Round to nearest 0.5
        return round(avg * 2) / 2
