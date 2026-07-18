"""IELTS test case management — loads and manages the question bank."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TestCaseManager:
    """Manages the IELTS question test bank for harness evaluation."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the test case manager.

        Args:
            data_dir: Path to data/ directory.
        """
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent / "data")
        self.data_dir = Path(data_dir)
        self._cases: List[Dict] = []
        self._load()

    def _load(self):
        """Load test cases from JSON file."""
        filepath = self.data_dir / "test_cases" / "ielts_questions.json"
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                self._cases = data.get("questions", [])
            logger.info(f"Loaded {len(self._cases)} test cases")

    def get_all(self, topic: Optional[str] = None) -> List[Dict]:
        """Get all test cases, optionally filtered by topic.

        Args:
            topic: Optional topic filter.

        Returns:
            List of test case dicts.
        """
        if not topic:
            return list(self._cases)

        return [
            c for c in self._cases
            if topic.lower() in [t.lower() for t in c.get("topics", [])]
        ]

    def get_by_question_type(self, question_type: str) -> List[Dict]:
        """Get test cases by question type.

        Args:
            question_type: e.g. 'opinion', 'discussion'.

        Returns:
            Filtered list of test case dicts.
        """
        return [
            c for c in self._cases
            if c.get("question_type", "").lower() == question_type.lower()
        ]

    def sample(self, count: int = 10, topic: Optional[str] = None) -> List[Dict]:
        """Get a random sample of test cases.

        Args:
            count: Number of cases to sample.
            topic: Optional topic filter.

        Returns:
            Sampled list of test case dicts.
        """
        import random

        pool = self.get_all(topic)
        if count >= len(pool):
            return pool
        return random.sample(pool, count)

    @property
    def count(self) -> int:
        """Total number of loaded test cases."""
        return len(self._cases)

    @property
    def topics_covered(self) -> List[str]:
        """List of unique topics covered by test cases."""
        topics = set()
        for case in self._cases:
            for t in case.get("topics", []):
                topics.add(t)
        return sorted(topics)
