"""Scenario detection and routing for the learning agent."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from .state import Scenario

logger = logging.getLogger(__name__)


class ScenarioRouter:
    """Detect which learning scenario the user is asking for."""

    DEEPEN_KEYWORDS = [
        "expand",
        "deepen",
        "elaborate",
        "my idea is",
        "my point is",
        "help me develop",
        "develop this",
    ]

    EVALUATE_KEYWORDS = [
        "evaluate",
        "assess",
        "score",
        "check my",
        "rate my",
        "what band",
        "band score",
        "can this get",
    ]

    TOPIC_KEYWORDS = {
        "education": ["education", "school", "university", "student"],
        "technology": ["technology", "internet", "digital", "ai"],
        "environment": ["environment", "pollution", "climate"],
        "crime": ["crime", "law", "punishment"],
        "health": ["health", "medical", "diet"],
        "transport": ["transport", "traffic", "road"],
        "work": ["work", "job", "career"],
        "globalization": ["globalization", "international", "trade"],
    }

    @classmethod
    def detect(
        cls, user_input: str, has_idea: bool = False, has_outline: bool = False
    ) -> Tuple[Scenario, Optional[str]]:
        text = user_input.lower()

        if has_outline:
            logger.info("Detected scenario: EVALUATE (outline provided)")
            return Scenario.EVALUATE, None

        if has_idea:
            logger.info("Detected scenario: DEEPEN (idea provided)")
            return Scenario.DEEPEN, None

        eval_score = sum(1 for keyword in cls.EVALUATE_KEYWORDS if keyword in text)
        deepen_score = sum(1 for keyword in cls.DEEPEN_KEYWORDS if keyword in text)

        if eval_score > deepen_score and eval_score > 0:
            return Scenario.EVALUATE, None
        if deepen_score > 0:
            return Scenario.DEEPEN, None

        topic = cls._detect_topic(text)
        if topic and cls._is_topic_only_query(text):
            return Scenario.PRACTICE, topic

        return Scenario.GENERATE, None

    @classmethod
    def _detect_topic(cls, text: str) -> Optional[str]:
        scores = {
            topic: sum(1 for keyword in keywords if keyword.lower() in text)
            for topic, keywords in cls.TOPIC_KEYWORDS.items()
        }
        scores = {topic: score for topic, score in scores.items() if score > 0}
        if not scores:
            return None
        return max(scores, key=scores.get)

    @staticmethod
    def _is_topic_only_query(text: str) -> bool:
        stripped = text.strip().rstrip("?.")
        return len(stripped) < 30
