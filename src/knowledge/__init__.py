"""Topic taxonomy and question-type detection for IELTS Task 2."""

from __future__ import annotations

from typing import Dict, List, Optional


TOPICS: Dict[str, Dict[str, object]] = {
    "education": {
        "zh": "\u6559\u80b2",
        "keywords": ["education", "school", "university", "student", "teacher", "curriculum", "learning"],
    },
    "technology": {
        "zh": "\u79d1\u6280",
        "keywords": ["technology", "internet", "computer", "online", "digital", "ai", "robot", "automation"],
    },
    "environment": {
        "zh": "\u73af\u5883",
        "keywords": ["environment", "pollution", "climate", "carbon", "recycle", "energy", "wildlife"],
    },
    "crime": {
        "zh": "\u72af\u7f6a",
        "keywords": ["crime", "criminal", "prison", "punishment", "law", "police", "offender"],
    },
    "health": {
        "zh": "\u5065\u5eb7",
        "keywords": ["health", "medical", "doctor", "hospital", "diet", "exercise", "disease"],
    },
    "transport": {
        "zh": "\u4ea4\u901a",
        "keywords": ["transport", "traffic", "road", "car", "bus", "train", "metro", "commute"],
    },
    "work": {
        "zh": "\u5de5\u4f5c",
        "keywords": ["work", "job", "career", "employee", "employer", "workplace", "salary"],
    },
    "globalization": {
        "zh": "\u5168\u7403\u5316",
        "keywords": [
            "globalization",
            "global",
            "international",
            "culture",
            "fashion",
            "trend",
            "trends",
            "brand",
            "brands",
            "dress",
            "clothing",
            "clothes",
            "consumer",
            "trade",
            "tourism",
            "migration",
        ],
    },
}


QUESTION_TYPES: Dict[str, Dict[str, object]] = {
    "opinion": {
        "zh": "\u89c2\u70b9\u7c7b",
        "en": "Opinion Essay",
        "patterns": [
            "to what extent do you agree or disagree",
            "to what extent do you think",
            "do you agree or disagree",
            "what is your opinion",
            "is this a positive or negative development",
            "is this a negative trend",
            "is this a positive trend",
            "negative trend",
            "positive trend",
        ],
    },
    "discussion": {
        "zh": "\u8ba8\u8bba\u7c7b",
        "en": "Discussion Essay",
        "patterns": [
            "discuss both views",
            "discuss both these views",
            "discuss both sides",
            "give your own opinion",
            "how has",
            "how have",
            "why has",
            "why have",
            "what has made",
            "what have made",
            "what are the reasons for",
            "what are the effects of",
            "what effects",
        ],
    },
    "problem_solution": {
        "zh": "\u95ee\u9898\u89e3\u51b3\u7c7b",
        "en": "Problem-Solution Essay",
        "patterns": [
            "what problems",
            "what are the problems",
            "problems caused",
            "what solutions",
            "how can",
            "how could",
            "what measures",
        ],
    },
    "advantages_disadvantages": {
        "zh": "\u5229\u5f0a\u7c7b",
        "en": "Advantages and Disadvantages Essay",
        "patterns": [
            "advantages and disadvantages",
            "advantages outweigh the disadvantages",
            "benefits outweigh the drawbacks",
            "outweigh the disadvantages",
        ],
    },
    "two_part": {
        "zh": "\u53cc\u95ee\u9898\u7c7b",
        "en": "Two-Part Question",
        "patterns": [
            "what are the reasons",
            "why is this",
            "why do",
            "what factors",
            "what can be done",
        ],
    },
}


class TopicMatcher:
    """Detect likely IELTS topics from a question."""

    @classmethod
    def detect_topic(cls, question: str) -> List[Dict[str, str]]:
        text = question.lower()
        matches: List[Dict[str, str]] = []

        for key, info in TOPICS.items():
            keywords = info.get("keywords", [])
            score = sum(1 for keyword in keywords if str(keyword).lower() in text)
            if score:
                matches.append({
                    "key": key,
                    "zh": str(info.get("zh", key)),
                    "score": str(score),
                })

        matches.sort(key=lambda item: int(item["score"]), reverse=True)
        return matches[:3]

    @classmethod
    def get_topic_info(cls, topic_key: str) -> Optional[Dict[str, object]]:
        return TOPICS.get(topic_key)

    @classmethod
    def list_topics(cls) -> List[Dict[str, str]]:
        return [{"key": key, "zh": str(info.get("zh", key))} for key, info in TOPICS.items()]


class QuestionTypeDetector:
    """Detect the IELTS Task 2 question type."""

    @classmethod
    def detect(cls, question: str) -> Dict[str, str]:
        text = question.lower()

        direct_discussion_patterns = [
            "how has",
            "how have",
            "why has",
            "why have",
            "what has made",
            "what have made",
            "what are the reasons for",
            "what are the effects of",
            "what effects",
        ]
        if text.count("?") <= 1 and any(pattern in text for pattern in direct_discussion_patterns):
            info = QUESTION_TYPES["discussion"]
            return {"key": "discussion", "zh": str(info["zh"]), "en": str(info["en"])}

        for key, info in QUESTION_TYPES.items():
            for pattern in info.get("patterns", []):
                if str(pattern).lower() in text:
                    result = {
                        "key": key,
                        "zh": str(info.get("zh", key)),
                        "en": str(info.get("en", key)),
                    }
                    if key == "opinion":
                        result.update(cls._detect_opinion_subtype(text))
                    return result

        if "?" in text and text.count("?") >= 2:
            info = QUESTION_TYPES["two_part"]
            return {"key": "two_part", "zh": str(info["zh"]), "en": str(info["en"])}

        info = QUESTION_TYPES["opinion"]
        result = {"key": "opinion", "zh": str(info["zh"]), "en": str(info["en"])}
        result.update(cls._detect_opinion_subtype(text))
        return result

    @staticmethod
    def _detect_opinion_subtype(text: str) -> Dict[str, str]:
        if "negative trend" in text:
            return {
                "subtype": "negative_trend",
                "subtype_zh": "\u8d1f\u9762\u8d8b\u52bf\u5224\u65ad",
            }
        if "positive trend" in text:
            return {
                "subtype": "positive_trend",
                "subtype_zh": "\u6b63\u9762\u8d8b\u52bf\u5224\u65ad",
            }
        if "positive or negative development" in text:
            return {
                "subtype": "positive_negative_development",
                "subtype_zh": "\u6b63\u8d1f\u9762\u53d1\u5c55\u5224\u65ad",
            }
        if "agree or disagree" in text:
            return {
                "subtype": "agree_disagree",
                "subtype_zh": "\u540c\u610f\u4e0e\u5426\u5224\u65ad",
            }
        return {"subtype": "general_opinion", "subtype_zh": "\u4e00\u822c\u89c2\u70b9\u5224\u65ad"}
