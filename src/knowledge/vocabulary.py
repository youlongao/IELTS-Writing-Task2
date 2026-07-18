"""Band 7 vocabulary and collocation management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


class CollocationManager:
    """Manages IELTS-friendly collocations for writing support."""

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent / "data")
        self.data_dir = Path(data_dir)
        self._collocations: List[dict] = []
        self._load()

    def _load(self):
        filepath = self.data_dir / "knowledge_base" / "vocabulary" / "collocations.json"
        if not filepath.exists():
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._collocations = data.get("collocations", [])

    def get_by_topic(self, topic: str, limit: int = 10) -> List[dict]:
        items = [
            item
            for item in self._collocations
            if item.get("topic") in (topic, "general")
        ]
        return items[:limit]

    def get_by_function(self, function: str, limit: int = 10) -> List[dict]:
        items = [item for item in self._collocations if item.get("function") == function]
        return items[:limit]

    def search(self, query: str, limit: int = 10) -> List[dict]:
        query_lower = query.lower()
        items = [
            item
            for item in self._collocations
            if query_lower in item.get("english", "").lower()
            or query_lower in item.get("chinese", "").lower()
            or query_lower in item.get("example_sentence", "").lower()
        ]
        return items[:limit]

    def format_for_prompt(self, topic: Optional[str] = None, limit: int = 20) -> str:
        items = self.get_by_topic(topic, limit) if topic else self._collocations[:limit]
        if not items:
            return ""

        lines = ["## Band 7 Vocabulary Reference"]
        for item in items:
            example = item.get("example_sentence", "")
            lines.append(f"- {item.get('english', '')}: {example}")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._collocations)
