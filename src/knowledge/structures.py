"""Essay structure pattern management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


class StructureManager:
    """Manages IELTS essay structure patterns."""

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent / "data")
        self.data_dir = Path(data_dir)
        self._structures: List[dict] = []
        self._load()

    def _load(self):
        filepath = self.data_dir / "knowledge_base" / "structures" / "essay_structures.json"
        if not filepath.exists():
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._structures = data.get("structures", [])

    def get_by_type(self, question_type: str) -> Optional[dict]:
        aliases = {"two_part": "two_part_question"}
        wanted = aliases.get(question_type, question_type)
        for structure in self._structures:
            if structure.get("type") == wanted:
                return structure
        return None

    def format_for_prompt(self, question_type: str) -> str:
        structure = self.get_by_type(question_type)
        if not structure:
            return ""

        lines = [
            f"## Essay Structure Reference: {structure.get('name_en', '')}",
            f"Description: {structure.get('description', '')}",
            "",
        ]

        for section, content in structure.get("structure", {}).items():
            section_name = section.replace("_", " ").title()
            lines.append(f"### {section_name}")
            if isinstance(content, dict):
                if content.get("purpose"):
                    lines.append(f"- Purpose: {content.get('purpose')}")
                if content.get("structure"):
                    lines.append(f"- Logic: {content.get('structure')}")
                if content.get("template"):
                    lines.append(f"- Template: {content.get('template')}")
                if content.get("word_count"):
                    lines.append(f"- Word count: {content.get('word_count')}")
            lines.append("")

        if structure.get("band7_tips"):
            lines.append("### Band 7 Tips")
            for tip in structure["band7_tips"]:
                lines.append(f"- {tip}")

        return "\n".join(lines)

    @property
    def available_types(self) -> List[str]:
        return [structure.get("type", "") for structure in self._structures]
