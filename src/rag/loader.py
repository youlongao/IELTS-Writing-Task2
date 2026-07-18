"""Knowledge base document loader.

Loads markdown topic files, rubric files, JSON data files, and chunks them
for ingestion into the ChromaDB vector store.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class KnowledgeLoader:
    """Loads and chunks IELTS knowledge base documents."""

    def __init__(self, data_dir: Optional[str] = None, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the loader.

        Args:
            data_dir: Path to data/ directory.
            chunk_size: Target chunk size in characters.
            chunk_overlap: Overlap between chunks in characters.
        """
        if data_dir is None:
            data_dir = str(Path(__file__).parent.parent.parent / "data")
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_topics(self) -> List[Tuple[str, Dict]]:
        """Load all topic markdown files.

        Returns:
            List of (content, metadata) tuples.
        """
        topics_dir = self.data_dir / "knowledge_base" / "topics"
        if not topics_dir.exists():
            return []

        documents = []
        for filepath in topics_dir.glob("*.md"):
            topic_name = filepath.stem
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Split into sections for better chunking
            sections = self._split_markdown_sections(content)
            for section_title, section_text in sections:
                chunks = self._chunk_text(section_text)
                for i, chunk in enumerate(chunks):
                    documents.append((
                        chunk,
                        {
                            "source": str(filepath),
                            "topic": topic_name,
                            "section": section_title,
                            "chunk_index": i,
                            "type": "topic_knowledge",
                        },
                    ))

        return documents

    def load_vocabulary(self) -> List[Tuple[str, Dict]]:
        """Load collocations from JSON.

        Returns:
            List of (content, metadata) tuples.
        """
        filepath = self.data_dir / "knowledge_base" / "vocabulary" / "collocations.json"
        if not filepath.exists():
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        for item in data.get("collocations", []):
            # Create a rich text representation
            text = (
                f"Collocation: {item.get('english', '')}\n"
                f"Chinese: {item.get('chinese', '')}\n"
                f"Function: {item.get('function', '')}\n"
                f"Topic: {item.get('topic', '')}\n"
                f"Example: {item.get('example_sentence', '')}\n"
                f"Band Level: {item.get('band_level', '')}"
            )
            documents.append((
                text,
                {
                    "source": str(filepath),
                    "topic": item.get("topic", "general"),
                    "function": item.get("function", ""),
                    "type": "vocabulary",
                },
            ))

        return documents

    def load_structures(self) -> List[Tuple[str, Dict]]:
        """Load essay structures from JSON.

        Returns:
            List of (content, metadata) tuples.
        """
        filepath = self.data_dir / "knowledge_base" / "structures" / "essay_structures.json"
        if not filepath.exists():
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        for item in data.get("structures", []):
            # Flatten the structure into a text representation
            text = f"Type: {item.get('type', '')}\n"
            text += f"Name: {item.get('name_en', '')} ({item.get('name_zh', '')})\n"
            text += f"Description: {item.get('description', '')}\n"

            for section, content in item.get("structure", {}).items():
                if isinstance(content, dict):
                    text += f"{section}: {json.dumps(content, ensure_ascii=False)}\n"

            if item.get("band7_tips"):
                text += "Band 7 Tips: " + "; ".join(item["band7_tips"])

            documents.append((
                text,
                {
                    "source": str(filepath),
                    "question_type": item.get("type", ""),
                    "type": "essay_structure",
                },
            ))

        return documents

    def load_rubrics(self) -> List[Tuple[str, Dict]]:
        """Load IELTS assessment criteria from rubric documents.

        Supported source formats:
        - .md / .txt for manually curated criteria notes
        - .pdf when pypdf or PyPDF2 is installed
        """
        rubrics_dir = self.data_dir / "knowledge_base" / "rubrics"
        if not rubrics_dir.exists():
            return []

        documents = []
        supported_suffixes = {".md", ".txt", ".pdf"}
        for filepath in sorted(rubrics_dir.iterdir()):
            if not filepath.is_file() or filepath.suffix.lower() not in supported_suffixes:
                continue

            content = self._read_rubric_file(filepath)
            if not content.strip():
                logger.warning("No readable rubric content found in %s", filepath)
                continue

            sections = self._split_markdown_sections(content)
            if not sections:
                sections = [(filepath.stem, content)]

            for section_title, section_text in sections:
                chunks = self._chunk_text(section_text)
                for i, chunk in enumerate(chunks):
                    documents.append((
                        chunk,
                        {
                            "source": str(filepath),
                            "rubric": filepath.stem,
                            "section": section_title,
                            "chunk_index": i,
                            "type": "assessment_rubric",
                        },
                    ))

        return documents

    def load_sample_essays(self) -> List[Tuple[str, Dict]]:
        """Load sample Band 7+ essays from JSON.

        Returns:
            List of (content, metadata) tuples.
        """
        filepath = self.data_dir / "knowledge_base" / "sample_essays" / "band7_samples.json"
        if not filepath.exists():
            return []

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents = []
        for item in data.get("samples", []):
            text = f"Topic: {item.get('topic', '')}\n"
            text += f"Question Type: {item.get('question_type', '')}\n"
            text += f"Paragraph Type: {item.get('paragraph_type', '')}\n"
            text += f"Content: {item.get('content', '')}\n"
            if item.get("analysis"):
                text += f"Analysis: {json.dumps(item['analysis'], ensure_ascii=False)}"

            documents.append((
                text,
                {
                    "source": str(filepath),
                    "topic": item.get("topic", ""),
                    "question_type": item.get("question_type", ""),
                    "type": "sample_essay",
                },
            ))

        return documents

    def load_all(self) -> Dict[str, List[Tuple[str, Dict]]]:
        """Load all knowledge base documents.

        Returns:
            Dict mapping collection_name -> List of (content, metadata).
        """
        return {
            "topics": self.load_topics(),
            "vocabulary": self.load_vocabulary(),
            "structures": self.load_structures(),
            "rubrics": self.load_rubrics(),
            "sample_essays": self.load_sample_essays(),
        }

    def _read_rubric_file(self, filepath: Path) -> str:
        suffix = filepath.suffix.lower()
        if suffix in {".md", ".txt"}:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        if suffix == ".pdf":
            return self._extract_pdf_text(filepath)
        return ""

    @staticmethod
    def _extract_pdf_text(filepath: Path) -> str:
        reader_cls = None
        try:
            from pypdf import PdfReader

            reader_cls = PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader

                reader_cls = PdfReader
            except ImportError:
                logger.warning(
                    "PDF rubric support requires pypdf or PyPDF2. Skipping %s",
                    filepath,
                )
                return ""

        try:
            reader = reader_cls(str(filepath))
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"## Page {page_number}\n{text.strip()}")
            return "\n\n".join(pages)
        except Exception as exc:
            logger.warning("Failed to extract PDF rubric text from %s: %s", filepath, exc)
            return ""

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks.

        Args:
            text: The text to chunk.

        Returns:
            List of text chunks.
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at a sentence boundary
            if end < len(text):
                # Look for last period, question mark, or newline
                for sep in ["\n\n", "\n", ". ", "? ", "! "]:
                    last_sep = chunk.rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        chunk = text[start:end]
                        break

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return [c for c in chunks if c]

    @staticmethod
    def _split_markdown_sections(content: str) -> List[Tuple[str, str]]:
        """Split markdown content into sections by headers.

        Args:
            content: Raw markdown text.

        Returns:
            List of (section_title, section_text) tuples.
        """
        # Split by ## headers
        sections = re.split(r"\n(?=## )", content)

        result = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            # Extract section title
            lines = section.split("\n", 1)
            title = lines[0].lstrip("#").strip() if lines else ""
            body = lines[1].strip() if len(lines) > 1 else ""
            result.append((title, body))

        return result
