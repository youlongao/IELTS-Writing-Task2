"""Tests for RAG retriever and related components."""

import sys
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.rag.retriever import Retriever
from src.rag.embedder import Embedder
from src.rag.loader import KnowledgeLoader
from src.rag.store import VectorStore
from src.knowledge.topics import TopicMatcher, QuestionTypeDetector


class TestTopicMatcher:
    """Test topic detection from IELTS questions."""

    def test_detect_education_topic(self):
        """Should detect education topic from education-related question."""
        question = "Some people believe that university education should be free for everyone."
        matches = TopicMatcher.detect_topic(question)
        assert len(matches) > 0
        assert matches[0]["key"] == "education"

    def test_detect_environment_topic(self):
        """Should detect environment topic from climate question."""
        question = "Climate change is the biggest threat facing humanity today."
        matches = TopicMatcher.detect_topic(question)
        assert len(matches) > 0
        assert matches[0]["key"] == "environment"

    def test_detect_multiple_topics(self):
        """Should detect multiple topics when applicable."""
        question = "Technology has changed the way students learn in schools."
        matches = TopicMatcher.detect_topic(question)
        topics = [m["key"] for m in matches]
        assert "technology" in topics or "education" in topics

    def test_detect_global_fashion_topic(self):
        """Global fashion and dress trends should map to globalization."""
        question = (
            "Many aspects of the way people dress today are influenced by global "
            "fashion trends. How has global fashion become such a strong influence "
            "on people's lives?"
        )
        matches = TopicMatcher.detect_topic(question)
        assert len(matches) > 0
        assert matches[0]["key"] == "globalization"

    def test_no_topic_match(self):
        """Should return empty list when no topic matches."""
        question = "What is the meaning of life?"
        matches = TopicMatcher.detect_topic(question)
        assert matches == []

    def test_list_topics(self):
        """Should list all 8 topics."""
        topics = TopicMatcher.list_topics()
        assert len(topics) == 8


class TestQuestionTypeDetector:
    """Test question type detection."""

    def test_detect_opinion_type(self):
        """Should detect opinion (agree/disagree) type."""
        question = "To what extent do you agree or disagree with this statement?"
        result = QuestionTypeDetector.detect(question)
        assert result["key"] == "opinion"

    def test_detect_negative_trend_opinion_subtype(self):
        """Negative trend prompts should be opinion with a specific subtype."""
        question = (
            "Many people consider shopping malls as great places to spend their leisure "
            "time and meet others. People in the past, however, visited shopping malls "
            "only when necessary. To what extent do you think this is a negative trend?"
        )
        result = QuestionTypeDetector.detect(question)
        assert result["key"] == "opinion"
        assert result["subtype"] == "negative_trend"

    def test_detect_discussion_type(self):
        """Should detect discussion type."""
        question = "Discuss both views and give your opinion."
        result = QuestionTypeDetector.detect(question)
        assert result["key"] == "discussion"

    def test_detect_how_has_explanation_as_discussion(self):
        """Single how-has explanation questions should not default to opinion."""
        question = (
            "Many aspects of the way people dress today are influenced by global "
            "fashion trends. How has global fashion become such a strong influence "
            "on people's lives?"
        )
        result = QuestionTypeDetector.detect(question)
        assert result["key"] == "discussion"

    def test_detect_problem_solution_type(self):
        """Should detect problem-solution type."""
        question = "What are the problems caused by pollution? What can be done?"
        result = QuestionTypeDetector.detect(question)
        assert result["key"] == "problem_solution"

    def test_detect_advantages_disadvantages(self):
        """Should detect advantages/disadvantages type."""
        question = "Do the advantages of technology outweigh the disadvantages?"
        result = QuestionTypeDetector.detect(question)
        assert result["key"] == "advantages_disadvantages"


class TestEmbedder:
    """Test embedding model with lazy import."""

    def test_embedder_lazy_loading(self):
        """Embedder should lazy-load the SentenceTransformer model."""
        # Inject mock sentence_transformers module
        mock_st = MagicMock()
        mock_model = Mock()
        # simulate numpy array behavior: .tolist() returns the list
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_model.encode.return_value = mock_array
        mock_st.SentenceTransformer.return_value = mock_model
        sys.modules["sentence_transformers"] = mock_st

        try:
            embedder = Embedder(model_name="test-model")
            # Model should not be loaded yet
            assert embedder._model is None

            # First call triggers loading
            result = embedder.embed(["test text"])
            assert result == [[0.1, 0.2, 0.3]]
            assert embedder._model is not None
        finally:
            del sys.modules["sentence_transformers"]

    def test_embed_query(self):
        """embed_query should return a flat list."""
        mock_st = MagicMock()
        mock_model = Mock()
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_model.encode.return_value = mock_array
        mock_st.SentenceTransformer.return_value = mock_model
        sys.modules["sentence_transformers"] = mock_st

        try:
            embedder = Embedder()
            result = embedder.embed_query("test query")
            assert result == [0.1, 0.2, 0.3]
        finally:
            del sys.modules["sentence_transformers"]


class TestKnowledgeLoader:
    """Test knowledge base source loading."""

    def test_load_rubrics_from_markdown(self, tmp_path):
        """Rubric markdown files should be loaded into assessment_rubric documents."""
        rubrics_dir = tmp_path / "knowledge_base" / "rubrics"
        rubrics_dir.mkdir(parents=True)
        (rubrics_dir / "task2_criteria.md").write_text(
            "# IELTS Writing Task 2 Criteria\n\n"
            "## Task Response\n"
            "Band 7: presents a clear position throughout the response.\n",
            encoding="utf-8",
        )

        loader = KnowledgeLoader(data_dir=str(tmp_path), chunk_size=1000)
        documents = loader.load_rubrics()

        assert len(documents) == 1
        content, metadata = documents[0]
        assert "Band 7" in content
        assert metadata["type"] == "assessment_rubric"
        assert metadata["rubric"] == "task2_criteria"

    def test_load_all_includes_rubrics_collection(self, tmp_path):
        """load_all should expose rubrics as a first-class collection."""
        loader = KnowledgeLoader(data_dir=str(tmp_path))
        all_data = loader.load_all()

        assert "rubrics" in all_data


class TestRetriever:
    """Test retrieval result formatting."""

    def test_format_context_prioritizes_rubrics(self):
        """Assessment criteria should appear before topic ideas in RAG context."""
        retriever = Retriever(embedder=Mock(), store=Mock())
        context = retriever.format_context(
            {
                "topics": [{"content": "Topic idea content", "metadata": {}, "distance": 0.1}],
                "rubrics": [{"content": "Band 7 assessment criteria", "metadata": {}, "distance": 0.1}],
            },
            max_chars=1000,
        )

        assert "IELTS Writing Task 2 Assessment Criteria" in context
        assert context.index("Assessment Criteria") < context.index("High-Frequency Topic Ideas")


class TestVectorStore:
    """Test ChromaDB vector store with lazy import."""

    def test_rubrics_is_valid_collection(self):
        """Rubrics should be a first-class vector store collection."""
        assert "rubrics" in VectorStore.COLLECTIONS

    def test_store_initialization(self):
        """VectorStore should initialize with persist dir."""
        # Inject mock chromadb module
        mock_chromadb = MagicMock()
        mock_client = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        sys.modules["chromadb"] = mock_chromadb
        # Also add chromadb.config
        mock_config_module = MagicMock()
        sys.modules["chromadb.config"] = mock_config_module

        try:
            store = VectorStore(persist_dir="/tmp/test_chroma")
            assert store.persist_dir == "/tmp/test_chroma"
        finally:
            del sys.modules["chromadb"]
            if "chromadb.config" in sys.modules:
                del sys.modules["chromadb.config"]

    def test_unknown_collection_raises_error(self):
        """Querying an unknown collection should raise ValueError."""
        mock_chromadb = MagicMock()
        mock_client = Mock()
        mock_chromadb.PersistentClient.return_value = mock_client
        sys.modules["chromadb"] = mock_chromadb
        sys.modules["chromadb.config"] = MagicMock()

        try:
            store = VectorStore(persist_dir="/tmp/test")
            with pytest.raises(ValueError, match="Unknown collection"):
                store.get_or_create_collection("invalid_collection")
        finally:
            del sys.modules["chromadb"]
            if "chromadb.config" in sys.modules:
                del sys.modules["chromadb.config"]
