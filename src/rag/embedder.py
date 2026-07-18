"""Embedding model wrapper for RAG system.

Uses all-MiniLM-L6-v2 by default — a lightweight, fast model
that performs well for semantic search on educational content.
"""

from typing import List, Optional


class Embedder:
    """Wraps a sentence-transformer model for embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedder.

        Args:
            model_name: HuggingFace sentence-transformers model name.
        """
        self.model_name = model_name
        self._model: Optional[object] = None

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each is List[float]).
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query string.

        Args:
            query: Query text.

        Returns:
            Embedding vector as List[float].
        """
        return self.embed([query])[0]
