"""ChromaDB vector store management."""

import os
from typing import Dict, List, Optional


class VectorStore:
    """Manages ChromaDB vector store for IELTS knowledge retrieval.

    Collections:
    - topics: Topic knowledge documents
    - vocabulary: Collocations and phrases
    - structures: Essay structure patterns
    - rubrics: IELTS assessment criteria
    - sample_essays: Band 7+ sample paragraphs
    """

    COLLECTIONS = ["topics", "vocabulary", "structures", "rubrics", "sample_essays"]

    def __init__(self, persist_dir: str):
        """Initialize the vector store.

        Args:
            persist_dir: Directory for ChromaDB persistence.
        """
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self._client = None

    @property
    def client(self):
        """Lazy-load the ChromaDB PersistentClient."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def get_or_create_collection(self, name: str):
        """Get an existing collection or create a new one.

        Args:
            name: Collection name.

        Returns:
            ChromaDB collection object.

        Raises:
            ValueError: If name is not a valid collection name.
        """
        if name not in self.COLLECTIONS:
            raise ValueError(
                f"Unknown collection: '{name}'. Valid: {self.COLLECTIONS}"
            )

        try:
            return self.client.get_collection(name)
        except Exception:
            return self.client.create_collection(name)

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None,
    ):
        """Add documents to a collection.

        Args:
            collection_name: Target collection name.
            documents: List of document texts.
            embeddings: List of embedding vectors.
            metadatas: List of metadata dicts.
            ids: Optional list of document IDs. Auto-generated if None.
        """
        collection = self.get_or_create_collection(collection_name)

        if ids is None:
            ids = [f"{collection_name}_{i}" for i in range(len(documents))]

        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 3,
    ) -> Dict:
        """Query a collection for similar documents.

        Args:
            collection_name: Collection to search.
            query_embedding: Query embedding vector.
            n_results: Number of results to return.

        Returns:
            Dict with 'documents', 'metadatas', 'distances' lists.
        """
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return {
            "documents": results.get("documents", [[]])[0],
            "metadatas": results.get("metadatas", [[]])[0],
            "distances": results.get("distances", [[]])[0],
        }

    def count(self, collection_name: str) -> int:
        """Get document count in a collection.

        Args:
            collection_name: Collection name.

        Returns:
            Number of documents.
        """
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    def clear_collection(self, collection_name: str):
        """Delete all documents from a collection.

        Args:
            collection_name: Collection to clear.
        """
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

    def reset(self):
        """Reset all collections."""
        for name in self.COLLECTIONS:
            self.clear_collection(name)
