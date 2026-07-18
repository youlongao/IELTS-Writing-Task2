"""RAG package for IELTS writing knowledge retrieval.

Heavy dependencies (sentence-transformers, chromadb) are lazily imported
at the source module level, so importing this package is always safe.
"""

from .embedder import Embedder
from .store import VectorStore
from .loader import KnowledgeLoader
from .retriever import Retriever

__all__ = ["Embedder", "VectorStore", "KnowledgeLoader", "Retriever"]
