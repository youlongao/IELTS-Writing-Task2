"""RAG retriever for the IELTS writing reference library."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .embedder import Embedder
from .loader import KnowledgeLoader
from .store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates retrieval across the writing reference collections."""

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        topic: Optional[str] = None,
        question_type: Optional[str] = None,
        top_k: int = 3,
        collections: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        if collections is None:
            collections = self.store.COLLECTIONS

        enriched_query = query
        if topic:
            enriched_query = f"{topic} {enriched_query}"
        if question_type:
            enriched_query = f"{question_type} {enriched_query}"

        query_embedding = self.embedder.embed_query(enriched_query)
        results: Dict[str, List[Dict]] = {}

        for collection_name in collections:
            try:
                raw = self.store.query(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    n_results=top_k,
                )
                items = []
                for i, document in enumerate(raw["documents"]):
                    items.append({
                        "content": document,
                        "metadata": raw["metadatas"][i] if i < len(raw["metadatas"]) else {},
                        "distance": raw["distances"][i] if i < len(raw["distances"]) else 0.0,
                    })
                results[collection_name] = items
            except Exception as exc:
                logger.warning("Error querying collection '%s': %s", collection_name, exc)
                results[collection_name] = []

        return results

    def format_context(self, results: Dict[str, List[Dict]], max_chars: int = 2200) -> str:
        labels = {
            "rubrics": "IELTS Writing Task 2 Assessment Criteria",
            "structures": "Essay Structure Reference",
            "topics": "High-Frequency Topic Ideas",
            "vocabulary": "Band 7 Vocabulary and Collocations",
            "sample_essays": "Sample Paragraph Logic",
        }

        sections = []
        total_chars = 0

        for collection_name in ["rubrics", "structures", "topics", "vocabulary", "sample_essays"]:
            items = results.get(collection_name, [])
            if not items:
                continue

            lines = [f"\n### {labels.get(collection_name, collection_name)}\n"]
            for item in items:
                content = item.get("content", "")
                if total_chars + len(content) > max_chars:
                    remaining = max_chars - total_chars
                    if remaining > 100:
                        lines.append(content[:remaining] + "...")
                    break
                lines.append(content)
                lines.append("---")
                total_chars += len(content)

            if len(lines) > 1:
                sections.append("\n".join(lines))

        if not sections:
            return "No relevant writing reference was found. Use general IELTS Task 2 standards."
        return "\n".join(sections)

    def build_knowledge_base(self, loader: KnowledgeLoader):
        logger.info("Building writing reference store from source files")
        all_data = loader.load_all()

        for collection_name, documents in all_data.items():
            if not documents:
                logger.warning("No documents found for collection: %s", collection_name)
                continue

            self.store.clear_collection(collection_name)
            texts = [document[0] for document in documents]
            metadatas = [document[1] for document in documents]
            ids = [f"{collection_name}_{i}" for i in range(len(texts))]

            logger.info("Embedding %s documents for '%s'", len(texts), collection_name)
            embeddings = self.embedder.embed(texts)
            self.store.add_documents(
                collection_name=collection_name,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            logger.info("Loaded %s documents into '%s'", len(texts), collection_name)

        logger.info("Writing reference store build complete")


def main():
    """CLI entry point for building and inspecting the reference store."""
    import argparse

    from ..utils.config import Config

    parser = argparse.ArgumentParser(description="Build or inspect the IELTS reference store.")
    parser.add_argument("--build", action="store_true", help="Rebuild the vector store from data/knowledge_base.")
    parser.add_argument("--count", action="store_true", help="Print document counts per collection.")
    args = parser.parse_args()

    config = Config()
    embedder = Embedder(config.get("rag.embedding_model", "all-MiniLM-L6-v2"))
    store = VectorStore(config.chroma_persist_dir)
    retriever = Retriever(embedder, store)

    if args.build:
        loader = KnowledgeLoader(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        retriever.build_knowledge_base(loader)

    if args.count or not args.build:
        for collection_name in store.COLLECTIONS:
            print(f"{collection_name}: {store.count(collection_name)}")


if __name__ == "__main__":
    main()
